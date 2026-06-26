"""Configuração de LLM env-based + DB-driven (padrão Tier Agent).

O usuário escolhe quais LLMs usar, em que ordem, e liga/desliga cada uma — tudo
persistido em `llm_providers` (sem redeploy). O agente percorre os provedores ATIVOS
por `priority` (menor = primeiro) e faz fallback pro próximo se um falhar.

- Provedores OpenAI-compatible (minimax, openai, openrouter, deepseek, gemini, nous,
  local) → `OpenAIModel(model, provider=OpenAIProvider(base_url, api_key))`.
- `anthropic` → `AnthropicModel` nativo (import defensivo); se indisponível, cai pro
  inference string `anthropic:<model>` (usa `ANTHROPIC_API_KEY` do ambiente).

Semântica de desligamento (igual Tier Agent): se há provedores cadastrados mas TODOS
desativados → `ProvidersAllDisabled` (o agente fica em silêncio, respeita o off). Se não
há nenhum cadastrado → `NoProvidersConfigured` (o caller usa a rede de segurança do env).
"""
import hashlib
import logging
from typing import Any

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.encryption import decrypt, encrypt
from src.models.tables import LlmProvider

logger = logging.getLogger(__name__)


# ── Catálogo (parametrizável; o frontend consome via GET /supported) ──────────

SUPPORTED_PROVIDERS: dict[str, str] = {
    "minimax": "MiniMax (MiniMax-M2)",
    "anthropic": "Anthropic Claude (haiku/sonnet/opus)",
    "openai": "OpenAI (gpt-4o, gpt-4.1, o-series)",
    "gemini": "Google Gemini (2.5 flash/pro) — via endpoint OpenAI-compatible",
    "openrouter": "OpenRouter (300+ modelos via 1 chave)",
    "deepseek": "DeepSeek (deepseek-chat, deepseek-reasoner)",
    "nous": "Nous Research (Hermes)",
    "local": "Endpoint OpenAI-compatible custom (Ollama/vLLM/LM Studio)",
}

# Base URL default por provider (sobreescrevível por registro via `base_url`).
DEFAULT_BASE_URL: dict[str, str | None] = {
    "minimax": "https://api.minimax.io/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "nous": "https://inference-api.nousresearch.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "local": None,
}

# Modelos sugeridos por provider (popula o dropdown da UI — não trava a escolha).
DEFAULT_PROVIDER_MODELS: dict[str, list[str]] = {
    "minimax": ["MiniMax-M2"],
    "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-20250514", "claude-sonnet-4-6", "claude-opus-4-7"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    "openrouter": [
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-haiku-4.5",
        "google/gemini-2.5-flash",
        "deepseek/deepseek-chat",
        "openai/gpt-4o-mini",
    ],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "nous": ["Hermes-3-Llama-3.1-70B"],
    "local": [],
}

# Selo de qualidade pro caso de uso tributário (curadoria, não trava). `experimental` = resto.
MODEL_TIERS: dict[str, str] = {
    "minimax-m2": "recommended",
    "claude-sonnet-4-20250514": "tested",
    "claude-sonnet-4-6": "recommended",
    "anthropic/claude-sonnet-4.5": "recommended",
    "gpt-4o": "recommended",
    "gpt-4.1": "recommended",
    "gemini-2.5-pro": "recommended",
    "deepseek-reasoner": "recommended",
}
MODEL_TIER_LABELS = {
    "tested": "Validado nos nossos testes de qualidade tributária.",
    "recommended": "Bom para raciocínio fiscal/jurídico complexo.",
    "experimental": "Funciona, mas ainda não passou pela nossa bateria.",
}

# Provedores que falam o protocolo OpenAI Chat Completions.
_OPENAI_COMPATIBLE = {"minimax", "openai", "gemini", "openrouter", "deepseek", "nous", "local"}


def model_tier(name: str) -> str:
    return MODEL_TIERS.get((name or "").strip().lower(), "experimental")


# ── Exceções de controle ──────────────────────────────────────────────────────

class ProvidersAllDisabled(RuntimeError):
    """Há provedores cadastrados, mas todos estão desativados — respeitar o off."""


class NoProvidersConfigured(RuntimeError):
    """Nenhum provedor cadastrado — caller deve usar a rede de segurança do env."""


# ── Cache de modelos construídos (reusa clients httpx; invalida em mudança) ────

_MODEL_CACHE: dict[str, Any] = {}


def bump_version() -> None:
    """Invalida o cache de modelos — chamar após qualquer create/update/delete."""
    _MODEL_CACHE.clear()


def _label(p: LlmProvider) -> str:
    return p.label or f"{p.provider}/{p.default_model}"


def _construct_model(provider: str, model: str, base_url: str | None, api_key: str):
    """Constrói o objeto de modelo do pydantic-ai pro provider."""
    if provider == "anthropic":
        if api_key:
            try:
                from pydantic_ai.models.anthropic import AnthropicModel
                from pydantic_ai.providers.anthropic import AnthropicProvider

                return AnthropicModel(model, provider=AnthropicProvider(api_key=api_key))
            except Exception as e:  # noqa: BLE001
                logger.warning("AnthropicModel indisponível (%s) — usando inference string", e)
        # Sem key própria ou import falhou: inference string usa ANTHROPIC_API_KEY do env.
        return f"anthropic:{model}"

    # OpenAI-compatible (default)
    eff_base = (base_url or DEFAULT_BASE_URL.get(provider) or "").rstrip("/") or None
    return OpenAIModel(
        model,
        provider=OpenAIProvider(base_url=eff_base, api_key=api_key or "no-key"),
    )


def build_model(p: LlmProvider) -> tuple[Any, dict, str]:
    """Retorna `(model_obj, model_settings, label)` pro registro, com cache."""
    api_key = decrypt(p.api_key_enc)
    base = p.base_url or DEFAULT_BASE_URL.get(p.provider)
    msettings = {"temperature": float(p.temperature), "max_tokens": int(p.max_tokens)}
    cache_key = (
        f"{p.id}:{p.provider}:{p.default_model}:{base}:"
        f"{hashlib.sha1((api_key or '').encode()).hexdigest()[:8]}"
    )
    model_obj = _MODEL_CACHE.get(cache_key)
    if model_obj is None:
        model_obj = _construct_model(p.provider, p.default_model, base, api_key)
        _MODEL_CACHE[cache_key] = model_obj
    return model_obj, msettings, _label(p)


# ── Carga / resolução ──────────────────────────────────────────────────────────

async def load_active(db: AsyncSession) -> list[LlmProvider]:
    rows = (
        await db.execute(
            select(LlmProvider)
            .where(LlmProvider.active.is_(True))
            .order_by(LlmProvider.priority.asc(), LlmProvider.id.asc())
        )
    ).scalars().all()
    return list(rows)


async def has_any(db: AsyncSession) -> bool:
    row = (await db.execute(select(LlmProvider.id).limit(1))).scalars().first()
    return row is not None


async def build_chain(db: AsyncSession) -> list[tuple[Any, dict, str]]:
    """Cadeia ordenada de modelos ativos. Levanta as exceções de controle nos vazios."""
    rows = await load_active(db)
    if not rows:
        if await has_any(db):
            raise ProvidersAllDisabled("Todos os provedores de LLM estão desativados")
        raise NoProvidersConfigured("Nenhum provedor de LLM cadastrado")
    return [build_model(p) for p in rows]


# ── Seed a partir do ambiente (zero regressão no 1º boot) ──────────────────────

async def seed_from_env_if_empty(db: AsyncSession) -> None:
    """Se a tabela está vazia, cria o provider primário (+ fallback) a partir das envs.

    Preserva exatamente o comportamento atual (MiniMax-M2 primário, Claude fallback) na
    primeira subida com a feature ligada. Idempotente: não roda se já houver registros.
    """
    if await has_any(db):
        return

    db.add(
        LlmProvider(
            provider=settings.DEFAULT_LLM_PROVIDER,
            default_model=settings.DEFAULT_LLM_MODEL,
            base_url=settings.DEFAULT_LLM_BASE_URL or None,
            api_key_enc=encrypt(settings.DEFAULT_LLM_API_KEY or ""),
            temperature=0.3,
            active=True,
            priority=10,
            label="Primário (env)",
        )
    )

    if settings.FALLBACK_LLM_PROVIDER and settings.FALLBACK_LLM_MODEL:
        fb_key = settings.ANTHROPIC_API_KEY if settings.FALLBACK_LLM_PROVIDER == "anthropic" else ""
        db.add(
            LlmProvider(
                provider=settings.FALLBACK_LLM_PROVIDER,
                default_model=settings.FALLBACK_LLM_MODEL,
                base_url=None,
                api_key_enc=encrypt(fb_key or ""),
                temperature=0.3,
                active=True,
                priority=20,
                label="Fallback (env)",
            )
        )

    await db.commit()
    logger.info("llm_config: seed inicial de provedores criado a partir do ambiente")
