"""Rotas admin pra configurar provedores de LLM (escolha, ordem, on/off).

Padrão Tier Agent: o usuário cadastra provider + modelo + API key + base_url, define a
ordem (priority) e liga/desliga (active). API key é Fernet-encrypted no DB. Endpoints
protegidos pela mesma API key admin do Murdock (`X-API-Key`; liberado em dev).
"""
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes import _check_api_key
from src.core.database import get_db
from src.core.encryption import decrypt, encrypt
from src.models.tables import LlmProvider
from src.services import llm_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm-providers", tags=["llm"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class ProviderIn(BaseModel):
    provider: str
    default_model: str
    api_key: str = ""          # plaintext na request, encrypted at rest
    base_url: str | None = None
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout_s: int = 60
    active: bool = True
    priority: int = 100
    label: str | None = None


class ProviderPatch(BaseModel):
    """Patch parcial — tudo opcional (toggle, reorder, edição pontual)."""

    provider: str | None = None
    default_model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_s: int | None = None
    active: bool | None = None
    priority: int | None = None
    label: str | None = None


def _to_out(p: LlmProvider, in_use: bool = False) -> dict:
    raw = ""
    try:
        raw = decrypt(p.api_key_enc) or ""
    except Exception:
        raw = ""
    return {
        "id": p.id,
        "provider": p.provider,
        "default_model": p.default_model,
        "base_url": p.base_url,
        "temperature": p.temperature,
        "max_tokens": p.max_tokens,
        "timeout_s": p.timeout_s,
        "active": p.active,
        "priority": p.priority,
        "label": p.label,
        "has_api_key": bool(raw),
        "api_key_suffix": raw[-4:] if len(raw) >= 4 else ("••••" if raw else None),
        "in_use": in_use,
        "tier": llm_config.model_tier(p.default_model),
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }


# ── Catálogo ─────────────────────────────────────────────────────────────────

@router.get("/supported")
async def supported_providers():
    """Provedores suportados + modelos sugeridos por provider (popula a UI)."""
    return {
        "providers": [
            {
                "key": k,
                "label": v,
                "default_base_url": llm_config.DEFAULT_BASE_URL.get(k),
                "models": [
                    {"name": m, "tier": llm_config.model_tier(m)}
                    for m in llm_config.DEFAULT_PROVIDER_MODELS.get(k, [])
                ],
            }
            for k, v in llm_config.SUPPORTED_PROVIDERS.items()
        ],
        "tier_labels": llm_config.MODEL_TIER_LABELS,
    }


# ── CRUD ─────────────────────────────────────────────────────────────────────

@router.get("")
async def list_providers(
    db: AsyncSession = Depends(get_db),
    _auth: bool = Depends(_check_api_key),
):
    """Lista provedores ordenados por priority (mesma ordem que o motor usa).

    Marca `in_use=True` no ativo de menor priority (o que o agente realmente pega primeiro).
    """
    rows = (
        await db.execute(
            select(LlmProvider).order_by(LlmProvider.priority.asc(), LlmProvider.id.asc())
        )
    ).scalars().all()
    in_use_id = next((r.id for r in rows if r.active), None)
    return [_to_out(r, in_use=(r.id == in_use_id)) for r in rows]


@router.post("", status_code=201)
async def create_provider(
    payload: ProviderIn,
    db: AsyncSession = Depends(get_db),
    _auth: bool = Depends(_check_api_key),
):
    if payload.provider not in llm_config.SUPPORTED_PROVIDERS:
        raise HTTPException(400, f"Provider não suportado. Use um de: {list(llm_config.SUPPORTED_PROVIDERS)}")
    item = LlmProvider(
        provider=payload.provider,
        default_model=payload.default_model,
        api_key_enc=encrypt(payload.api_key),
        base_url=payload.base_url or None,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        timeout_s=payload.timeout_s,
        active=payload.active,
        priority=payload.priority,
        label=payload.label,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    llm_config.bump_version()
    return _to_out(item)


@router.patch("/{provider_id}")
async def update_provider(
    provider_id: int,
    payload: ProviderPatch,
    db: AsyncSession = Depends(get_db),
    _auth: bool = Depends(_check_api_key),
):
    item = await db.get(LlmProvider, provider_id)
    if not item:
        raise HTTPException(404, "Provider não encontrado")
    data = payload.model_dump(exclude_unset=True)
    if "provider" in data and data["provider"] not in llm_config.SUPPORTED_PROVIDERS:
        raise HTTPException(400, "Provider não suportado")
    if "api_key" in data:
        new_key = data.pop("api_key")
        if new_key:  # só sobrescreve se veio uma key nova não-vazia (não apaga sem querer)
            item.api_key_enc = encrypt(new_key)
    for k, v in data.items():
        setattr(item, k, v)
    await db.commit()
    await db.refresh(item)
    llm_config.bump_version()
    return _to_out(item)


@router.delete("/{provider_id}", status_code=204)
async def delete_provider(
    provider_id: int,
    db: AsyncSession = Depends(get_db),
    _auth: bool = Depends(_check_api_key),
):
    item = await db.get(LlmProvider, provider_id)
    if not item:
        raise HTTPException(404, "Provider não encontrado")
    await db.delete(item)
    await db.commit()
    llm_config.bump_version()


@router.post("/{provider_id}/test")
async def test_provider(
    provider_id: int,
    db: AsyncSession = Depends(get_db),
    _auth: bool = Depends(_check_api_key),
):
    """Ping real no provider (sem tools, sem fallback) — veredito honesto da credencial."""
    from pydantic_ai import Agent

    from src.services.agent import _strip_reasoning

    item = await db.get(LlmProvider, provider_id)
    if not item:
        raise HTTPException(404, "Provider não encontrado")

    model_obj, msettings, label = llm_config.build_model(item)
    started = time.perf_counter()
    try:
        probe = Agent(model_obj, system_prompt="Você é um teste de conexão. Responda só 'ok'.")
        result = await probe.run("ping", model_settings=msettings)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": True,
            "provider": item.provider,
            "model": item.default_model,
            "latency_ms": latency_ms,
            "sample": _strip_reasoning(result.output)[:80],
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "provider": item.provider,
            "model": item.default_model,
            "detail": str(e)[:300],
        }
