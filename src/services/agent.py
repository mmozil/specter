"""Agente Matt Murdock — especialista tributário brasileiro com Pydantic AI."""
import logging
import re
import time
import uuid
from typing import Optional

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.models.tables import Conversation, Message
from src.services.memory import (
    get_or_create_profile,
    process_message_for_profile,
    build_profile_context,
)
from src.services.learning import learn_from_engaged_conversation
from src.tools.tools import (
    MurdockDeps,
    search_law,
    calculate_tax,
    check_ncm,
    reform_2026,
    credit_recovery,
    calendar,
    jurisprudence,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# System Prompt — Matt Murdock, Tributarista Brasileiro
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Você é **Matt Murdock** — tributarista, contabilista e fiscalista brasileiro de elite. Assim como o advogado Daredevil, você tem percepção sobre-humana para detectar riscos ocultos e oportunidades que passam despercebidas.

## Tom e Estilo — REGRAS OBRIGATÓRIAS

**Você entrega como um parecer de consultoria tributária de alto nível — completo, estruturado e fundamentado, como uma Big Four entrega a um cliente exigente.**

1. **Profundidade com estrutura.** Respostas completas e bem organizadas, com seções, tabelas e cálculo demonstrado. Seja detalhista quando a questão tributária exige — mas SEMPRE com estrutura (headers, tabelas, listas), NUNCA em parágrafos longos e corridos.
2. **Toda resposta tributária segue o Formato de Resposta abaixo** (Enquadramento → Tributos → Cálculo → Riscos → Recomendação). Adapte: perguntas simples usam só as seções relevantes; perguntas complexas usam todas.
3. **Sempre demonstre o cálculo** passo a passo (base, alíquota, deduções, resultado). **Sempre cite a base legal específica** (artigo, lei, tema STF).
4. **Sempre sinalize riscos** (multa, autuação, perda de crédito) e **recomende ação prática** quando aplicável.
5. **Nunca repita conteúdo que já está no histórico da conversa.** Se já explicou algo, referencie ("como mencionei acima") — não copie de novo.
6. **Leia o tom do usuário.** Se ele está frustrado ou direto, releia o histórico, identifique o que ele quer e responda diretamente — sem pedir pra reformular.
7. **Nunca diga "não tenho histórico" ou "cada interação começa do zero".** Você TEM acesso ao histórico da conversa atual. Use-o.
8. **Zero auto-explicação.** Nunca explique o que você é, como funciona, ou o que suas ferramentas fazem — a menos que perguntem.
9. **Sem pedidos de desculpa vazios.** Se errou, corrija uma vez e siga em frente.

## Missão

- Orientar com precisão técnica — fundamento legal sempre, achismo nunca.
- Proteger o contribuinte — riscos fiscais e obrigações negligenciadas antes do auto de infração.
- Otimizar carga tributária dentro da lei — elisão é dever, evasão é crime.
- Tornar o complexo executável — decisão prática sem perder rigor.

## Ferramentas

Use suas 7 ferramentas quando a pergunta exigir (search_law, calculate_tax, check_ncm, reform_2026, credit_recovery, calendar, jurisprudence). Integre resultados naturalmente — não anuncie que vai usar uma ferramenta.

**REGRA DURA — cálculo:** para QUALQUER cálculo de DAS, imposto, alíquota efetiva ou comparação de regime, você DEVE chamar `calculate_tax`. NUNCA calcule de cabeça nem invente alíquota/anexo/parcela a deduzir — o retorno da tool é a única fonte da verdade do número. Ao chamar, passe a `atividade` correta (o anexo do Simples é derivado dela automaticamente):
- comércio / revenda / varejo / e-commerce de produtos → `atividade="comercio"` (Anexo I)
- fábrica / produção / industrialização → `atividade="industria"` (Anexo II)
- prestação de serviço → `atividade="servicos"` (Anexo III/IV/V conforme fator R)
- Faturamento dos últimos 12 meses vai em `rbt12`; a receita do mês corrente vai em `receita_mensal`.

## Regras Técnicas

### Informações necessárias
Antes de responder sobre tributação, idealmente você precisa: regime tributário, CNAE, UF origem/destino, tipo de operação, NCM/NBS, faturamento 12 meses.
- Se faltarem 3+ dessas e a resposta DEPENDER delas → pergunte de forma concisa (1-2 linhas, não um formulário de 6 itens).
- Se o usuário já forneceu parte das informações no histórico → USE-AS, não peça de novo.
- Se der para responder com o que tem (resposta genérica mas útil) → responda e indique que pode refinar com mais dados.

### Esferas tributárias
Federal (IRPJ, CSLL, PIS, Cofins, IPI, IRRF) · Estadual (ICMS, ITCMD) · Municipal (ISS, ITBI). Sempre especifique tributo, base e alíquota.

### Hierarquia de fontes
CF/88 → CTN → LCs (87, 116, 123, 214) → Leis Ordinárias → Decretos → INs RFB → COSIT → CONFAZ → CGSN

### Simplificações proibidas
- "MEI paga só DAS" (errado com ICMS-ST/DIFAL)
- "Simples é sempre mais barato" (depende de faturamento, atividade, fator R)
- "ICMS é 18%" (depende de UF, NCM, operação, benefício)
- "PIS/Cofins é 3,65%" (errado no não-cumulativo: 1,65% + 7,6%)
- "Lucro Presumido é viável acima de R$78M/ano" (PROIBIDO — acima de R$78M/ano é Lucro Real OBRIGATÓRIO, Lei 9.718/98 art. 14)

## Formato de Resposta

Português brasileiro (pt-BR), **Markdown rico** (headers `##`/`###`, tabelas, **bold**, listas).

Estrutura padrão (adapte ao que a pergunta exige — não force seções vazias):

```
### Enquadramento
Regime · atividade/CNAE · tipo de operação · UF(s) — quando relevante

### Tributos e incidências
| Tributo | Esfera | Base | Alíquota | Valor |
|---------|--------|------|----------|-------|

### Cálculo
Passo a passo com fórmula e números reais

### Riscos e pontos de atenção
Cada risco: consequência + penalidade concreta + como evitar

### Recomendação
Ação imediata + planejamento, quando aplicável
```

Regras de formatação:
- Sempre cite fonte legal específica (ex: "Art. 13, LC 123/2006", "Lei 9.718/98 art. 14")
- Comparativos SEMPRE em tabela Markdown
- Valores monetários e alíquotas em **bold**; conclusões-chave destacadas
- Cálculo SEMPRE demonstrado (nunca só o resultado final)

## Referência 2026

Salário mínimo R$1.518 · Teto INSS R$8.475,55 · MEI R$81k/ano · Simples R$4,8M/ano · Sublimite ICMS/ISS R$3,6M · **Lucro Presumido até R$78M/ano (acima = Lucro Real obrigatório)** · IRPF isento até R$5k/mês · Dividendos isentos até R$50k/mês, 10% acima · CBS 0,9% + IBS 0,1% (teste 2026, LC 214/2025)

## Personalidade

Direto. Preciso. Confiante. Como um advogado tributarista sênior que cobra R$2.000/hora — cada palavra tem valor. Quando não souber, diga em uma linha e indique onde buscar. Quando identificar risco fiscal, alerte sem rodeio. Quando encontrar oportunidade de economia, destaque com convicção.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Agente Pydantic AI
# ═══════════════════════════════════════════════════════════════════════════

# Modelo primário construído 100% a partir do ambiente (zero hardcode).
# Provider OpenAI-compatible: MiniMax (default), OpenRouter, OpenAI ou endpoint local.
primary_model = OpenAIModel(
    settings.DEFAULT_LLM_MODEL,
    provider=OpenAIProvider(
        base_url=settings.DEFAULT_LLM_BASE_URL,
        api_key=settings.DEFAULT_LLM_API_KEY or "no-key",
    ),
)

murdock_agent = Agent(
    primary_model,
    deps_type=MurdockDeps,
    system_prompt=SYSTEM_PROMPT,
    tools=[
        search_law,
        calculate_tax,
        check_ncm,
        reform_2026,
        credit_recovery,
        calendar,
        jurisprudence,
    ],
    retries=2,
)

# Client ID default para sessões anônimas (será substituído por auth futuramente)
DEFAULT_CLIENT_ID = "anonymous"


# ── Filtro de raciocínio ─────────────────────────────────────────────────────
# Modelos de raciocínio (MiniMax-M2 etc.) emitem o pensamento em <think>...</think>
# dentro do próprio conteúdo. Nunca deve vazar pro usuário.

def _strip_reasoning(text: str) -> str:
    """Remove blocos <think>...</think> da saída final."""
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


async def _stream_filtered(result, full_response: list):
    """Streama os deltas de texto suprimindo o bloco <think>...</think>.

    `full_response` acumula o texto cru (pra salvar/limpar depois).
    """
    emit_buffer = ""
    state = None  # None=indeciso, True=dentro do <think>, False=já liberou
    async for chunk in result.stream_text(delta=True):
        full_response.append(chunk)
        if state is False:
            yield chunk
            continue
        emit_buffer += chunk
        if state is None:
            ls = emit_buffer.lstrip()
            if not ls:
                continue
            if ls.startswith("<think>"):
                state = True
            elif "<think>".startswith(ls[: len("<think>")]):
                continue  # prefixo parcial ("<th"), aguarda mais chunks
            else:
                state = False  # resposta sem bloco de raciocínio
                yield emit_buffer
                emit_buffer = ""
                continue
        if state is True and "</think>" in emit_buffer:
            after = emit_buffer.split("</think>", 1)[1].lstrip("\n")
            state = False
            emit_buffer = ""
            if after:
                yield after


# ═══════════════════════════════════════════════════════════════════════════
# Serviço de Conversa
# ═══════════════════════════════════════════════════════════════════════════

async def get_or_create_conversation(
    db: AsyncSession, conversation_id: Optional[str] = None
) -> Conversation:
    """Obtém ou cria uma conversa."""
    if conversation_id:
        conv = (await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )).scalar_one_or_none()
        if conv:
            return conv

    conv = Conversation(title="Nova consulta tributária")
    db.add(conv)
    await db.flush()
    return conv


async def load_history(db: AsyncSession, conversation_id: str) -> list[dict]:
    """Carrega histórico de mensagens para contexto."""
    msgs = (await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )).scalars().all()

    return [{"role": m.role, "content": m.content} for m in msgs]


async def save_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    sources_used: list = None,
    tools_called: list = None,
    model_used: str = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    latency_ms: int = 0,
) -> Message:
    """Salva mensagem no banco."""
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources_used=sources_used,
        tools_called=tools_called,
        model_used=model_used,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
    )
    db.add(msg)
    return msg


def _build_message_history(history: list[dict]) -> list[ModelRequest | ModelResponse]:
    """Converte histórico do banco em objetos Pydantic AI (exclui a última msg do user, que vai como prompt)."""
    messages = []
    # Pegar até as últimas 20 mensagens (10 pares user/assistant), excluindo a última (user atual)
    recent = history[-21:-1] if len(history) > 1 else []
    for msg in recent:
        if msg["role"] == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=msg["content"])]))
        elif msg["role"] == "assistant":
            messages.append(ModelResponse(parts=[TextPart(content=msg["content"])]))
    return messages


async def _build_dynamic_prompt(db: AsyncSession, client_id: str) -> str:
    """Constrói prompt adicional com contexto do cliente."""
    profile = await get_or_create_profile(db, client_id)
    profile_ctx = build_profile_context(profile)
    if profile_ctx:
        return f"\n\n{profile_ctx}"
    return ""


async def chat(
    db: AsyncSession,
    user_message: str,
    conversation_id: Optional[str] = None,
    client_id: str = None,
) -> dict:
    """Processa mensagem e retorna resposta completa (sem streaming)."""
    start = time.time()
    client_id = client_id or DEFAULT_CLIENT_ID

    # Conversa
    conv = await get_or_create_conversation(db, conversation_id)

    # Extrair dados de perfil da mensagem do usuário (assíncrono, não bloqueia)
    await process_message_for_profile(db, client_id, user_message)

    # Salvar mensagem do usuário
    await save_message(db, str(conv.id), "user", user_message)

    # Carregar histórico e construir message_history para o agente
    history = await load_history(db, str(conv.id))
    message_history = _build_message_history(history)

    # Prompt dinâmico com perfil do cliente
    dynamic_ctx = await _build_dynamic_prompt(db, client_id)
    full_prompt = user_message + dynamic_ctx if dynamic_ctx else user_message

    # Rodar agente
    deps = MurdockDeps(db=db)
    model_name = settings.DEFAULT_LLM_MODEL
    fallback_model = f"{settings.FALLBACK_LLM_PROVIDER}:{settings.FALLBACK_LLM_MODEL}"

    try:
        result = await murdock_agent.run(
            full_prompt, deps=deps, message_history=message_history
        )
        model_used = model_name
    except Exception as e:
        logger.warning(f"MiniMax falhou ({e}), tentando fallback Claude...")
        try:
            result = await murdock_agent.run(
                full_prompt, deps=deps, model=fallback_model,
                message_history=message_history,
            )
            model_used = fallback_model
        except Exception as e2:
            logger.error(f"Fallback também falhou: {e2}")
            raise

    response_text = _strip_reasoning(result.output)
    latency = int((time.time() - start) * 1000)

    # Salvar resposta
    await save_message(
        db, str(conv.id), "assistant", response_text,
        model_used=model_used,
        latency_ms=latency,
    )

    # Atualizar conversa
    conv.total_messages = (conv.total_messages or 0) + 2
    conv.model_used = model_used

    # Título automático na primeira interação
    if conv.total_messages <= 2:
        conv.title = user_message[:100]

    # Learning loop: aprender de conversas engajadas (6+ msgs)
    if (conv.total_messages or 0) >= 6 and (conv.total_messages or 0) % 6 == 0:
        try:
            await learn_from_engaged_conversation(db, str(conv.id))
        except Exception as e:
            logger.warning(f"Learning loop falhou: {e}")

    await db.commit()

    return {
        "conversation_id": str(conv.id),
        "response": response_text,
        "model": model_used,
        "latency_ms": latency,
    }


async def chat_stream(
    db: AsyncSession,
    user_message: str,
    conversation_id: Optional[str] = None,
    client_id: str = None,
):
    """Processa mensagem com streaming SSE (yield de chunks)."""
    start = time.time()
    client_id = client_id or DEFAULT_CLIENT_ID

    conv = await get_or_create_conversation(db, conversation_id)

    # Extrair dados de perfil da mensagem do usuário
    await process_message_for_profile(db, client_id, user_message)

    await save_message(db, str(conv.id), "user", user_message)

    # Carregar histórico e construir message_history para o agente
    history = await load_history(db, str(conv.id))
    message_history = _build_message_history(history)

    # Prompt dinâmico com perfil do cliente
    dynamic_ctx = await _build_dynamic_prompt(db, client_id)
    full_prompt = user_message + dynamic_ctx if dynamic_ctx else user_message

    deps = MurdockDeps(db=db)
    model_name = settings.DEFAULT_LLM_MODEL
    fallback_model = f"{settings.FALLBACK_LLM_PROVIDER}:{settings.FALLBACK_LLM_MODEL}"

    full_response = []
    model_used = model_name

    try:
        async with murdock_agent.run_stream(
            full_prompt, deps=deps, message_history=message_history
        ) as result:
            async for piece in _stream_filtered(result, full_response):
                yield {"event": "token", "data": piece}
    except Exception as e:
        logger.warning(f"MiniMax stream falhou ({e}), tentando fallback...")
        model_used = fallback_model
        full_response = []
        try:
            async with murdock_agent.run_stream(
                full_prompt, deps=deps, model=fallback_model,
                message_history=message_history,
            ) as result:
                async for piece in _stream_filtered(result, full_response):
                    yield {"event": "token", "data": piece}
        except Exception as e2:
            logger.error(f"Fallback stream falhou: {e2}")
            yield {"event": "error", "data": f"Erro: {e2}"}
            return

    response_text = _strip_reasoning("".join(full_response))
    latency = int((time.time() - start) * 1000)

    # Salvar resposta
    await save_message(
        db, str(conv.id), "assistant", response_text,
        model_used=model_used,
        latency_ms=latency,
    )

    conv.total_messages = (conv.total_messages or 0) + 2
    conv.model_used = model_used
    if (conv.total_messages or 0) <= 2:
        conv.title = user_message[:100]

    # Learning loop: aprender de conversas engajadas (6+ msgs)
    if (conv.total_messages or 0) >= 6 and (conv.total_messages or 0) % 6 == 0:
        try:
            await learn_from_engaged_conversation(db, str(conv.id))
        except Exception as e:
            logger.warning(f"Learning loop falhou: {e}")

    await db.commit()

    yield {
        "event": "done",
        "data": f'{{"conversation_id": "{conv.id}", "model": "{model_used}", "latency_ms": {latency}}}',
    }
