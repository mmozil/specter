"""Rotas da API do Murdock."""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.database import get_db
from src.api.schemas import (
    ChatRequest, ChatResponse, IngestRequest, IngestResponse,
    FeedbackRequest, HealthResponse,
)
from sqlalchemy import select as sa_select

from src.services.agent import chat, chat_stream
from src.crawler.ingest import ingest_fonte, ingest_todas, get_status, update_search_vectors
from src.models.tables import Conversation, Message, Feedback as FeedbackModel, User
from src.core.security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Auth helper ────────────────────────────────────────────────────────────

def _check_api_key(x_api_key: str = Header(None)):
    """Valida API key para endpoints administrativos."""
    if settings.ENVIRONMENT == "development":
        return True
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="API key inválida")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Chat
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/chat", response_model=None)
async def chat_endpoint(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Chat com o Specter — streaming SSE ou resposta completa. Requer autenticação."""
    client_id = str(user.id)  # client_id sempre = usuário logado (perfil/conversas por usuário)
    if req.stream:
        async def event_generator():
            async for event in chat_stream(db, req.message, req.conversation_id, client_id):
                yield event

        return EventSourceResponse(event_generator())

    result = await chat(db, req.message, req.conversation_id, client_id)
    return ChatResponse(**result)


# ═══════════════════════════════════════════════════════════════════════════
# Conversations
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/conversations")
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Lista as conversas DO USUÁRIO logado, por última atualização."""
    convs = (await db.execute(
        sa_select(Conversation)
        .where(Conversation.total_messages > 0, Conversation.client_id == str(user.id))
        .order_by(Conversation.updated_at.desc())
        .limit(50)
    )).scalars().all()
    return [
        {
            "id": str(c.id),
            "title": c.title,
            "total_messages": c.total_messages,
            "model_used": c.model_used,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        }
        for c in convs
    ]


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Retorna mensagens de uma conversa (só se for do usuário logado)."""
    conv = (await db.execute(
        sa_select(Conversation).where(Conversation.id == conversation_id)
    )).scalar_one_or_none()
    if not conv or conv.client_id != str(user.id):
        raise HTTPException(status_code=404, detail="Conversa não encontrada")
    msgs = (await db.execute(
        sa_select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )).scalars().all()
    return [
        {
            "id": str(m.id),
            "role": m.role,
            "content": m.content,
            "model_used": m.model_used,
            "latency_ms": m.latency_ms,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in msgs
    ]


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Deleta uma conversa e suas mensagens (só se for do usuário logado)."""
    conv = (await db.execute(
        sa_select(Conversation).where(Conversation.id == conversation_id)
    )).scalar_one_or_none()
    if not conv or conv.client_id != str(user.id):
        raise HTTPException(status_code=404, detail="Conversa não encontrada")
    await db.delete(conv)
    await db.commit()
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    req: IngestRequest,
    db: AsyncSession = Depends(get_db),
    _auth: bool = Depends(_check_api_key),
):
    """Ingere fontes oficiais na knowledge base."""
    if req.fonte_id:
        result = await ingest_fonte(db, req.fonte_id)
    else:
        result = await ingest_todas(db, req.source_type)

    # Atualizar tsvectors após ingestão
    await update_search_vectors(db)

    status = "ok" if result.get("status") == "ok" or "total_fontes" in result else "erro"
    return IngestResponse(status=status, detail=result)


@router.post("/ingest-text")
async def ingest_text_endpoint(
    req: dict,
    db: AsyncSession = Depends(get_db),
    _auth: bool = Depends(_check_api_key),
):
    """Ingere texto direto (para fontes que o servidor não consegue acessar)."""
    from src.crawler.sources import get_fonte
    from src.crawler.ingest import _clean, _detect_section
    from src.models.tables import Document, Chunk
    from src.rag.embeddings import generate_embedding, chunk_text as chunk_text_fn
    from sqlalchemy import delete as sa_delete

    fonte_id = req.get("fonte_id")
    text = req.get("text", "")
    fonte = get_fonte(fonte_id)
    if not fonte:
        raise HTTPException(status_code=400, detail=f"Fonte '{fonte_id}' não encontrada")
    if len(text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Texto muito curto (min 100 chars)")

    cleaned = _clean(text)
    content_hash = __import__("hashlib").sha256(cleaned.encode()).hexdigest()[:16]

    doc = (await db.execute(
        sa_select(Document).where(Document.source_id == fonte_id)
    )).scalar_one_or_none()

    if doc:
        await db.execute(sa_delete(Chunk).where(Chunk.document_id == doc.id))
        doc.content_hash = content_hash
        doc.raw_size = len(cleaned)
    else:
        doc = Document(
            source_id=fonte_id, title=fonte.nome, url=fonte.url,
            source_type=fonte.source_type, orgao=fonte.orgao,
            fundamentacao=fonte.fundamentacao, content_hash=content_hash,
            raw_size=len(cleaned),
        )
        db.add(doc)
        await db.flush()

    chunks = chunk_text_fn(cleaned)
    count = 0
    for i, ct in enumerate(chunks):
        emb = generate_embedding(ct)
        if not emb:
            continue
        db.add(Chunk(
            document_id=doc.id, content=ct, embedding=emb,
            chunk_index=i, section=_detect_section(ct),
            metadata_={"fonte_id": fonte_id, "url": fonte.url},
        ))
        count += 1
        if count % 50 == 0:
            await db.flush()

    doc.total_chunks = count
    from src.crawler.ingest import update_search_vectors
    await update_search_vectors(db)
    await db.commit()
    return {"status": "ok", "fonte_id": fonte_id, "chunks": count, "tamanho": len(cleaned)}


@router.get("/status")
async def status_endpoint(
    db: AsyncSession = Depends(get_db),
):
    """Status da knowledge base."""
    return await get_status(db)


# ═══════════════════════════════════════════════════════════════════════════
# Feedback
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/feedback")
async def feedback_endpoint(
    req: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Registra feedback sobre uma resposta. Feedback positivo (>=4) dispara learning loop."""
    fb = FeedbackModel(
        message_id=req.message_id,
        rating=req.rating,
        comment=req.comment,
    )
    db.add(fb)
    await db.flush()

    # Learning loop: aprender com feedback positivo
    learn_result = {}
    if req.rating and req.rating >= 4:
        try:
            from src.services.learning import learn_from_feedback
            learn_result = await learn_from_feedback(db, req.message_id, req.rating)
        except Exception as e:
            logger.warning(f"Learning from feedback falhou: {e}")

    await db.commit()
    return {"status": "ok", "learning": learn_result}


# ═══════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/health", response_model=HealthResponse)
async def health_endpoint(
    db: AsyncSession = Depends(get_db),
):
    """Health check do Murdock."""
    try:
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    try:
        kb = await get_status(db)
        kb_info = {"total_chunks": kb.get("total_chunks", 0), "documentos": len(kb.get("documentos", []))}
    except Exception:
        kb_info = {"total_chunks": 0, "documentos": 0}

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        database=db_status,
        knowledge_base=kb_info,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fontes
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/sources")
async def sources_endpoint():
    """Lista fontes oficiais registradas."""
    from src.crawler.sources import FONTES
    return [
        {
            "id": f.id,
            "nome": f.nome,
            "url": f.url,
            "source_type": f.source_type,
            "orgao": f.orgao,
            "fundamentacao": f.fundamentacao,
        }
        for f in FONTES
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Client Profile
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/profile/{client_id}")
async def get_profile_endpoint(
    client_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Retorna o perfil fiscal do usuário logado (client_id derivado do token)."""
    client_id = str(user.id)
    from src.services.memory import get_profile
    profile = await get_profile(db, client_id)
    if not profile:
        return {"client_id": client_id, "exists": False}
    return {
        "client_id": client_id,
        "exists": True,
        "nome": profile.nome,
        "cnpj": profile.cnpj,
        "regime_tributario": profile.regime_tributario,
        "cnae_principal": profile.cnae_principal,
        "uf": profile.uf,
        "municipio": profile.municipio,
        "tipo_atividade": profile.tipo_atividade,
        "faturamento_12m": profile.faturamento_12m,
        "funcionarios": profile.funcionarios,
        "porte": profile.porte,
        "total_conversas": profile.total_conversas,
    }


@router.put("/profile/{client_id}")
async def update_profile_endpoint(
    client_id: str,
    req: dict,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Atualiza o perfil fiscal do usuário logado."""
    client_id = str(user.id)
    from src.services.memory import update_profile
    profile = await update_profile(db, client_id, req)
    await db.commit()
    return {"status": "ok", "client_id": client_id}


# ═══════════════════════════════════════════════════════════════════════════
# Learning Stats
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/learning/stats")
async def learning_stats_endpoint(
    db: AsyncSession = Depends(get_db),
):
    """Estatísticas do learning loop."""
    from src.services.learning import get_learning_stats
    return await get_learning_stats(db)
