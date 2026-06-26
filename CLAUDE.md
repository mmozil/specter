# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Murdock é um agente IA especializado em direito tributário, contábil e fiscal brasileiro. Deploy em `murdock.hovio.com.br`. Usa Pydantic AI como framework de agente, **MiniMax-M2** como LLM primário (OpenAI-compatible API via `OpenAIProvider`), Claude Sonnet como fallback, PostgreSQL+pgvector para RAG, e knowledge base alimentada por fontes oficiais gov.br + Q&A aprendidos via learning loop.

> **Config LLM (jun/2026):** DB-driven, configurável pela UI (padrão Tier Agent). O usuário escolhe quais LLMs, em que ordem e liga/desliga cada uma em **Configurações → LLM** (link "LLM" no rodapé da sidebar do chat). Persistido em `llm_providers` (sem redeploy). As envs `DEFAULT_LLM_*`/`FALLBACK_LLM_*` agora servem só de **SEED** no 1º boot (`seed_from_env_if_empty`). `pydantic-ai` pinado em `==1.104.0` (na 1.x, `base_url`/`api_key` vão via `OpenAIProvider`, não direto no `OpenAIModel`). MiniMax-M2 é modelo de raciocínio: emite `<think>...</think>` no conteúdo — `_strip_reasoning()` + `_stream_filtered()` em `agent.py` removem isso da saída (stream e resposta final).

## Provedores de LLM configuráveis (DB-driven)

Tabela `llm_providers` (criada no startup via `Base.metadata.create_all`): `provider`, `default_model`, `api_key_enc` (Fernet), `base_url`, `temperature`, `max_tokens`, `timeout_s`, `active` (on/off), `priority` (menor = primeiro), `label`.

- **Resolução:** `agent.chat/chat_stream` → `_resolve_models(db)` → `llm_config.build_chain(db)` devolve a cadeia ordenada de **ativos** (priority asc). O agente tenta de cima pra baixo e **cai pro próximo** se um falhar. `model=` + `model_settings={temperature,max_tokens}` passados por chamada ao `murdock_agent.run/run_stream` (o `Agent` e as 7 tools não mudam).
- **Desligamento respeitado:** se há provedores cadastrados mas TODOS off → `ProvidersAllDisabled` → o chat responde "Nenhuma LLM ativa" (não inventa fallback escondido). Sem nenhum cadastrado → `NoProvidersConfigured` → rede de segurança usa o `primary_model` do env.
- **Providers suportados** (`SUPPORTED_PROVIDERS`): `minimax`, `anthropic`, `openai`, `gemini`, `openrouter`, `deepseek`, `nous`, `local`. Todos OpenAI-compatible via `OpenAIModel`+`OpenAIProvider` (gemini usa o endpoint `/v1beta/openai/`); `anthropic` usa `AnthropicModel` nativo (import defensivo → fallback pro inference string `anthropic:<model>` com `ANTHROPIC_API_KEY` do env).
- **Admin:** endpoints `/api/llm-providers/*` protegidos por `X-API-Key` (= `API_KEY`; liberado em dev). A UI guarda a chave em `localStorage` (`murdock_admin_key`). Botão "testar" faz ping real no provider. `bump_version()` invalida o cache de modelos a cada create/update/delete.
- **Gotcha:** `OpenAIModel` é construído por registro e cacheado (reusa o client httpx). Trocar key/modelo/base_url muda a `cache_key` e reconstrói. Anthropic exige `temperature` ≤ 1.

## Commands

```bash
# Dev (requer PostgreSQL com pgvector + Redis)
uvicorn main:app --reload --port 8010

# Docker (inclui PostgreSQL + Redis)
docker compose up -d

# Ingestão da knowledge base (após subir o servidor)
curl -X POST http://localhost:8010/api/ingest -H "Content-Type: application/json" -d '{}'

# Lint
ruff check src/ --fix && ruff format src/
```

## Architecture

```
main.py                    → FastAPI app (entry point, porta 8010)
src/
  core/
    config.py              → Pydantic Settings (DATABASE_URL, DEFAULT_LLM_*/FALLBACK_LLM_* p/ SEED, GEMINI_API_KEY p/ embeddings, etc.)
    database.py            → AsyncPG engine, session factory, init_db()
    encryption.py          → Fernet encrypt/decrypt (chave derivada de MURDOCK_ENCRYPTION_KEY|SECRET_KEY) — protege API keys de LLM em repouso
  models/
    tables.py              → Document, Chunk (pgvector 768d), Conversation, Message, Feedback, ClientProfile, LlmProvider
  rag/
    embeddings.py          → Gemini embedding-001 (768d, LRU cache)
    search.py              → Hybrid search: pgvector cosine + tsvector BM25 + RRF fusion
  crawler/
    sources.py             → 15 FonteOficial (gov.br, jus.br, leg.br)
    ingest.py              → Fetch → parse → chunk → embed → save (dedup por content_hash)
  tools/
    tools.py               → 7 tools Pydantic AI (search_law, calculate_tax, check_ncm, reform_2026, credit_recovery, calendar, jurisprudence)
  services/
    agent.py               → Pydantic AI agent (modelo DB-driven via _resolve_models → build_chain), chat + chat_stream, dynamic prompt, _strip_reasoning/_stream_filtered (remove <think> do MiniMax)
    llm_config.py          → Provedores DB-driven: catálogo, build_chain (cadeia ordenada de ativos), build_model (constrói OpenAIModel/AnthropicModel), seed_from_env_if_empty, ProvidersAllDisabled/NoProvidersConfigured
    memory.py              → Client Memory: extração de perfil fiscal, persistência, injeção no prompt
    learning.py            → Learning Loop: aprende com feedback positivo + conversas engajadas → embeda Q&A no RAG
  api/
    schemas.py             → Pydantic request/response models
    routes.py              → POST /chat (SSE), POST /ingest, POST /ingest-text, GET /status, GET /health, GET /sources, POST /feedback, GET /conversations, GET /conversations/{id}/messages, DELETE /conversations/{id}, GET /profile/{id}, PUT /profile/{id}, GET /learning/stats
    llm_routes.py          → /llm-providers (admin, X-API-Key): GET /supported, GET (list), POST, PATCH/{id}, DELETE/{id}, POST /{id}/test
  static/
    index.html             → Chat UI Harvey.ai design (m.murdock, warm neutrals, Source Serif 4, SSE streaming)
```

## Key Patterns

- **Hybrid Search (RRF)**: Dense (pgvector cosine) + Sparse (tsvector ts_rank_cd) + RRF fusion. Score = sum(1/(k + rank_i)), k=60.
- **Agent Framework**: Pydantic AI `==1.104.0` (pinado). Tools recebem `RunContext[MurdockDeps]` com sessão DB. Model: MiniMax-M2 via `OpenAIModel(model, provider=OpenAIProvider(base_url, api_key))` → fallback Claude Sonnet. Histórico de conversa (últimas 20 msgs) enviado via `message_history` param.
- **Reasoning strip (MiniMax-M2)**: o modelo emite o raciocínio em `<think>...</think>` no conteúdo. `_stream_filtered()` segura os tokens no streaming até passar o `</think>`; `_strip_reasoning()` limpa a resposta final e o histórico. Nunca vaza o pensamento pro usuário.
- **SSE Streaming**: `sse-starlette` no backend, `EventSource` no frontend. Events: `token`, `done`, `error`.
- **Embeddings**: Gemini embedding-001, 768 dimensões, LRU cache 1000 entries. Task types: RETRIEVAL_DOCUMENT (ingest), RETRIEVAL_QUERY (search).
- **Crawler**: Domain validation (.gov.br, .jus.br, .leg.br, .ibpt.org.br). Browser-like headers. Parsers: HTML (BeautifulSoup), JSON API. Dedup via SHA-256 content hash.
- **Chunking**: 1000 chars, 150 overlap, quebra natural em Art./§/parágrafo.
- **Geo-blocking**: Servidor Hetzner (IP alemão) é bloqueado por alguns sites .gov.br. Usar `POST /api/ingest-text` para ingestão direta de texto copiado manualmente dessas fontes.
- **Histórico de conversas**: Sidebar lista conversas anteriores. Endpoints: `GET /api/conversations`, `GET /api/conversations/{id}/messages`, `DELETE /api/conversations/{id}`.
- **Client Memory**: Perfil fiscal persistente por `client_id` (regime, CNAE, UF, faturamento). Extração automática de dados das mensagens do usuário via regex. Injetado no prompt como contexto — o agente nunca pede informações que já tem.
- **Learning Loop**: Após feedback positivo (rating >= 4) ou conversa engajada (6+ msgs), Q&A validados são embeddados como chunks (`source_type=learned_qa`) e surfam na hybrid search em consultas futuras.
- **Feedback UI**: Thumbs up/down em cada resposta do assistant. Feedback positivo dispara learning loop automaticamente.

## Environment

```bash
DATABASE_URL=postgresql+asyncpg://murdock:murdock@localhost:5432/murdock
REDIS_URL=redis://localhost:6379/5
# LLM primário (env-based, padrão Tier Agent — zero hardcode)
DEFAULT_LLM_PROVIDER=minimax
DEFAULT_LLM_MODEL=MiniMax-M2
DEFAULT_LLM_BASE_URL=https://api.minimax.io/v1
DEFAULT_LLM_API_KEY=...    # chave MiniMax COM saldo (sk-cp-...) — sk-api-... estava sem saldo
# Fallback
FALLBACK_LLM_PROVIDER=anthropic
FALLBACK_LLM_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=...      # Claude Sonnet (fallback LLM)
GEMINI_API_KEY=...         # Google AI (embeddings only)
API_KEY=...                # Auth para endpoints admin
```

## Design

- **Estilo**: Harvey.ai editorial — warm neutrals, serif headlines, zero decoração
- **Nome visual**: "m.murdock" (Source Serif 4) + "TAX & LEGAL AI" (subtítulo)
- **Paleta**: warm grays (#0f0e0d ink → #fafaf9 ivory)
- **Fontes**: Source Serif 4 (headlines, Google Fonts) + system sans (body)
- **Radius**: 4px / 8px, accent dourado sutil (#c9a96e) em links e blockquotes, espaçamento generoso
- **Feedback**: Thumbs up/down com estados visuais (verde/vermelho) em hover
- **Perfil sidebar**: Card com dados fiscais do cliente, atualizado automaticamente

## Deploy

> **Workflow (regra do dono, jun/2026):** SEMPRE commitar as mudanças sem perguntar — fluxo padrão `commit → push → deploy → verificar`. Não deixar trabalho só local. Deploy é **manual** (sem webhook), via API Coolify abaixo.

- **Repo**: github.com/mmozil/murdock (público)
- **Domínio**: murdock.hovio.com.br
- **Coolify App**: `xw0wks4oo0gcsossss4kowc4`
- **Coolify PostgreSQL**: `sk4csooc8g8owkkk444sckos` (pgvector:pg16)
- **Coolify Redis**: `yg800wc0kws4k8o444g44gg4`
- **Coolify Project**: `s88c48s0kg884ck8s0gow440` (Hovio)
- **Deploy trigger** (MANUAL — sem webhook GitHub, `git push` NÃO deploya): `curl -s "https://coolify.tier.finance/api/v1/deploy?uuid=xw0wks4oo0gcsossss4kowc4&force=false" -H "Authorization: Bearer 5|claude-deploy-token-2026"` · poll: `GET /api/v1/deployments/<deployment_uuid>`
- **Docker build**: Python 3.12-slim, porta 8010
- **DB**: PostgreSQL 16 + pgvector (docker-compose inclui)
