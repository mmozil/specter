"""Criptografia simétrica (Fernet) para segredos em repouso — ex: API keys de LLM.

Chave derivada de `MURDOCK_ENCRYPTION_KEY` (se setada) ou de `SECRET_KEY` via SHA-256.
Assim, enquanto o `SECRET_KEY` for estável, as keys salvas continuam decriptáveis entre
deploys — sem exigir uma env nova. `decrypt` tolera valor em texto puro (legado).
"""
import base64
import hashlib
import logging
import os

from cryptography.fernet import Fernet

from src.core.config import settings

logger = logging.getLogger(__name__)


def _build_fernet() -> Fernet:
    secret = os.environ.get("MURDOCK_ENCRYPTION_KEY") or settings.SECRET_KEY or "murdock-fallback-secret"
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


_fernet = _build_fernet()


def encrypt(plain: str | None) -> str:
    """Criptografa uma string. None/vazio → string vazia."""
    if not plain:
        return ""
    return _fernet.encrypt(plain.encode()).decode()


def decrypt(token: str | None) -> str:
    """Decripta um token. Vazio → vazio. Texto puro (legado) → retorna como veio."""
    if not token:
        return ""
    try:
        return _fernet.decrypt(token.encode()).decode()
    except Exception:
        # Pode ser um valor salvo em texto puro antes da criptografia entrar.
        return token
