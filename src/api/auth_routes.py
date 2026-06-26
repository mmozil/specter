"""Rotas de autenticação — cadastro, login e usuário atual."""
import logging
import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.core.security import create_token, get_current_user, hash_password, verify_password
from src.models.tables import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegisterIn(BaseModel):
    name: str = ""
    email: str
    password: str


class LoginIn(BaseModel):
    email: str
    password: str


def _user_out(u: User) -> dict:
    return {"id": str(u.id), "name": u.name, "email": u.email, "is_admin": u.is_admin}


@router.post("/register")
async def register(payload: RegisterIn, db: AsyncSession = Depends(get_db)):
    email = (payload.email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(400, "Email inválido")
    if len(payload.password or "") < 6:
        raise HTTPException(400, "A senha precisa ter ao menos 6 caracteres")

    exists = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if exists:
        raise HTTPException(409, "Este email já está cadastrado")

    user = User(
        email=email,
        name=(payload.name or "").strip()[:200] or email.split("@")[0],
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info("[AUTH] novo usuário: %s", email)
    return {"token": create_token(user.id), "user": _user_out(user)}


@router.post("/login")
async def login(payload: LoginIn, db: AsyncSession = Depends(get_db)):
    email = (payload.email or "").strip().lower()
    user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if not user or not verify_password(payload.password or "", user.password_hash):
        raise HTTPException(401, "Email ou senha incorretos")
    if not user.is_active:
        raise HTTPException(403, "Conta desativada")
    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()
    return {"token": create_token(user.id), "user": _user_out(user)}


@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return _user_out(user)
