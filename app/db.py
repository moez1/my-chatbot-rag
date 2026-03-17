"""Database connection and session management.

Two connection modes:
- Async (asyncpg): used by FastAPI endpoints — non-blocking I/O
- Sync (psycopg2): used by Alembic migrations — simpler, blocking

Why async?
FastAPI is async. If we use a sync DB driver, every SQL query
blocks the event loop and all other requests wait. With asyncpg,
queries run without blocking.
"""

import os
from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.settings import get_settings


@lru_cache()
def _build_url(driver: str = "asyncpg") -> str:
    """Build the database URL from settings + env vars."""
    settings = get_settings()
    password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", settings.database.host)
    port = int(os.getenv("DB_PORT", settings.database.port))
    name = os.getenv("DB_NAME", settings.database.name)
    user = os.getenv("DB_USER", settings.database.user)
    return f"postgresql+{driver}://{user}:{password}@{host}:{port}/{name}"


def get_database_url() -> str:
    """Sync URL for Alembic (psycopg2)."""
    return _build_url("psycopg2")


def get_async_database_url() -> str:
    """Async URL for FastAPI (asyncpg)."""
    return _build_url("asyncpg")


# Async engine + session for FastAPI
async_engine = create_async_engine(get_async_database_url(), echo=False)
async_session_factory = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


async def get_session() -> AsyncGenerator[AsyncSession]:
    """Provide an async database session for FastAPI dependency injection.

    Usage in endpoints:
        async def my_endpoint(db: AsyncSession = Depends(get_session)):
            result = await db.execute(select(...))
    """
    async with async_session_factory() as session:
        yield session


# Sync engine for Alembic
sync_engine = create_engine(get_database_url(), future=True)
SyncSessionLocal = sessionmaker(
    bind=sync_engine, autoflush=False, autocommit=False, expire_on_commit=False
)


