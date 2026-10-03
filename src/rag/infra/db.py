"""Acceso a PostgreSQL: pool de conexiones."""

from __future__ import annotations

from psycopg_pool import ConnectionPool

from rag.config import Settings


def create_pool(settings: Settings) -> ConnectionPool:
    """Crea el pool sin abrir conexiones hasta que se use (`open=False`)."""
    return ConnectionPool(
        conninfo=(
            f"host={settings.postgres_host} port={settings.postgres_port} "
            f"dbname={settings.postgres_db} user={settings.postgres_user} "
            f"password={settings.postgres_password} "
            f"connect_timeout={int(settings.postgres_connect_timeout_seconds)}"
        ),
        min_size=1,
        max_size=settings.postgres_pool_size,
        timeout=settings.postgres_connect_timeout_seconds,
        open=False,
        name="rag-bancos",
    )
