"""Esquema de la base de datos de conversaciones (PostgreSQL).

Se aplica de forma idempotente al iniciar (`CREATE ... IF NOT EXISTS`), por lo que no requiere
una herramienta de migraciones para esta versión. Si el esquema evolucionara, se añadirían
sentencias `ALTER ... IF NOT EXISTS` al final de la lista.
"""

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS conversations (
        session_id        TEXT PRIMARY KEY,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        last_activity_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id               BIGSERIAL PRIMARY KEY,
        session_id       TEXT NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
        role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content          TEXT NOT NULL,
        created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
        bank_filter      TEXT,
        latency_ms       INTEGER,
        retrieval_ms     INTEGER,
        rerank_ms        INTEGER,
        generation_ms    INTEGER,
        top_score        DOUBLE PRECISION,
        answered         BOOLEAN,
        rewritten_query  TEXT,
        feedback         SMALLINT CHECK (feedback IN (-1, 1))
    )
    """,
    "CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (session_id, id)",
    "CREATE INDEX IF NOT EXISTS messages_created_idx ON messages (created_at)",
    """
    CREATE TABLE IF NOT EXISTS message_citations (
        message_id  BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
        position    SMALLINT NOT NULL,
        title       TEXT NOT NULL,
        url         TEXT NOT NULL,
        bank        TEXT NOT NULL,
        PRIMARY KEY (message_id, position)
    )
    """,
)

# Evita que varios procesos creen el esquema a la vez al arrancar
SCHEMA_LOCK_ID = 7_431_001
