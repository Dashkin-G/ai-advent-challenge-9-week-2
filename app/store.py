"""Хранилище состояния: единственное место, которое знает про диск.

Пока приложение запущено, агент помнит разговор в себе. Чтобы он помнил его и
после перезапуска, состояние нужно куда-то класть — этим занимается `Store`.

Здесь нет ни агента, ни модели, ни интерфейса: только таблицы, словари и SQL.
Устройство ровно такое же, как у `llm.py`: тот знает про HTTP API модели и больше
ни про что, этот знает про базу и больше ни про что. Поэтому смена формата
хранения — правка одного этого файла, агент её не замечает.

База — SQLite (`data/agents.db`), она встроена в Python, отдельный сервер и новые
зависимости не нужны. Две таблицы:

    agents    id, created_at, turns, active + паспорт (name/role/instructions)
              и настройки (model, temperature, memory_turns, tools_enabled, …)
    messages  id, agent_id, role, content, at — по строке на сообщение

Почему база, а не файл целиком: сообщение дописывается одной строкой (INSERT), а
не переписыванием всей истории, обращение фиксируется транзакцией (на диске либо
всё обращение, либо ничего), и удаление агента уносит его переписку каскадом.

Соединение открывается на операцию и тут же закрывается: обращение к модели идёт
в фоновом потоке, а соединение sqlite3 привязано к своему потоку. Заодно все
операции проходят под общим замком — чтения и записи не наступают друг другу на
пятки. Битая база не роняет запуск: файл откладывается рядом, приложение стартует
с чистым состоянием.
"""
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from . import config

logger = logging.getLogger("app.store")

VERSION = 1  # версия схемы, хранится в PRAGMA user_version

# Колонки таблицы agents: отсюда собирается и CREATE TABLE, и мягкая миграция.
# Появится новая настройка — колонка допишется в существующую базу сама; объявляй
# такие колонки с DEFAULT, иначе SQLite не сможет добавить их к готовой таблице.
AGENT_COLUMNS = {
    "id": "TEXT PRIMARY KEY",
    "created_at": "REAL NOT NULL DEFAULT 0",
    "turns": "INTEGER NOT NULL DEFAULT 0",
    "active": "INTEGER NOT NULL DEFAULT 0",      # с кем продолжать разговор
    "name": "TEXT NOT NULL DEFAULT ''",
    "role": "TEXT NOT NULL DEFAULT ''",
    "instructions": "TEXT NOT NULL DEFAULT ''",
    "model": "TEXT NOT NULL DEFAULT ''",
    "temperature": "REAL",
    "max_tokens": "INTEGER",
    "memory_turns": "INTEGER",
    "tools_enabled": "INTEGER NOT NULL DEFAULT 1",
    "planning": "INTEGER NOT NULL DEFAULT 1",
    "max_steps": "INTEGER",
}

MESSAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id       INTEGER PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    role     TEXT NOT NULL,
    content  TEXT NOT NULL,
    at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_agent ON messages(agent_id, id);
"""


class Store:
    """Состояние приложения в базе: агенты, их настройки и переписка."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or config.STATE_FILE)
        self._lock = threading.RLock()
        self._prepare()

    # ---------------------------------------------------------------- база --

    @contextmanager
    def _connect(self):
        """Соединение на одну операцию: транзакция закрывается вместе с ним."""
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=5)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")  # чтобы работал каскад messages
            try:
                with conn:  # commit при успехе, откат при ошибке
                    yield conn
            finally:
                conn.close()

    def _prepare(self) -> None:
        """Создать базу и схему; на битом файле начать с чистой базы."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._create_schema()
        except sqlite3.DatabaseError as e:
            broken = self.path.with_name(self.path.name + ".broken")
            logger.warning("База не открылась (%s): файл отложен в %s", e, broken.name)
            try:
                self.path.replace(broken)
            except OSError:
                pass
            self._create_schema()

        with self._connect() as conn:
            agents = conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
            messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        logger.info("Состояние: %d агент(ов), %d сообщ. · %s", agents, messages, self.path)

    def _create_schema(self) -> None:
        columns = ", ".join(f"{name} {declaration}" for name, declaration in AGENT_COLUMNS.items())
        with self._connect() as conn:
            conn.execute(f"CREATE TABLE IF NOT EXISTS agents ({columns})")
            conn.executescript(MESSAGES_SCHEMA)
            self._add_new_columns(conn)
            conn.execute(f"PRAGMA user_version = {VERSION}")

    @staticmethod
    def _add_new_columns(conn: sqlite3.Connection) -> None:
        """Дописать колонки, которых нет в уже существующей базе.

        Так база, сделанная прошлой версией приложения, продолжает работать: новая
        настройка агента появляется колонкой, а история остаётся на месте.
        """
        have = {row["name"] for row in conn.execute("PRAGMA table_info(agents)")}
        for name, declaration in AGENT_COLUMNS.items():
            if name not in have:
                conn.execute(f"ALTER TABLE agents ADD COLUMN {name} {declaration}")
                logger.info("В таблицу agents добавлена колонка %s", name)

    # --------------------------------------------------------------- чтение --

    def agents(self) -> list[dict]:
        """Состояния сохранённых агентов без переписки, в порядке создания."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM agents ORDER BY created_at, rowid").fetchall()
        return [_state(row) for row in rows]

    def active_id(self) -> str | None:
        """Кто был активен в прошлый раз (или None, если неизвестно)."""
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM agents WHERE active = 1").fetchone()
        return row["id"] if row else None

    def messages(self, agent_id: str, limit: int | None = None) -> list[dict]:
        """Переписка агента: вся или последние `limit` сообщений, по порядку."""
        with self._connect() as conn:
            if limit is None:
                rows = conn.execute(
                    "SELECT id, role, content, at FROM messages WHERE agent_id = ? ORDER BY id",
                    (agent_id,),
                ).fetchall()
            else:
                # Последние `limit`: берём с конца, затем возвращаем прямой порядок.
                rows = conn.execute(
                    "SELECT * FROM (SELECT id, role, content, at FROM messages "
                    "WHERE agent_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id",
                    (agent_id, max(0, limit)),
                ).fetchall()
        return [dict(row) for row in rows]

    def count(self, agent_id: str) -> int:
        """Сколько сообщений агента лежит в истории."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM messages WHERE agent_id = ?", (agent_id,)
            ).fetchone()[0]

    def last_at(self, agent_id: str) -> float | None:
        """Когда агент разговаривал в последний раз (время последнего сообщения)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT at FROM messages WHERE agent_id = ? ORDER BY id DESC LIMIT 1", (agent_id,)
            ).fetchone()
        return row["at"] if row else None

    # --------------------------------------------------------------- запись --

    def save_agent(self, state: dict) -> None:
        """Записать паспорт, настройки и счётчики агента (переписку не трогаем)."""
        with self._connect() as conn:
            self._upsert(conn, state)

    def save_turn(self, state: dict, messages: list[dict]) -> None:
        """Зафиксировать обращение: сообщения и состояние агента — одной транзакцией."""
        with self._connect() as conn:
            self._upsert(conn, state)
            self._append(conn, state["id"], messages)

    def set_active(self, agent_id: str) -> None:
        """Запомнить, с кем продолжать разговор при следующем запуске."""
        with self._connect() as conn:
            conn.execute("UPDATE agents SET active = (id = ?)", (agent_id,))

    def forget(self, agent_id: str) -> None:
        """Стереть переписку агента, оставив самого агента с его настройками."""
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE agent_id = ?", (agent_id,))
        logger.info("История агента [%s] стёрта", agent_id)

    def remove_agent(self, agent_id: str) -> None:
        """Удалить агента; переписка уходит следом каскадом."""
        with self._connect() as conn:
            conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        logger.info("Агент [%s] удалён из истории", agent_id)

    # ------------------------------------------------------------ внутреннее --

    @staticmethod
    def _upsert(conn: sqlite3.Connection, state: dict) -> None:
        """Записать состояние агента: новая строка или обновление существующей."""
        row = _row(state)
        columns = ", ".join(row)
        marks = ", ".join(f":{name}" for name in row)
        # created_at и active при обновлении не трогаем: первое задаётся один раз,
        # второе — дело переключения агентов, а не сохранения состояния.
        updates = ", ".join(
            f"{name} = excluded.{name}" for name in row if name not in ("id", "created_at")
        )
        conn.execute(
            f"INSERT INTO agents ({columns}) VALUES ({marks}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            row,
        )

    @staticmethod
    def _append(conn: sqlite3.Connection, agent_id: str, messages: list[dict]) -> None:
        """Дописать сообщения в историю агента, удержав её в пределах лимита."""
        now = time.time()
        conn.executemany(
            "INSERT INTO messages (agent_id, role, content, at) VALUES (?, ?, ?, ?)",
            [(agent_id, m["role"], m["content"], now) for m in messages],
        )
        # Истории нужен потолок: самые старые сообщения уходят первыми — в контекст
        # модели они всё равно уже не попадают.
        conn.execute(
            "DELETE FROM messages WHERE agent_id = ? AND id NOT IN ("
            "    SELECT id FROM messages WHERE agent_id = ? ORDER BY id DESC LIMIT ?)",
            (agent_id, agent_id, config.HISTORY_LIMIT),
        )


def _row(state: dict) -> dict:
    """Состояние агента → плоская строка таблицы agents."""
    profile = state.get("profile") or {}
    settings = state.get("settings") or {}
    return {
        "id": state["id"],
        "created_at": float(state.get("created_at") or time.time()),
        "turns": int(state.get("turns") or 0),
        "name": profile.get("name") or "",
        "role": profile.get("role") or "",
        "instructions": profile.get("instructions") or "",
        "model": settings.get("model") or "",
        "temperature": settings.get("temperature"),
        "max_tokens": settings.get("max_tokens"),
        "memory_turns": settings.get("memory_turns"),
        "tools_enabled": int(bool(settings.get("tools_enabled"))),
        "planning": int(bool(settings.get("planning"))),
        "max_steps": settings.get("max_steps"),
    }


def _state(row: sqlite3.Row) -> dict:
    """Строка таблицы agents → состояние агента в том виде, в каком он его отдал."""
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "turns": row["turns"],
        "profile": {
            "name": row["name"],
            "role": row["role"],
            "instructions": row["instructions"],
        },
        "settings": {
            "model": row["model"],
            "temperature": row["temperature"],
            "max_tokens": row["max_tokens"],
            "memory_turns": row["memory_turns"],
            "tools_enabled": bool(row["tools_enabled"]),
            "planning": bool(row["planning"]),
            "max_steps": row["max_steps"],
        },
    }
