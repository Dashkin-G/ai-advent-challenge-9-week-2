"""Хранилище состояния: единственное место, которое знает про диск.

Пока приложение запущено, агент помнит разговор в себе. Чтобы он помнил его и
после перезапуска, состояние нужно куда-то класть — этим занимается `Store`.

Здесь нет ни агента, ни модели, ни интерфейса: только таблицы, словари и SQL.
Устройство ровно такое же, как у `llm.py`: тот знает про HTTP API модели и больше
ни про что, этот знает про базу и больше ни про что. Поэтому смена формата
хранения — правка одного этого файла, агент её не замечает.

База — SQLite (`data/agents.db`), она встроена в Python, отдельный сервер и новые
зависимости не нужны. Семь таблиц:

    agents       id, created_at, turns, active, branch + паспорт (name/role/instructions)
                 и настройки (model, temperature, memory_turns, strategy, summarize, …)
    messages     id, agent_id, branch, role, content, at — по строке на сообщение
    summaries    по строке на версию суммаризации: суммаризация хранится отдельно от
                 истории, сообщения в ней остаются как были
    facts        по строке на версию блока фактов «ключ — значение» (JSON в content)
    usage        по строке на обращение: токены, стоимость, вес контекста, стратегия
    checkpoints  точки ветвления: место в ветке (id сообщения), от которого можно
                 создать ветку
    branches     ветки диалога; основная ветка — это branch = 0 без строки в таблице,
                 остальные — строки с именем и точкой развилки

Ветка — отдельная линия истории: при создании она получает КОПИЮ общего начала
(сообщений до точки ветвления вместе с суммаризацией и фактами на тот момент), а
дальше живёт своей перепиской. Копия, а не ссылка на родителя, потому что так все
запросы к истории остаются одним `WHERE agent_id = ? AND branch = ?`, а удаление и
подрезка одной ветки никогда не ломают другую.

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

VERSION = 5  # версия схемы, хранится в PRAGMA user_version

MAIN_BRANCH = 0  # основная ветка: строки в branches у неё нет, это просто «ветка 0»

# Колонки таблицы agents: отсюда собирается и CREATE TABLE, и мягкая миграция.
# Появится новая настройка — колонка допишется в существующую базу сама; объявляй
# такие колонки с DEFAULT, иначе SQLite не сможет добавить их к готовой таблице.
AGENT_COLUMNS = {
    "id": "TEXT PRIMARY KEY",
    "created_at": "REAL NOT NULL DEFAULT 0",
    "turns": "INTEGER NOT NULL DEFAULT 0",
    "active": "INTEGER NOT NULL DEFAULT 0",      # с кем продолжать разговор
    "branch": "INTEGER NOT NULL DEFAULT 0",      # активная ветка диалога (0 — основная)
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
    "strategy": "TEXT NOT NULL DEFAULT ''",      # стратегия контекста: window / facts / branches
    # Сжатие истории — опция поверх стратегии. Колонка НЕ называется `compression`
    # нарочно: та осталась в базах прошлых версий со своим значением, и отличить в
    # ней «пользователь выключил» от «эта версия сюда не писала» уже нельзя. Новая
    # колонка приходит пустой (NULL), и по пустоте агент понимает, что значение надо
    # вывести из прежних настроек, — дальше оно записывается явно.
    "summarize": "INTEGER",                      # сжимать ли историю (NULL — ещё не записано)
    "summary_every": "INTEGER",                  # сообщений за окном до обновления суммаризации
}

MESSAGE_COLUMNS = {
    "id": "INTEGER PRIMARY KEY",
    "agent_id": "TEXT NOT NULL",
    "branch": "INTEGER NOT NULL DEFAULT 0",      # в какой ветке диалога лежит сообщение
    "role": "TEXT NOT NULL",
    "content": "TEXT NOT NULL",
    "at": "REAL NOT NULL",
}

# Расход токенов: по строке на обращение. Отдельная таблица, а не колонки-счётчики
# в agents, потому что интересна не только сумма, но и то, КАК она набиралась —
# из ряда строк видно, что запрос дорожает с каждым обменом, а ответ нет.
# Колонки объявлены словарём по тому же принципу, что и у agents: новые дописываются
# в существующую базу через ALTER TABLE, поэтому объявляй их с DEFAULT.
USAGE_COLUMNS = {
    "id": "INTEGER PRIMARY KEY",
    "agent_id": "TEXT NOT NULL",
    "turn": "INTEGER NOT NULL DEFAULT 0",           # какое это по счёту обращение
    "at": "REAL NOT NULL DEFAULT 0",
    "model": "TEXT NOT NULL DEFAULT ''",
    "prompt_tokens": "INTEGER NOT NULL DEFAULT 0",       # факт по всем вызовам обращения
    "completion_tokens": "INTEGER NOT NULL DEFAULT 0",
    "total_tokens": "INTEGER NOT NULL DEFAULT 0",
    "cost_usd": "REAL",                                  # теоретическая стоимость
    "llm_calls": "INTEGER NOT NULL DEFAULT 0",           # план + шаги + итог
    "estimated": "INTEGER NOT NULL DEFAULT 0",           # оценка агента до отправки
    "context_tokens": "INTEGER NOT NULL DEFAULT 0",      # сколько занял контекст запроса
    "memory_tokens": "INTEGER NOT NULL DEFAULT 0",       # из них память диалога
    "context_limit": "INTEGER NOT NULL DEFAULT 0",       # окно модели на тот момент
    "trimmed_pairs": "INTEGER NOT NULL DEFAULT 0",       # сколько пар памяти выброшено
    "summary_tokens": "INTEGER NOT NULL DEFAULT 0",      # из контекста — суммаризация
    "folded_messages": "INTEGER NOT NULL DEFAULT 0",     # сколько сообщений она заменяла
    "folded_tokens": "INTEGER NOT NULL DEFAULT 0",       # сколько они весили бы сами
    "shadow_tokens": "INTEGER NOT NULL DEFAULT 0",       # теневой вызов для сравнения
    "strategy": "TEXT NOT NULL DEFAULT ''",              # стратегия контекста в этом обращении
    "summarize": "INTEGER NOT NULL DEFAULT 0",           # было ли включено сжатие истории
    "branch": "INTEGER NOT NULL DEFAULT 0",              # в какой ветке шёл разговор
    "facts_tokens": "INTEGER NOT NULL DEFAULT 0",        # из контекста — блок фактов
    "dropped_messages": "INTEGER NOT NULL DEFAULT 0",    # сообщений за окном, не ушедших в модель дословно
    "dropped_tokens": "INTEGER NOT NULL DEFAULT 0",      # сколько они весили бы
}

# Суммаризации: по строке на версию. Отдельная таблица, а не колонка в agents, потому
# что суммаризация — это данные разговора, а не настройка: у неё есть история версий,
# граница в сообщениях и свой вес, и стереть его нужно вместе с перепиской.
SUMMARY_COLUMNS = {
    "id": "INTEGER PRIMARY KEY",
    "agent_id": "TEXT NOT NULL",
    "branch": "INTEGER NOT NULL DEFAULT 0",          # суммаризация у каждой ветки своя
    "version": "INTEGER NOT NULL DEFAULT 0",         # сколько раз суммаризация обновлялась
    "at": "REAL NOT NULL DEFAULT 0",
    "turn": "INTEGER NOT NULL DEFAULT 0",            # после какого обращения свёрнут
    "upto": "INTEGER NOT NULL DEFAULT 0",            # id последнего сообщения, вошедшего в суммаризацию
    "folded_messages": "INTEGER NOT NULL DEFAULT 0", # сколько сообщений заменяет (всего)
    "folded_tokens": "INTEGER NOT NULL DEFAULT 0",   # сколько они весили бы в запросе (оценка)
    "summary_tokens": "INTEGER NOT NULL DEFAULT 0",  # сколько весит сама суммаризация
    "content": "TEXT NOT NULL DEFAULT ''",
}

# Факты: по строке на версию блока «ключ — значение». Устроены как суммаризации —
# это тоже сжатая память со своей историей версий и границей в сообщениях.
FACTS_COLUMNS = {
    "id": "INTEGER PRIMARY KEY",
    "agent_id": "TEXT NOT NULL",
    "branch": "INTEGER NOT NULL DEFAULT 0",
    "version": "INTEGER NOT NULL DEFAULT 0",         # сколько раз блок обновлялся
    "at": "REAL NOT NULL DEFAULT 0",
    "turn": "INTEGER NOT NULL DEFAULT 0",            # после какого обращения обновлён
    "upto": "INTEGER NOT NULL DEFAULT 0",            # id последнего сообщения, учтённого в фактах
    "items": "INTEGER NOT NULL DEFAULT 0",           # сколько фактов в блоке
    "facts_tokens": "INTEGER NOT NULL DEFAULT 0",    # сколько весит блок в запросе
    "content": "TEXT NOT NULL DEFAULT '{}'",         # сами факты: JSON-объект ключ → значение
}

# Точки ветвления: место в ветке, от которого можно создать ветку.
CHECKPOINT_COLUMNS = {
    "id": "INTEGER PRIMARY KEY",
    "agent_id": "TEXT NOT NULL",
    "branch": "INTEGER NOT NULL DEFAULT 0",          # в какой ветке стоит точка
    "upto": "INTEGER NOT NULL DEFAULT 0",            # id последнего сообщения до точки
    "messages": "INTEGER NOT NULL DEFAULT 0",        # сколько сообщений до точки (для показа)
    "name": "TEXT NOT NULL DEFAULT ''",
    "at": "REAL NOT NULL DEFAULT 0",
}

# Ветки диалога. Основная ветка строки не имеет (branch = 0), поэтому база от
# прошлых версий приложения — это просто агенты с одной основной веткой.
BRANCH_COLUMNS = {
    "id": "INTEGER PRIMARY KEY",
    "agent_id": "TEXT NOT NULL",
    "name": "TEXT NOT NULL DEFAULT ''",
    "checkpoint": "INTEGER NOT NULL DEFAULT 0",      # от какой точки ветвления создана
    "origin": "TEXT NOT NULL DEFAULT ''",            # имя той точки (точку могут удалить вместе с её веткой)
    "fork_at": "INTEGER NOT NULL DEFAULT 0",         # id последнего скопированного сообщения (в своей нумерации)
    "shared": "INTEGER NOT NULL DEFAULT 0",          # сколько сообщений общего начала скопировано
    "at": "REAL NOT NULL DEFAULT 0",
}

# Таблицы с внешним ключом на agents: удаление агента уносит их строки каскадом.
TABLES = {
    "messages": MESSAGE_COLUMNS,
    "usage": USAGE_COLUMNS,
    "summaries": SUMMARY_COLUMNS,
    "facts": FACTS_COLUMNS,
    "checkpoints": CHECKPOINT_COLUMNS,
    "branches": BRANCH_COLUMNS,
}

INDEXES = (
    "CREATE INDEX IF NOT EXISTS messages_by_agent ON messages(agent_id, id)",
    "CREATE INDEX IF NOT EXISTS messages_by_branch ON messages(agent_id, branch, id)",
    "CREATE INDEX IF NOT EXISTS usage_by_agent ON usage(agent_id, id)",
    "CREATE INDEX IF NOT EXISTS summaries_by_agent ON summaries(agent_id, id)",
    "CREATE INDEX IF NOT EXISTS facts_by_agent ON facts(agent_id, id)",
)


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
            conn.execute("PRAGMA foreign_keys = ON")  # чтобы работал каскад
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
            branches = conn.execute("SELECT COUNT(*) FROM branches").fetchone()[0]
            spent = conn.execute("SELECT COALESCE(SUM(total_tokens), 0) FROM usage").fetchone()[0]
        logger.info(
            "Состояние: %d агент(ов), %d сообщ., %d веток, %d токен(ов) израсходовано · %s",
            agents, messages, branches, spent, self.path,
        )

    def _create_schema(self) -> None:
        agents = ", ".join(f"{name} {declaration}" for name, declaration in AGENT_COLUMNS.items())
        cascade = "FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE"
        with self._connect() as conn:
            conn.execute(f"CREATE TABLE IF NOT EXISTS agents ({agents})")
            self._add_new_columns(conn, "agents", AGENT_COLUMNS)
            for table, columns in TABLES.items():
                declared = ", ".join(f"{name} {declaration}" for name, declaration in columns.items())
                # Внешний ключ дописан отдельной строкой: ALTER TABLE его добавить не
                # умеет, а таблица целиком создаётся и в базе от прошлой версии.
                conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({declared}, {cascade})")
                self._add_new_columns(conn, table, columns)
            for index in INDEXES:   # после колонок: индекс по branch требует самой колонки
                conn.execute(index)
            conn.execute(f"PRAGMA user_version = {VERSION}")

    @staticmethod
    def _add_new_columns(conn: sqlite3.Connection, table: str, columns: dict) -> None:
        """Дописать колонки, которых нет в уже существующей таблице.

        Так база, сделанная прошлой версией приложения, продолжает работать: новая
        настройка агента появляется колонкой, а история остаётся на месте.
        """
        have = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, declaration in columns.items():
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
                logger.info("В таблицу %s добавлена колонка %s", table, name)

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

    def messages(self, agent_id: str, branch: int = MAIN_BRANCH, limit: int | None = None) -> list[dict]:
        """Переписка ветки агента: вся или последние `limit` сообщений, по порядку."""
        with self._connect() as conn:
            if limit is None:
                rows = conn.execute(
                    "SELECT id, role, content, at FROM messages WHERE agent_id = ? AND branch = ? ORDER BY id",
                    (agent_id, branch),
                ).fetchall()
            else:
                # Последние `limit`: берём с конца, затем возвращаем прямой порядок.
                rows = conn.execute(
                    "SELECT * FROM (SELECT id, role, content, at FROM messages "
                    "WHERE agent_id = ? AND branch = ? ORDER BY id DESC LIMIT ?) ORDER BY id",
                    (agent_id, branch, max(0, limit)),
                ).fetchall()
        return [dict(row) for row in rows]

    def count(self, agent_id: str, branch: int = MAIN_BRANCH) -> int:
        """Сколько сообщений лежит в истории ветки агента."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM messages WHERE agent_id = ? AND branch = ?", (agent_id, branch)
            ).fetchone()[0]

    def last_id(self, agent_id: str, branch: int = MAIN_BRANCH) -> int:
        """Номер последнего сообщения ветки (0, если сообщений нет) — граница для точки ветвления."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(id) AS last FROM messages WHERE agent_id = ? AND branch = ?", (agent_id, branch)
            ).fetchone()
        return int(row["last"] or 0)

    def usage(self, agent_id: str, limit: int | None = None) -> list[dict]:
        """Расход по обращениям агента: строка на обращение, в порядке времени."""
        with self._connect() as conn:
            if limit is None:
                rows = conn.execute(
                    "SELECT * FROM usage WHERE agent_id = ? ORDER BY id", (agent_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM (SELECT * FROM usage WHERE agent_id = ? ORDER BY id DESC "
                    "LIMIT ?) ORDER BY id",
                    (agent_id, max(0, limit)),
                ).fetchall()
        return [dict(row) for row in rows]

    def usage_totals(self, agent_id: str) -> dict:
        """Сколько агент израсходовал за всё время: токены, стоимость, вызовы модели."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS turns, "
                "       COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, "
                "       COALESCE(SUM(completion_tokens), 0) AS completion_tokens, "
                "       COALESCE(SUM(total_tokens), 0) AS total_tokens, "
                "       SUM(cost_usd) AS cost_usd, "
                "       COALESCE(SUM(llm_calls), 0) AS llm_calls "
                "FROM usage WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        totals = dict(row)
        totals["cost_usd"] = totals["cost_usd"] or 0.0
        return totals

    def last_at(self, agent_id: str, branch: int = MAIN_BRANCH) -> float | None:
        """Когда агент разговаривал в последний раз (время последнего сообщения ветки)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT at FROM messages WHERE agent_id = ? AND branch = ? ORDER BY id DESC LIMIT 1",
                (agent_id, branch),
            ).fetchone()
        return row["at"] if row else None

    def summary(self, agent_id: str, branch: int = MAIN_BRANCH) -> dict | None:
        """Действующая суммаризация ветки — последняя версия (None, если суммаризации нет)."""
        return self._latest(agent_id, branch, "summaries")

    def summaries(self, agent_id: str, branch: int = MAIN_BRANCH) -> list[dict]:
        """Все версии суммаризации ветки по порядку: видно, как она росла вместе с разговором."""
        return self._versions(agent_id, branch, "summaries")

    def facts(self, agent_id: str, branch: int = MAIN_BRANCH) -> dict | None:
        """Действующий блок фактов ветки — последняя версия (None, если фактов ещё нет)."""
        return self._latest(agent_id, branch, "facts")

    def facts_versions(self, agent_id: str, branch: int = MAIN_BRANCH) -> list[dict]:
        """Все версии блока фактов ветки по порядку."""
        return self._versions(agent_id, branch, "facts")

    def checkpoints(self, agent_id: str) -> list[dict]:
        """Точки ветвления агента во всех ветках, по порядку создания."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM checkpoints WHERE agent_id = ? ORDER BY id", (agent_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def branches(self, agent_id: str) -> list[dict]:
        """Ветки агента кроме основной, с числом сообщений в каждой."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT b.*, (SELECT COUNT(*) FROM messages m WHERE m.agent_id = b.agent_id "
                "AND m.branch = b.id) AS messages FROM branches b WHERE b.agent_id = ? ORDER BY b.id",
                (agent_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def branch_exists(self, agent_id: str, branch: int) -> bool:
        """Есть ли такая ветка у агента (основная есть всегда)."""
        if branch == MAIN_BRANCH:
            return True
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM branches WHERE agent_id = ? AND id = ?", (agent_id, branch)
            ).fetchone()
        return row is not None

    # --------------------------------------------------------------- запись --

    def save_agent(self, state: dict) -> None:
        """Записать паспорт, настройки и счётчики агента (переписку не трогаем)."""
        with self._connect() as conn:
            self._upsert(conn, state)

    def save_turn(
        self,
        state: dict,
        messages: list[dict],
        usage: dict | None = None,
        facts: dict | None = None,
    ) -> list[int]:
        """Зафиксировать обращение: сообщения, состояние, расход и факты — одной транзакцией.

        Одной транзакцией, потому что иначе счётчик обращений, история и расход
        токенов разъедутся: строка в `usage` без своей пары сообщений врала бы про
        цену разговора. Возвращает id записанных сообщений: по ним агент отличает,
        что из истории уже свёрнуто в суммаризацию или учтено в фактах, а что ещё нет.
        Блок фактов, обновлённый после ответа, пишется той же транзакцией и получает
        границу `upto` по последнему записанному сообщению.
        """
        branch = int(state.get("branch") or MAIN_BRANCH)
        with self._connect() as conn:
            self._upsert(conn, state)
            ids = self._append(conn, state["id"], branch, messages)
            if usage:
                self._spend(conn, state["id"], usage)
            if facts is not None:
                row = dict(facts)
                row["upto"] = ids[-1] if ids else row.get("upto", 0)
                self._insert_version(conn, "facts", FACTS_COLUMNS, state["id"], branch, row)
        return ids

    def save_summary(self, agent_id: str, summary: dict, branch: int = MAIN_BRANCH) -> None:
        """Записать новую версию суммаризации (старые остаются — это её история)."""
        with self._connect() as conn:
            self._insert_version(conn, "summaries", SUMMARY_COLUMNS, agent_id, branch, summary)

    def save_facts(self, agent_id: str, facts: dict, branch: int = MAIN_BRANCH) -> None:
        """Записать новую версию блока фактов отдельно от обращения."""
        with self._connect() as conn:
            self._insert_version(conn, "facts", FACTS_COLUMNS, agent_id, branch, facts)

    def add_checkpoint(self, agent_id: str, branch: int, upto: int, messages: int, name: str) -> int:
        """Поставить точку ветвления после сообщения `upto` в ветке; вернуть её id."""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO checkpoints (agent_id, branch, upto, messages, name, at) VALUES (?, ?, ?, ?, ?, ?)",
                (agent_id, branch, upto, messages, name, time.time()),
            )
        return int(cursor.lastrowid)

    def fork(
        self,
        agent_id: str,
        source: int,
        upto: int,
        name: str,
        origin: str = "",
        checkpoint: int = 0,
    ) -> int:
        """Создать ветку: скопировать в неё общее начало ветки `source` до сообщения `upto`.

        Копируются сообщения, а с ними — версии суммаризации и фактов, не выходящие
        за эту границу (их `upto` переводится в новую нумерацию сообщений). Дальше
        ветка живёт своей жизнью: новые сообщения, суммаризации и факты пишутся уже
        в неё, а исходная ветка ничего не замечает. Возвращает id новой ветки.
        """
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO branches (agent_id, name, checkpoint, origin, at) VALUES (?, ?, ?, ?, ?)",
                (agent_id, name, checkpoint, origin, now),
            )
            branch = int(cursor.lastrowid)
            rows = conn.execute(
                "SELECT id, role, content, at FROM messages WHERE agent_id = ? AND branch = ? AND id <= ? "
                "ORDER BY id",
                (agent_id, source, upto),
            ).fetchall()
            mapping: dict[int, int] = {}   # старый id → новый id
            for row in rows:
                copied = conn.execute(
                    "INSERT INTO messages (agent_id, branch, role, content, at) VALUES (?, ?, ?, ?, ?)",
                    (agent_id, branch, row["role"], row["content"], row["at"]),
                )
                mapping[row["id"]] = int(copied.lastrowid)
            fork_at = mapping[rows[-1]["id"]] if rows else 0
            conn.execute(
                "UPDATE branches SET fork_at = ?, shared = ? WHERE id = ?", (fork_at, len(rows), branch)
            )
            for table, columns in (("summaries", SUMMARY_COLUMNS), ("facts", FACTS_COLUMNS)):
                versions = conn.execute(
                    f"SELECT * FROM {table} WHERE agent_id = ? AND branch = ? AND upto <= ? ORDER BY id",
                    (agent_id, source, upto),
                ).fetchall()
                for version in versions:
                    row = dict(version)
                    row["upto"] = _remap(row["upto"], mapping)
                    self._insert_version(conn, table, columns, agent_id, branch, row, at=row["at"])
        logger.info(
            "Агент [%s]: ветка «%s» [%d] создана от ветки %d — скопировано %d сообщ.",
            agent_id, name, branch, source, len(rows),
        )
        return branch

    def set_active(self, agent_id: str) -> None:
        """Запомнить, с кем продолжать разговор при следующем запуске."""
        with self._connect() as conn:
            conn.execute("UPDATE agents SET active = (id = ?)", (agent_id,))

    def forget(self, agent_id: str) -> None:
        """Стереть переписку агента во всех ветках, её суммаризации, факты, ветки, точки и расход.

        Сам агент с паспортом и настройками остаётся.
        """
        with self._connect() as conn:
            for table in TABLES:
                conn.execute(f"DELETE FROM {table} WHERE agent_id = ?", (agent_id,))
            conn.execute("UPDATE agents SET branch = ? WHERE id = ?", (MAIN_BRANCH, agent_id))
        logger.info("История агента [%s] стёрта", agent_id)

    def remove_branch(self, agent_id: str, branch: int) -> None:
        """Удалить ветку вместе с её сообщениями, суммаризациями, фактами и точками.

        Основную ветку удалить нельзя — у неё нет строки, это сама история агента.
        Ветки, созданные от точек этой ветки, не страдают: у них своя копия начала.
        """
        if branch == MAIN_BRANCH:
            return
        with self._connect() as conn:
            for table in ("messages", "summaries", "facts", "checkpoints"):
                conn.execute(f"DELETE FROM {table} WHERE agent_id = ? AND branch = ?", (agent_id, branch))
            conn.execute("DELETE FROM branches WHERE agent_id = ? AND id = ?", (agent_id, branch))
            conn.execute(
                "UPDATE agents SET branch = ? WHERE id = ? AND branch = ?", (MAIN_BRANCH, agent_id, branch)
            )
        logger.info("Агент [%s]: ветка [%d] удалена", agent_id, branch)

    def remove_agent(self, agent_id: str) -> None:
        """Удалить агента; переписка, ветки и расход уходят следом каскадом."""
        with self._connect() as conn:
            conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        logger.info("Агент [%s] удалён из истории", agent_id)

    # ------------------------------------------------------------ внутреннее --

    def _latest(self, agent_id: str, branch: int, table: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE agent_id = ? AND branch = ? ORDER BY id DESC LIMIT 1",
                (agent_id, branch),
            ).fetchone()
        return dict(row) if row else None

    def _versions(self, agent_id: str, branch: int, table: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM {table} WHERE agent_id = ? AND branch = ? ORDER BY id", (agent_id, branch)
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _insert_version(
        conn: sqlite3.Connection,
        table: str,
        columns: dict,
        agent_id: str,
        branch: int,
        version: dict,
        at: float | None = None,
    ) -> None:
        """Вставить строку-версию (суммаризации или фактов): чего в словаре нет — DEFAULT."""
        row = {name: version[name] for name in columns
               if name in version and name not in ("id", "agent_id", "branch", "at")}
        row["agent_id"] = agent_id
        row["branch"] = branch
        row["at"] = at if at is not None else time.time()
        names = ", ".join(row)
        marks = ", ".join(f":{name}" for name in row)
        conn.execute(f"INSERT INTO {table} ({names}) VALUES ({marks})", row)

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
    def _append(conn: sqlite3.Connection, agent_id: str, branch: int, messages: list[dict]) -> list[int]:
        """Дописать сообщения в историю ветки, удержав её в пределах лимита."""
        now = time.time()
        ids = []
        for m in messages:
            cursor = conn.execute(
                "INSERT INTO messages (agent_id, branch, role, content, at) VALUES (?, ?, ?, ?, ?)",
                (agent_id, branch, m["role"], m["content"], now),
            )
            ids.append(int(cursor.lastrowid))
        # Истории нужен потолок: самые старые сообщения уходят первыми — в контекст
        # модели они всё равно уже не попадают. Потолок — на каждую ветку свой.
        conn.execute(
            "DELETE FROM messages WHERE agent_id = ? AND branch = ? AND id NOT IN ("
            "    SELECT id FROM messages WHERE agent_id = ? AND branch = ? ORDER BY id DESC LIMIT ?)",
            (agent_id, branch, agent_id, branch, config.HISTORY_LIMIT),
        )
        return ids

    @staticmethod
    def _spend(conn: sqlite3.Connection, agent_id: str, usage: dict) -> None:
        """Записать расход обращения строкой в `usage`."""
        # Чего в словаре нет, того нет и в запросе: за такие колонки ответит DEFAULT.
        row = {name: usage[name] for name in USAGE_COLUMNS
               if name in usage and name not in ("id", "agent_id", "at")}
        row["agent_id"] = agent_id
        row["at"] = time.time()
        columns = ", ".join(row)
        marks = ", ".join(f":{name}" for name in row)
        conn.execute(f"INSERT INTO usage ({columns}) VALUES ({marks})", row)


def _remap(upto: int, mapping: dict[int, int]) -> int:
    """Граница `upto` в нумерации исходной ветки → в нумерации копии.

    Берём самое позднее скопированное сообщение, которое не новее границы: границы
    суммаризации и фактов всегда указывают на сообщение, попавшее в копию.
    """
    return max((new for old, new in mapping.items() if old <= upto), default=0)


def _row(state: dict) -> dict:
    """Состояние агента → плоская строка таблицы agents."""
    profile = state.get("profile") or {}
    settings = state.get("settings") or {}
    return {
        "id": state["id"],
        "created_at": float(state.get("created_at") or time.time()),
        "turns": int(state.get("turns") or 0),
        "branch": int(state.get("branch") or MAIN_BRANCH),
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
        "strategy": settings.get("strategy") or "",
        # Чего агент не сказал, того не пишем: NULL здесь значит «значение ещё не
        # записано», и агент выведет его сам (см. комментарий у колонки).
        "summarize": None if settings.get("summarize") is None else int(bool(settings["summarize"])),
        "summary_every": settings.get("summary_every"),
    }


def _state(row: sqlite3.Row) -> dict:
    """Строка таблицы agents → состояние агента в том виде, в каком он его отдал."""
    settings = {
        "model": row["model"],
        "temperature": row["temperature"],
        "max_tokens": row["max_tokens"],
        "memory_turns": row["memory_turns"],
        "tools_enabled": bool(row["tools_enabled"]),
        "planning": bool(row["planning"]),
        "max_steps": row["max_steps"],
        "strategy": row["strategy"],
        # None — колонку только что дописали и значения в ней ещё нет: агент выведет
        # его из прежних настроек сам.
        "summarize": None if row["summarize"] is None else bool(row["summarize"]),
        "summary_every": row["summary_every"],
    }
    if "compression" in row.keys():
        # Колонка из базы позапрошлой версии, где сжатие было отдельным тумблером:
        # агент возьмёт её, если настройки поновее ещё не записаны.
        settings["compression"] = bool(row["compression"])
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "turns": row["turns"],
        "branch": row["branch"],
        "profile": {
            "name": row["name"],
            "role": row["role"],
            "instructions": row["instructions"],
        },
        "settings": settings,
    }
