"""Агент — самостоятельная сущность приложения.

Ключевая идея: агент — это не «один вызов API», а объект со своим паспортом
(имя, роль, инструкция), собственными настройками, собственной памятью и
собственными инструментами. Наружу он отдаёт один вход — `ask()`; весь цикл
работы спрятан внутри.

Что происходит в `ask()`:

    1. проверка и нормализация входа
    2. план: агент отдельным вызовом решает, нужен ли план, и пишет шаги
    3. бюджет контекста: агент считает вес будущего запроса в токенах и, если тот
       не помещается в окно модели, подрезает память — до вызова, а не после
    4. цикл работы: модель либо просит вызвать инструмент, либо даёт ответ.
       Инструмент исполняет агент, результат возвращается в диалог, цикл идёт
       дальше — до финального ответа или до потолка шагов
    5. разбор ответа и сверка оценки токенов с фактическим `usage`
    6. запись в память, в историю на диске и в расход токенов (в памяти только
       вопрос и итоговый ответ, без служебной переписки с инструментами); дальше
       слово стратегии контекста и сжатия: факты обновляют свой блок
       «ключ — значение», суммаризация сворачивает выпавшее из окна ещё одним
       вызовом модели

Именно шаг 4 отличает агента от чата: одна фраза пользователя разворачивается в
последовательность действий, которые агент выбирает и выполняет сам.

Токены агент считает сам и до отправки (`app/tokens.py`): контекст — ресурс с
жёстким потолком, и знать его цену постфактум поздно. Оценка сверяется с
фактическим `usage` из ответа, расхождение уходит в калибровку, а расход каждого
обращения ложится в хранилище — по нему видно, как дорожает разговор.

Память устроена слоями. В `_memory` живёт окно контекста — последние
`memory_turns` пар «вопрос-ответ» как есть, ровно то, что уходит в модель
дословно. Что делать с остальным, решает стратегия контекста (`strategy`):

    window    скользящее окно (Sliding Window) — всё, что выпало из окна,
              отбрасывается;
    facts     факты (Sticky Facts / Key-Value Memory): после каждого ответа
              отдельный вызов обновляет блок «ключ — значение» (`_facts`): цель,
              ограничения, предпочтения, решения, договорённости; в запрос уходят
              факты + окно;
    branches  ветки диалога (Branching): точка ветвления фиксирует место, от неё
              создаются ветки, каждая продолжается независимо; в модель уходит
              окно активной ветки.

Поверх любой из них включается сжатие истории (`summarize`) — это не стратегия, а
опция: выпавшее из окна копится в `_pending_summary` и каждые `summary_every`
сообщений сворачивается в суммаризацию (`_summary`), которая уходит в запрос
вместо самих сообщений. Поэтому сочетания работают вместе: «факты + суммаризация»,
«ветки + суммаризация» и так далее.

Полная переписка каждой ветки вместе с суммаризациями, фактами, паспортом и
настройками пишется в хранилище (`store.py`), поэтому агент не начинает с нуля
после перезапуска приложения: `load_agents()` поднимает тех же агентов с их
историей, и разговор продолжается так, будто его не прерывали.

Всё вокруг агента намеренно «глупое»: `llm.py` умеет только сходить в HTTP API
модели, `tools.py` — только выполнить работу, `store.py` — только положить
состояние на диск, а интерфейс — только показать ввод, вывод и трассу шагов. Ни
один из них не знает, как формируется запрос и что происходит с ответом, поэтому
агента можно поднять в любом окружении: окне, консоли, сервере или тесте.
"""
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

from . import config, llm, tokens, tools
from .store import MAIN_BRANCH, Store

logger = logging.getLogger("app.agent")


class AgentError(Exception):
    """Ошибка на стороне агента: плохой вход, неверная настройка или сбой вызова.

    Интерфейсы ловят только её и показывают текст пользователю — им не нужно
    знать про устройство API модели.
    """


@dataclass(frozen=True)
class AgentProfile:
    """Паспорт агента: кто он и как себя ведёт.

    `instructions` уходит в модель system-сообщением, `name` и `role` нужны
    интерфейсу. Понадобится другой агент — меняется здесь, остальной код тот же.
    """
    name: str
    role: str
    instructions: str


DEFAULT_PROFILE = AgentProfile(
    name="Адвент",
    role="ИИ-ассистент по курсу",
    instructions=(
        "Ты — «Адвент», агент участника курса по разработке ИИ-агентов. "
        "Отвечай по-русски, по делу и без воды. Если вопрос неоднозначный — задай "
        "один уточняющий вопрос вместо догадок. Если чего-то не знаешь, прямо скажи "
        "об этом и не выдумывай факты. Учитывай, о чём шла речь раньше в диалоге."
    ),
)

# Инструкция про инструменты добавляется к роли, только когда инструменты включены:
# если их выключили, обещать модели несуществующие возможности нельзя.
TOOLS_NOTE = (
    "\n\nУ тебя есть инструменты, и ты умеешь действовать, а не только говорить. "
    "Никогда не выдумывай то, что можно получить инструментом: текущее время, "
    "результат вычисления, содержимое файлов рабочей папки, текст страницы по адресу, "
    "погоду. Работай шагами: вызови инструмент, посмотри результат, реши, что дальше. "
    "Если инструмент вернул ошибку — прочитай её и попробуй иначе. "
    "Когда задача выполнена, дай короткий финальный ответ и перечисли, что именно "
    "сделал."
)

# Про память модели надо сказать прямо. Переписка уходит в запрос, но роль агента
# пишет пользователь, и без этой заметки модель отвечает выученным «я не помню
# прошлые разговоры» — хотя весь разговор лежит у неё же в контексте.
MEMORY_NOTE = (
    "\n\nДальше в этом диалоге идёт твоя память: {count} сообщ. прошлого разговора "
    "(последнее — {when}). Память хранится на диске и восстанавливается при запуске "
    "приложения, так что разговор продолжается, даже если его прерывали. Ты "
    "действительно помнишь всё, что в ней есть: спросят, о чём говорили раньше — "
    "отвечай по этой переписке и никогда не заявляй, что не помнишь прошлые разговоры "
    "или что каждый диалог начинается с чистого листа."
)

# Начало разговора: памяти нет, и придумывать «прошлые беседы» тоже нельзя.
NO_MEMORY_NOTE = (
    "\n\nЭто начало разговора: прошлых сообщений в памяти нет. Если спросят, о чём "
    "говорили раньше, честно скажи, что разговор только начался."
)

# Суммаризация — начало разговора, свёрнутое в список фактов. Модели надо сказать, что
# это именно её память, а не чужой текст, и что подробности в ней могли потеряться:
# иначе она либо игнорирует суммаризацию, либо уверенно «вспоминает» то, чего в ней нет.
SUMMARY_NOTE = (
    "\n\nРазговор длинный, поэтому его начало ({count} сообщ.) заменено суммаризацией — это "
    "твоя память о той части разговора. Опирайся на суммаризацию как на факты, но помни, что "
    "подробности в ней могли потеряться: если спросят о том, чего в ней нет, честно скажи, "
    "что такая деталь не сохранилась.\nСуммаризация прошлого разговора:\n{summary}"
)

# Роль для вызова, который обновляет суммаризацию. Прежняя суммаризация подаётся на вход,
# поэтому результат — суммаризация всего разговора, а не только последних сообщений.
SUMMARY_SYSTEM = (
    "Ты составляешь суммаризацию долгого разговора между пользователем и агентом «{name}». Тебе "
    "дают прежнюю суммаризацию и новые сообщения, которые уходят из памяти агента. Верни "
    "обновлённую суммаризацию целиком: прежние факты, которые ещё важны, плюс новое.\n"
    "Обязательно сохраняй: как зовут пользователя и чем он занимается, его предпочтения и "
    "просьбы, договорённости и решения, числа, даты, названия файлов и тем, что агент уже "
    "сделал (в том числе инструментами) и что осталось открытым. Убирай вежливость, "
    "повторы и пересказ общеизвестного.\n"
    "Формат: список коротких пунктов, каждый — одна строка, не больше {points} пунктов. "
    "Пиши по-русски, только факты из сообщений, ничего не выдумывай. Верни только "
    "суммаризацию, без заголовков и пояснений."
)

SUMMARY_USER = "Прежняя суммаризация:\n{summary}\n\nНовые сообщения ({count}):\n{messages}"

# Факты — блок «ключ — значение», который агент ведёт сам после каждого ответа. В
# запросе он стоит в инструкции: модели надо объяснить, что это её собственная память,
# собранная и из сообщений, которых в контексте уже нет.
FACTS_NOTE = (
    "\n\nНиже — факты этого разговора ({count} шт.), которые ты сам собрал из всей переписки, "
    "в том числе из сообщений, которых в контексте уже нет. Это твоя память: опирайся на "
    "факты как на установленное; если новое сообщение им противоречит, верно новое. Если "
    "спросят о том, чего нет ни в фактах, ни в сообщениях ниже, честно скажи, что такая "
    "деталь не сохранилась.\nФакты разговора:\n{facts}"
)

# Роль для вызова, который обновляет факты. На входе текущий блок и новые сообщения,
# на выходе — блок целиком: так устаревший факт заменяется, отменённый — исчезает.
FACTS_SYSTEM = (
    "Ты ведёшь блок фактов разговора между пользователем и агентом «{name}» — короткую память "
    "вида «ключ: значение». Тебе дают текущие факты и новые сообщения. Верни обновлённый блок "
    "целиком.\n"
    "Что считать фактом: цель и задача пользователя, ограничения и требования, предпочтения, "
    "принятые решения и договорённости, имена, числа, сроки, названия, что агент уже сделал. "
    "Не факты: вежливость, рассуждения, пересказ общеизвестного, то, о чём только спросили, "
    "но ничего не решили.\n"
    "Правила: ключ — короткое существительное или словосочетание (до четырёх слов), значение — "
    "одна короткая фраза; если факт изменился — замени значение под тем же ключом, если "
    "отменён — убери ключ; ничего не выдумывай; не больше {limit} фактов — если их больше, "
    "оставь самые важные. Пиши на языке разговора.\n"
    'Верни СТРОГО JSON без пояснений и markdown: {{"facts": {{"ключ": "значение"}}}}'
)

FACTS_USER = "Текущие факты:\n{facts}\n\nНовые сообщения ({count}):\n{messages}"

PLANNER_SYSTEM = (
    "Ты — планировщик агента. По задаче пользователя реши, нужен ли план действий.\n"
    "Инструменты, доступные исполнителю:\n{tools}\n\n"
    "Если задача решается одним ответом без действий (вопрос, объяснение, беседа) — "
    "верни пустой список шагов. Если нужны действия — 2–5 коротких шагов в "
    "повелительном наклонении, каждый шаг — одно действие, по-русски.\n"
    "У исполнителя есть память прошлого разговора, ты её не видишь: вопросы вроде "
    "«о чём мы говорили раньше» он решает сам, без действий — на них возвращай "
    "пустой список шагов.\n"
    'Верни СТРОГО JSON без пояснений и markdown: {{"steps": ["...", "..."]}}'
)

# Когда шаги закончились, а модель всё ещё зовёт инструменты — просим подвести итог.
FINISH_NUDGE = (
    "Лимит шагов исчерпан. Больше инструменты не вызывай: дай финальный ответ по "
    "тому, что уже сделано, и честно скажи, если что-то осталось невыполненным."
)

# Потолок глубины памяти. Ограничение не техническое, а денежное: окно контекста у
# моделей огромное, но каждая пара из памяти уезжает в модель заново при каждом
# обращении — сто пар в памяти означают сто пар в каждом счёте.
MEMORY_TURNS_MAX = 200

# Суммаризация обновляется, когда за окном памяти накопилось столько сообщений. Меньше
# двух не бывает — сообщения ходят парами; больше сотни бессмысленно — такая очередь
# сама по себе весит как хороший запрос.
SUMMARY_EVERY_MIN, SUMMARY_EVERY_MAX = 2, 100

# Имя ветки или точки ветвления — короткая подпись для вкладки.
NAME_MAX = 40
MAIN_BRANCH_NAME = "основная"

# По этим словам в ответе провайдера видно, что запрос отклонён именно по длине.
# Такую ошибку агент переводит на человеческий язык: «Модель не ответила: 400 …»
# ничего не объясняет, а «в запрос ушло 1.2M токенов при окне 1M» — объясняет.
LENGTH_ERROR_MARKERS = (
    "input length", "range of input", "context length", "maximum context",
    "too long", "exceeds", "token limit",
)


@dataclass
class AgentStep:
    """Один выполненный шаг: какой инструмент вызвал агент и что получил."""
    number: int
    tool: str
    title: str
    arguments: dict
    result: str
    ok: bool = True
    elapsed_s: float | None = None

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "tool": self.tool,
            "title": self.title,
            "arguments": self.arguments,
            "result": self.result,
            "ok": self.ok,
            "elapsed_s": self.elapsed_s,
        }


@dataclass
class _Totals:
    """Счётчик по всему обращению: у агента вызовов модели теперь несколько."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    has_cost: bool = False
    llm_calls: int = 0

    def add(self, raw: dict) -> dict:
        usage = raw.get("usage")
        if usage:
            self.prompt_tokens += usage["prompt_tokens"]
            self.completion_tokens += usage["completion_tokens"]
            self.total_tokens += usage["total_tokens"]
        if raw.get("cost_usd") is not None:
            self.cost_usd += raw["cost_usd"]
            self.has_cost = True
        self.llm_calls += 1
        return raw

    def usage(self) -> dict | None:
        if not self.total_tokens:
            return None
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Summary:
    """Суммаризация: начало разговора, свёрнутое в короткий список фактов.

    Живёт отдельно от истории: сообщения в базе остаются как были, а в запрос
    вместо них уходит этот текст. `upto` — граница в истории: всё, что не новее
    этого сообщения, уже в суммаризации и в окно памяти больше не возвращается.
    """
    text: str = ""
    version: int = 0          # сколько раз суммаризация обновлялась
    upto: int = 0             # id последнего сообщения истории, вошедшего в суммаризацию
    messages: int = 0         # сколько сообщений она заменяет
    tokens: int = 0           # сколько они весили бы в запросе (оценка)
    at: float | None = None   # когда обновлён

    def __bool__(self) -> bool:
        return bool(self.text)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "version": self.version,
            "upto": self.upto,
            "messages": self.messages,
            "tokens": self.tokens,
            "at": self.at,
        }


@dataclass
class Facts:
    """Факты разговора: блок «ключ — значение», который живёт отдельно от сообщений.

    В отличие от суммаризации, факты не заменяют выпавшие из окна сообщения, а
    собираются из каждого обмена сразу — поэтому важная деталь остаётся в запросе,
    даже когда сообщение, где она прозвучала, из окна давно ушло. `upto` — до
    какого сообщения истории факты уже учтены.
    """
    items: dict[str, str] = field(default_factory=dict)
    version: int = 0
    upto: int = 0
    at: float | None = None

    def __bool__(self) -> bool:
        return bool(self.items)

    def text(self) -> str:
        """Блок в том виде, в каком он уходит модели: строка на факт."""
        return "\n".join(f"- {key}: {value}" for key, value in self.items.items())

    def to_dict(self) -> dict:
        return {
            "items": dict(self.items),
            "count": len(self.items),
            "text": self.text(),
            "version": self.version,
            "upto": self.upto,
            "at": self.at,
        }


@dataclass
class Compression:
    """Что произошло при сжатии: сколько сообщений свёрнуто и во что это обошлось."""
    messages: int            # сколько сообщений ушло в суммаризацию этим разом
    before: int              # сколько они весили в токенах (оценка)
    after: int               # сколько весит обновлённая суммаризация целиком
    version: int
    text: str                # сама суммаризация
    elapsed_s: float
    call_tokens: int         # во что обошёлся вызов, который его составил

    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "before": self.before,
            "after": self.after,
            "version": self.version,
            "text": self.text,
            "elapsed_s": self.elapsed_s,
            "call_tokens": self.call_tokens,
        }


@dataclass
class FactsUpdate:
    """Что произошло с фактами после ответа: что добавилось, изменилось, ушло."""
    messages: int            # сколько сообщений обработано этим разом
    version: int
    items: int               # сколько фактов в блоке теперь
    tokens: int              # сколько весит блок в запросе (вместе с заметкой)
    added: list[str]         # «ключ: значение»
    changed: list[str]       # «ключ: было → стало»
    removed: list[str]       # «ключ»
    text: str                # блок целиком
    elapsed_s: float
    call_tokens: int         # во что обошёлся вызов извлечения

    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "version": self.version,
            "items": self.items,
            "tokens": self.tokens,
            "added": list(self.added),
            "changed": list(self.changed),
            "removed": list(self.removed),
            "text": self.text,
            "elapsed_s": self.elapsed_s,
            "call_tokens": self.call_tokens,
        }


@dataclass
class Shadow:
    """Теневой ответ для сравнения: тот же вопрос, но вся история как есть вместо стратегии.

    Считается и оплачивается по-настоящему, но в память не идёт: он нужен, только
    чтобы положить рядом два ответа и два счёта — со стратегией и с полной историей.
    """
    text: str
    messages: int             # сколько сообщений истории ушло в модель
    breakdown: tokens.Breakdown
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    elapsed_s: float
    trimmed_pairs: int = 0    # даже полная история не влезла в окно — что-то выброшено
    tool_calls: int = 0       # модель попросила инструменты: теневой прогон их не исполняет
    finish_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "messages": self.messages,
            "breakdown": self.breakdown.to_dict(),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "elapsed_s": self.elapsed_s,
            "trimmed_pairs": self.trimmed_pairs,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
        }


@dataclass
class TokenReport:
    """Счёт за обращение в токенах: оценка до вызова и факт после.

    Оценку агент считает сам (`app/tokens.py`) ещё до того, как запрос ушёл в
    сеть, — иначе о цене и о переполнении контекста узнаёшь только по факту, то
    есть когда платить уже поздно. Факт приходит в `usage`, и разница между
    оценкой и фактом здесь же: по ней видно, можно ли счётчику верить.

    Здесь же — цена стратегии контекста: сколько сообщений истории в модель не
    ушло дословно, сколько они весили бы, чем их заменили (суммаризация, факты
    или ничем) и сколько весил бы весь запрос, уйди история как есть.
    """
    breakdown: tokens.Breakdown = field(default_factory=tokens.Breakdown)
    estimated: int = 0            # оценка запроса до отправки
    prompt_tokens: int = 0        # факт по тому же запросу
    completion_tokens: int = 0    # факт по итоговому ответу
    total_prompt: int = 0         # факт по всем вызовам обращения (план, шаги, итог)
    total_completion: int = 0
    limit: int = 0                # окно контекста модели
    reserve: int = 0              # запас, оставленный под ответ
    max_output: int = 0           # потолок генерации у модели
    trimmed_pairs: int = 0        # сколько пар памяти выкинуто, чтобы влезть в окно
    truncated: bool = False       # ответ упёрся в лимит генерации и оборван
    strategy: str = ""            # стратегия контекста в этом обращении
    summarize: bool = False       # было ли включено сжатие истории
    branch: int = MAIN_BRANCH     # в какой ветке шёл разговор
    summary_version: int = 0      # какая суммаризация ушла в запрос (0 — без суммаризации)
    folded_messages: int = 0      # сколько сообщений она заменила
    folded_tokens: int = 0        # сколько они весили бы сами (оценка)
    facts_version: int = 0        # какой блок фактов ушёл в запрос (0 — без фактов)
    facts_items: int = 0          # сколько в нём фактов
    pending_messages: int = 0     # ждут суммаризации
    pending_tokens: int = 0
    facts_pending: int = 0        # ждут извлечения в блок фактов
    dropped_messages: int = 0     # сообщений истории, не ушедших в модель дословно и не свёрнутых
    dropped_tokens: int = 0       # сколько они весили бы в запросе

    @property
    def context_used(self) -> int:
        """Сколько токенов реально заняло окно контекста (факт, пока его нет — оценка)."""
        return self.prompt_tokens or self.estimated

    @property
    def fill(self) -> float:
        """Доля окна контекста, занятая запросом."""
        return self.context_used / self.limit if self.limit else 0.0

    @property
    def error_pct(self) -> float | None:
        """На сколько процентов оценка разошлась с фактом (со знаком)."""
        if not self.prompt_tokens or not self.estimated:
            return None
        return round((self.estimated - self.prompt_tokens) / self.prompt_tokens * 100, 1)

    @property
    def uncompressed(self) -> int:
        """Сколько весил бы тот же запрос с полной историей как есть.

        Без блоков стратегии, зато со всеми сообщениями, которые она свернула или
        отбросила: это и есть «до» для сравнения со стратегией.
        """
        return (self.context_used - self.breakdown.summary - self.breakdown.facts
                + self.folded_tokens + self.dropped_tokens)

    @property
    def saved(self) -> int:
        """Сколько токенов сберегла стратегия в этом запросе.

        Может быть и меньше нуля: пока суммаризация или факты молоды, они бывают
        тяжелее тех нескольких сообщений, которые заменили, — окупаются на длине.
        """
        return self.uncompressed - self.context_used

    def to_dict(self) -> dict:
        return {
            "breakdown": self.breakdown.to_dict(),
            "estimated": self.estimated,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_prompt": self.total_prompt,
            "total_completion": self.total_completion,
            "limit": self.limit,
            "reserve": self.reserve,
            "max_output": self.max_output,
            "trimmed_pairs": self.trimmed_pairs,
            "truncated": self.truncated,
            "strategy": self.strategy,
            "summarize": self.summarize,
            "branch": self.branch,
            "summary_version": self.summary_version,
            "folded_messages": self.folded_messages,
            "folded_tokens": self.folded_tokens,
            "facts_version": self.facts_version,
            "facts_items": self.facts_items,
            "pending_messages": self.pending_messages,
            "pending_tokens": self.pending_tokens,
            "facts_pending": self.facts_pending,
            "dropped_messages": self.dropped_messages,
            "dropped_tokens": self.dropped_tokens,
            "uncompressed": self.uncompressed,
            "saved": self.saved,
            "context_used": self.context_used,
            "fill": self.fill,
            "error_pct": self.error_pct,
        }


@dataclass
class AgentReply:
    """Ответ агента: текст, трасса выполненных шагов и метрики всего обращения."""
    text: str
    model: str
    turn: int
    plan: list[str] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    llm_calls: int = 1
    finish_reason: str | None = None
    usage: dict | None = None
    elapsed_s: float | None = None
    cost_usd: float | None = None
    temperature: float | None = None
    tier: str | None = None
    sent_messages: int = 0          # сколько сообщений ушло в модель в последнем вызове
    tokens: TokenReport | None = None   # счёт за обращение: оценка, факт и лимиты
    compression: Compression | None = None  # после ответа часть истории свёрнута в суммаризацию
    facts: FactsUpdate | None = None        # после ответа обновлён блок фактов
    shadow: Shadow | None = None    # теневой ответ «с полной историей» для сравнения
    request: dict | None = None     # «сырой обмен»: тело последнего запроса
    response: dict | None = None    # «сырой обмен»: ответ модели как есть

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model": self.model,
            "model_label": config.model_label(self.model),
            "turn": self.turn,
            "plan": self.plan,
            "steps": [s.to_dict() for s in self.steps],
            "llm_calls": self.llm_calls,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "elapsed_s": self.elapsed_s,
            "cost_usd": self.cost_usd,
            "temperature": self.temperature,
            "tier": self.tier,
            "sent_messages": self.sent_messages,
            "tokens": self.tokens.to_dict() if self.tokens else None,
            "compression": self.compression.to_dict() if self.compression else None,
            "facts": self.facts.to_dict() if self.facts else None,
            "shadow": self.shadow.to_dict() if self.shadow else None,
            "request": self.request,
            "response": self.response,
        }


def _main_branch() -> dict:
    """Описание основной ветки: у неё нет строки в хранилище, она есть всегда."""
    return {"id": MAIN_BRANCH, "name": MAIN_BRANCH_NAME, "origin": "", "shared": 0, "fork_at": 0}


@dataclass
class Agent:
    """Экземпляр агента: паспорт + настройки + память + инструменты + счётчики.

    Память диалога хранится здесь, в самом агенте: интерфейс присылает только
    очередное сообщение, а контекст для модели агент собирает сам. Агентов может
    быть несколько — они независимы, память одного не видна другому.

    Если агенту дали хранилище (`store`), он сам записывает туда каждое обращение
    и каждую смену настроек — и переживает перезапуск приложения. Без хранилища
    агент полностью работоспособен, просто помнит разговор только до закрытия;
    без хранилища нет только веток и точек ветвления — они живут в истории.
    """
    profile: AgentProfile = DEFAULT_PROFILE
    model: str = config.DEFAULT_MODEL
    temperature: float = config.AGENT_TEMPERATURE
    max_tokens: int | None = config.AGENT_MAX_TOKENS
    memory_turns: int = config.AGENT_MEMORY_TURNS
    tools_enabled: bool = config.AGENT_TOOLS      # давать ли модели инструменты
    planning: bool = config.AGENT_PLANNING        # писать ли план перед работой
    max_steps: int = config.AGENT_MAX_STEPS       # потолок шагов цикла за обращение
    strategy: str = config.AGENT_STRATEGY         # стратегия контекста: window / facts / branches
    summarize: bool = config.AGENT_SUMMARIZE      # сжимать ли историю — опция поверх любой стратегии
    summary_every: int = config.AGENT_SUMMARY_EVERY  # сколько сообщений копить до обновления суммаризации

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)
    turns: int = 0                                # сколько запросов агент обработал
    branch: int = MAIN_BRANCH                     # активная ветка диалога
    store: Store | None = field(default=None, repr=False)   # куда класть состояние
    restored: bool = False                        # поднят из истории, а не создан заново
    _memory: list[dict] = field(default_factory=list, repr=False)    # окно контекста: как есть
    # Две очереди, потому что сжатие и факты работают независимо и могут быть включены
    # одновременно: в первую попадает выпавшее из окна, во вторую — каждый новый обмен.
    _pending_summary: list[dict] = field(default_factory=list, repr=False)  # ждут суммаризации
    _pending_facts: list[dict] = field(default_factory=list, repr=False)    # ждут извлечения фактов
    _summary: Summary = field(default_factory=Summary, repr=False)   # свёрнутое начало разговора
    _facts: Facts = field(default_factory=Facts, repr=False)         # блок фактов «ключ — значение»
    _branch: dict = field(default_factory=_main_branch, repr=False)  # описание активной ветки

    # ------------------------------------------------------------------- вход --

    def ask(self, message: str, compare: bool = False) -> AgentReply:
        """Единственный публичный вход: принять запрос и вернуть ответ агента.

        `compare=True` — заодно получить теневой ответ на тот же вопрос, но с
        полной историей вместо стратегии контекста: два ответа и два счёта рядом.
        Настоящий ответ при этом один — тот, что идёт в память.
        """
        text = self._prepare(message)                       # 1. проверка входа
        started = time.perf_counter()
        totals = _Totals()
        steps: list[AgentStep] = []

        logger.info(
            "Агент «%s» [%s] ← запрос #%d (%d симв.) · инструменты=%s · план=%s · стратегия=%s · "
            "сжатие=%s · ветка=%s",
            self.profile.name, self.id, self.turns + 1, len(text), self.tools_enabled, self.planning,
            self.strategy, self.summarize, self._branch["name"],
        )

        plan = self._make_plan(text, totals)                # 2. план действий
        specs = tools.specs() if self.tools_enabled else None
        prompt = self._system_prompt(plan)
        summary_block = self._summary_block()               #    блоки стратегии внутри инструкции
        facts_block = self._facts_block()
        window, dropped = self._fit_context(prompt, text, specs)   # 3. бюджет контекста
        beyond, beyond_tokens = self._beyond_window()
        report = TokenReport(
            breakdown=tokens.measure(prompt, window, text, specs, self.model,
                                     summary=summary_block, facts=facts_block),
            limit=self._context_limit(),
            reserve=self._answer_reserve(),
            max_output=config.model_max_output(self.model),
            trimmed_pairs=dropped,
            strategy=self.strategy,
            summarize=self.summarize,
            branch=self.branch,
            summary_version=self._summary.version if summary_block else 0,
            folded_messages=self._summary.messages if summary_block else 0,
            folded_tokens=self._summary.tokens if summary_block else 0,
            facts_version=self._facts.version if facts_block else 0,
            facts_items=len(self._facts.items) if facts_block else 0,
            pending_messages=len(self._pending_summary),
            pending_tokens=tokens.measure_messages(self._pending_summary, self.model),
            facts_pending=len(self._pending_facts),
            dropped_messages=beyond,
            dropped_tokens=beyond_tokens,
        )
        report.estimated = report.breakdown.total
        messages = self._build_messages(prompt, window, text)      #    сборка запроса
        logger.info(
            "Агент «%s» [%s]: в запрос уйдёт ≈%d токенов (инструкция %d + суммаризация %d + факты %d "
            "+ память %d + вопрос %d + схемы %d) из окна %d · за окном %d сообщ. ≈ %d токенов",
            self.profile.name, self.id, report.estimated, report.breakdown.system,
            report.breakdown.summary, report.breakdown.facts, report.breakdown.memory,
            report.breakdown.question, report.breakdown.tools, report.limit, beyond, beyond_tokens,
        )

        raw, first_usage = None, None
        for _ in range(max(1, self.max_steps)):             # 4. цикл работы
            raw = totals.add(self._call(messages, specs))
            first_usage = first_usage or raw.get("usage")   #    факт по первому запросу
            if not raw["tool_calls"]:
                break                                       #    модель дала ответ
            messages.append(raw["message"])                 #    протокол требует вернуть
            for call in raw["tool_calls"]:                  #    запрос вызова в диалог
                step = self._run_tool(len(steps) + 1, call)
                steps.append(step)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": step.result})
        else:
            # Шаги кончились, а модель всё ещё зовёт инструменты — просим итог.
            logger.warning("Агент «%s» [%s]: исчерпан лимит в %d шагов", self.profile.name, self.id, self.max_steps)
            messages.append({"role": "system", "content": FINISH_NUDGE})
            raw = totals.add(self._call(messages, None))

        answer = self._postprocess(raw["content"], steps)   # 5. разбор ответа
        shadow = self._shadow(plan, text, specs, totals) if compare else None
        self.turns += 1
        pair = self._remember(text, answer)                 # 6. память: окно и очереди
        sticky = self._update_facts(totals)                 #    факты: обновить блок по новому обмену
        folded = self._compress(totals)                     #    сжатие: очередь набралась — свернуть
        self._close_report(report, totals, raw, first_usage)
        self._save(pair, report, totals, shadow, sticky)    #    история, состояние, расход — одной записью

        elapsed = time.perf_counter() - started
        logger.info(
            "Агент «%s» [%s] → ответ #%d за %.2f c · шагов=%d · вызовов модели=%d · в памяти %d сообщ., "
            "ждут суммаризации %d, ждут фактов %d · токены: оценка %d → факт %d (%s%%), ответ %d, "
            "всего за обращение %d",
            self.profile.name, self.id, self.turns, elapsed, len(steps), totals.llm_calls,
            len(self._memory), len(self._pending_summary), len(self._pending_facts), report.estimated,
            report.prompt_tokens, report.error_pct, report.completion_tokens,
            report.total_prompt + report.total_completion,
        )

        return AgentReply(
            text=answer,
            model=raw["model"],
            turn=self.turns,
            plan=plan,
            steps=steps,
            llm_calls=totals.llm_calls,
            finish_reason=raw["finish_reason"],
            usage=totals.usage(),
            elapsed_s=round(elapsed, 3),
            cost_usd=totals.cost_usd if totals.has_cost else None,
            temperature=raw["temperature"],
            tier=raw["tier"],
            sent_messages=len(messages),
            tokens=report,
            compression=folded,
            facts=sticky,
            shadow=shadow,
            request=raw["request"],
            response=raw["response"],
        )

    # ---------------------------------------------------------- шаги обработки --

    def _prepare(self, message: str) -> str:
        """Шаг 1. Проверить и нормализовать вход — до всякого обращения к API.

        Пустое и слишком длинное отсекаем здесь: вызов модели не должен уходить
        с заведомо непригодным запросом.
        """
        text = (message or "").strip()
        if not text:
            raise AgentError("Пустой запрос: агенту нечего обрабатывать.")
        if len(text) > config.AGENT_MAX_INPUT_CHARS:
            raise AgentError(
                f"Запрос слишком длинный: {len(text)} символов при лимите "
                f"{config.AGENT_MAX_INPUT_CHARS}."
            )
        return text

    def _make_plan(self, text: str, totals: _Totals) -> list[str]:
        """Шаг 2. Отдельным вызовом решить, нужен ли план, и получить его шаги.

        План — это не украшение: он попадает в system-сообщение исполнителя, и
        дальше агент идёт по нему. На простой вопрос планировщик возвращает пустой
        список — тогда лишней работы не будет.
        """
        if not self.planning:
            return []
        listing = (
            "\n".join(f"- {t.name}: {t.description}" for t in tools.TOOLS)
            if self.tools_enabled else "- инструментов нет, доступен только текстовый ответ"
        )
        try:
            raw = totals.add(self._call(
                [{"role": "system", "content": PLANNER_SYSTEM.format(tools=listing)},
                 {"role": "user", "content": text}],
                None,
            ))
        except AgentError as e:
            logger.warning("Агент «%s» [%s]: планировщик недоступен (%s) — работаем без плана",
                           self.profile.name, self.id, e)
            return []

        data = _extract_json(raw["content"])
        if not isinstance(data, dict):
            logger.warning("Агент «%s» [%s]: план не разобрать, работаем без него", self.profile.name, self.id)
            return []
        plan = [str(s).strip() for s in data.get("steps", []) if str(s).strip()][:6]
        logger.info("Агент «%s» [%s]: план из %d шагов", self.profile.name, self.id, len(plan))
        return plan

    def _build_messages(self, prompt: str, window: list[dict], text: str) -> list[dict]:
        """Собрать сообщения для модели: роль агента + план + память + новый вход.

        Здесь и видно отличие агента от голого вызова API: интерфейс прислал одну
        строку, а в модель уходит контекст, который агент собрал сам. `window` —
        это память, уже подрезанная под окно контекста (см. `_fit_context`); в
        запрос из неё идут только роль и текст — номера строк истории, которые
        агент держит при сообщениях для себя, модели не нужны.
        """
        return (
            [{"role": "system", "content": prompt}]
            + [{"role": m["role"], "content": m["content"]} for m in window]
            + [{"role": "user", "content": text}]
        )

    def _context_limit(self) -> int:
        """Окно контекста выбранной модели, токенов."""
        return config.model_context(self.model)

    def _answer_reserve(self) -> int:
        """Сколько токенов окна держим под ответ: контекст общий на запрос и ответ.

        Если лимит ответа задан явно — резервируем ровно его, иначе берём запас по
        умолчанию, но не больше того, что модель вообще способна сгенерировать. И
        в любом случае не больше половины окна: у модели с тесным контекстом запас
        под ответ иначе съел бы место под сам запрос.
        """
        wanted = int(self.max_tokens) if self.max_tokens else min(
            config.ANSWER_RESERVE, config.model_max_output(self.model)
        )
        return max(1, min(wanted, self._context_limit() // 2))

    def _fit_context(self, prompt: str, question: str, specs: list[dict] | None) -> tuple[list[dict], int]:
        """Уложить запрос в окно контекста, при нехватке — забыв самое старое.

        Это и есть поведение агента при переполнении. Ждать ошибки от API нельзя:
        она приходит после того, как запрос уже ушёл, и ничего не объясняет. Агент
        считает вес запроса сам и выкидывает из окна самые старые пары «вопрос-
        ответ», пока запрос не поместится, — ценой того, что начало разговора он
        забывает. История на диске при этом цела: подрезается только контекст.

        Если не помещается даже запрос без памяти (огромный вопрос, раздутая
        инструкция), честнее отказаться до вызова: платить за заведомо отклонённый
        запрос незачем.
        """
        room = self._context_limit() - self._answer_reserve()
        fixed = tokens.measure(prompt, [], question, specs, self.model).total
        if fixed > room:
            raise AgentError(
                f"Запрос не помещается в контекст модели: без памяти это уже ≈{fixed} токенов "
                f"при доступных {room} (окно {self._context_limit()} минус запас под ответ "
                f"{self._answer_reserve()}). Сократите вопрос, выключите инструменты или "
                f"возьмите модель с большим окном."
            )
        window = list(self._memory)
        dropped = 0
        while window and fixed + tokens.measure_messages(window, self.model) > room:
            del window[:2]      # самая старая пара «вопрос-ответ» уходит первой
            dropped += 1
        if dropped:
            logger.warning(
                "Агент «%s» [%s]: контекст переполнен — из памяти выброшено %d пар(ы), "
                "в запрос уйдут только последние %d сообщ.",
                self.profile.name, self.id, dropped, len(window),
            )
        return window, dropped

    def _close_report(
        self,
        report: TokenReport,
        totals: _Totals,
        raw: dict,
        first_usage: dict | None,
    ) -> None:
        """Дописать в счёт фактические токены и сверить с ними собственную оценку.

        Сверка нужна не для красоты: счётчик оценивает текст по символам, и без
        сравнения с `usage` невозможно понять, можно ли доверять его прогнозу и
        проверке на переполнение. Расхождение запоминается в калибровке модели —
        следующая оценка будет точнее.
        """
        if first_usage:
            report.prompt_tokens = first_usage["prompt_tokens"]
            tokens.observe(self.model, report.estimated, report.prompt_tokens)
        last_usage = raw.get("usage") or {}
        report.completion_tokens = last_usage.get("completion_tokens", 0)
        totals_usage = totals.usage() or {}
        report.total_prompt = totals_usage.get("prompt_tokens", 0)
        report.total_completion = totals_usage.get("completion_tokens", 0)
        report.truncated = raw.get("finish_reason") == "length"
        if report.truncated:
            logger.warning(
                "Агент «%s» [%s]: ответ оборван по лимиту генерации (%d токенов)",
                self.profile.name, self.id, report.completion_tokens,
            )

    def _system_prompt(
        self,
        plan: list[str],
        memory: list[dict] | None = None,
        blocks: bool = True,
    ) -> str:
        """Роль агента, дополненная правилами про инструменты, память стратегии и план.

        Роль пишет пользователь, и полагаться на неё в этих вопросах нельзя: про
        инструменты, суммаризацию, факты и собственную память агент рассказывает
        модели сам. `memory` — окно, которое пойдёт следом (по умолчанию своё),
        `blocks=False` собирает инструкцию без блоков стратегии — так строится
        теневой запрос «с полной историей».
        """
        window = self._memory if memory is None else memory
        prompt = self.profile.instructions
        if self.tools_enabled:
            prompt += TOOLS_NOTE
            prompt += f"\n\nРабочая папка для файловых инструментов: {tools.workspace()}"
        block = (self._summary_block() + self._facts_block()) if blocks else ""
        prompt += block
        if window:
            prompt += MEMORY_NOTE.format(count=len(window), when=self._last_seen())
        elif not block:
            prompt += NO_MEMORY_NOTE
        if plan:
            listed = "\n".join(f"{i}. {step}" for i, step in enumerate(plan, 1))
            prompt += (
                "\n\nТы сам составил план на эту задачу:\n" + listed +
                "\nСледуй ему. Если по ходу дела план оказался неверным — скажи об этом в ответе."
            )
        return prompt

    def _summary_block(self) -> str:
        """Суммаризация в том виде, в каком она уходит в инструкцию.

        Пусто, если сворачивать пока нечего или сжатие выключено: выключенное сжатие
        не стирает суммаризацию, а лишь перестаёт её подставлять — включат обратно,
        и она снова в деле.
        """
        if not self.summarize or not self._summary:
            return ""
        return SUMMARY_NOTE.format(count=self._summary.messages, summary=self._summary.text)

    def _facts_block(self) -> str:
        """Факты в том виде, в каком они уходят в инструкцию (пусто вне стратегии facts)."""
        if self.strategy != "facts" or not self._facts:
            return ""
        return FACTS_NOTE.format(count=len(self._facts.items), facts=self._facts.text())

    def _call(
        self,
        messages: list[dict],
        specs: list[dict] | None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Один вызов модели через транспорт; сбой превращаем в ошибку агента.

        Температура и лимит ответа по умолчанию — настройки агента; служебные
        вызовы (суммаризация, факты) передают свои.
        """
        try:
            return llm.chat(
                messages,
                model=self.model,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                tools=specs,
            )
        except Exception as e:  # 401/403/429, сеть и прочее
            logger.warning("Агент «%s» [%s]: вызов модели не удался: %s", self.profile.name, self.id, e)
            text = str(e)
            if any(marker in text.lower() for marker in LENGTH_ERROR_MARKERS):
                # Переполнение, которое проскочило собственную проверку: оценка по
                # символам не токенайзер и может ошибиться. Показываем не «400», а
                # то, из-за чего запрос отклонён.
                weight = tokens.measure_request(messages, specs, self.model)
                raise AgentError(
                    f"Модель отклонила запрос по длине: в него ушло ≈{weight} токенов при окне "
                    f"{self._context_limit()} и лимите ответа {self._answer_reserve()}. "
                    f"Уменьшите глубину памяти, сократите вопрос или начните разговор заново.\n"
                    f"Ответ провайдера: {text}"
                ) from e
            raise AgentError(f"Модель не ответила: {e}") from e

    def _run_tool(self, number: int, call: dict) -> AgentStep:
        """Шаг 3. Выполнить то, что попросила модель, и записать результат в трассу.

        Ошибку инструмента не поднимаем наверх: она возвращается модели как
        результат, и агент получает шанс исправиться на следующем шаге.
        """
        name = call["name"]
        try:
            arguments = json.loads(call["arguments"] or "{}")
        except (json.JSONDecodeError, TypeError):
            arguments = {"_raw": call["arguments"]}

        started = time.perf_counter()
        try:
            result, ok = tools.call(name, call["arguments"]), True
        except tools.ToolError as e:
            result, ok = f"Ошибка инструмента: {e}", False
        elapsed = round(time.perf_counter() - started, 3)

        logger.info(
            "Агент «%s» [%s] · шаг %d: %s(%s) → %s за %.2f c",
            self.profile.name, self.id, number, name,
            ", ".join(f"{k}={_short(v)}" for k, v in arguments.items()),
            "ок" if ok else "ошибка", elapsed,
        )
        tool = tools.BY_NAME.get(name)
        return AgentStep(
            number=number,
            tool=name,
            title=tool.title if tool else name,
            arguments=arguments,
            result=result,
            ok=ok,
            elapsed_s=elapsed,
        )

    def _postprocess(self, content: str, steps: list[AgentStep]) -> str:
        """Шаг 4. Привести ответ модели к тому, что агент готов отдать наружу.

        Если итога нет, но шаги выполнены (упёрлись в лимит, модель не подвела
        черту), обращение не роняем: работа уже сделана, и честнее показать, что
        именно успел агент, чем отдать ошибку.
        """
        answer = (content or "").strip()
        if answer:
            return answer
        if steps:
            done = ", ".join(f"{s.tool}{'' if s.ok else ' (ошибка)'}" for s in steps)
            return ("Итог не сформулирован — модель остановилась без финального ответа. "
                    f"Выполненные шаги: {done}.")
        raise AgentError(
            "Модель вернула пустой ответ — возможно, генерация упёрлась в ограничение длины."
        )

    def _remember(self, question: str, answer: str) -> list[dict]:
        """Шаг 6. Запомнить обмен в окне контекста.

        В памяти держим только вопрос и итоговый ответ: служебная переписка с
        инструментами нужна внутри одного обращения, а в долгой памяти она бы
        быстро съела контекст и деньги. Со стратегией фактов пара сразу встаёт в
        очередь на извлечение; что при этом выпало из окна, уходит в очередь на
        суммаризацию (см. `_trim_memory`) — если сжатие включено.
        """
        pair = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        self._memory.extend(pair)
        if self.strategy == "facts":
            self._pending_facts.extend(pair)  # факты извлекаются из каждого обмена, а не из выпавшего
        self._trim_memory()
        return pair

    def _save(
        self,
        pair: list[dict],
        report: TokenReport,
        totals: _Totals,
        shadow: Shadow | None = None,
        sticky: FactsUpdate | None = None,
    ) -> None:
        """Записать обращение в хранилище: пара сообщений, состояние агента, расход и факты.

        Одной записью, чтобы история, счётчик обращений и токены не разъезжались.
        Строка расхода и превращает «сколько стоило» в наблюдаемую величину: по
        ней видно, как цена растёт от обращения к обращению — и как перестаёт
        расти, когда в дело вступает стратегия. Обновлённый блок фактов пишется
        той же транзакцией: его граница — последнее сообщение этой пары.
        """
        if self.store is None:
            return
        facts_row = None
        if sticky is not None:
            facts_row = {
                "version": self._facts.version,
                "turn": self.turns,
                "upto": self._facts.upto,
                "items": len(self._facts.items),
                "facts_tokens": sticky.tokens,
                "content": json.dumps(self._facts.items, ensure_ascii=False),
            }
        ids = self.store.save_turn(self.state(), pair, self._usage_row(report, totals, shadow), facts=facts_row)
        for message, number in zip(pair, ids):
            message["id"] = number   # окно помнит, какой строке истории отвечает сообщение
        if sticky is not None and ids:
            self._facts.upto = ids[-1]

    def _usage_row(self, report: TokenReport, totals: _Totals, shadow: Shadow | None = None) -> dict:
        """Расход обращения одной строкой — то, из чего потом рисуется рост цены."""
        return {
            "turn": self.turns,
            "model": self.model,
            "prompt_tokens": report.total_prompt,
            "completion_tokens": report.total_completion,
            "total_tokens": report.total_prompt + report.total_completion,
            "cost_usd": totals.cost_usd if totals.has_cost else None,
            "llm_calls": totals.llm_calls,
            "estimated": report.estimated,
            "context_tokens": report.context_used,
            "memory_tokens": report.breakdown.memory,
            "context_limit": report.limit,
            "trimmed_pairs": report.trimmed_pairs,
            "summary_tokens": report.breakdown.summary,
            "folded_messages": report.folded_messages,
            "folded_tokens": report.folded_tokens,
            "shadow_tokens": shadow.prompt_tokens + shadow.completion_tokens if shadow else 0,
            "strategy": self.strategy,
            "summarize": int(self.summarize),
            "branch": self.branch,
            "facts_tokens": report.breakdown.facts,
            "dropped_messages": report.dropped_messages,
            "dropped_tokens": report.dropped_tokens,
        }

    def _trim_memory(self) -> None:
        """Оставить в окне последние `memory_turns` пар; выпавшее — по правилам сжатия.

        Без сжатия выпавшие сообщения просто перестают уходить в модель (в истории
        они остаются; фактам они и не нужны — блок уже собран из них). Со сжатием
        они встают в очередь и ждут, пока наберётся на обновление суммаризации.
        """
        extra = len(self._memory) - max(0, self.memory_turns) * 2
        if extra > 0:
            overflow = self._memory[:extra]
            del self._memory[:extra]
            if self.summarize:
                self._pending_summary.extend(overflow)

    def _compress(self, totals: _Totals) -> Compression | None:
        """Свернуть очередь в суммаризацию, если она набралась.

        Это и есть сжатие истории: сообщения, выпавшие из окна, не выбрасываются и
        не уходят в модель дословно, а превращаются отдельным вызовом в короткий
        список фактов. Вызов устроен как у планировщика — своя роль, низкая
        температура, строгий формат. Прежняя суммаризация подаётся на вход, поэтому
        новая — суммаризация всего разговора, а не только последних сообщений.

        Сжатие — опция, а не стратегия: оно включается поверх любой из них, поэтому
        сюда агент заходит и со скользящим окном, и с фактами, и в ветке.

        Сбой здесь не роняет обращение: ответ пользователь уже получил, а очередь
        подождёт следующего раза.
        """
        if not self.summarize:
            return None
        # С хранилищем сворачиваем только то, что уже записано в историю (у таких
        # сообщений есть номер строки): по нему после перезапуска видно, что уже в
        # суммаризации, а что ещё нет. Без хранилища сворачиваем всё, что накопилось.
        batch = [m for m in self._pending_summary if self.store is None or m.get("id")]
        if len(batch) < max(1, self.summary_every):
            return None

        listing = "\n".join(
            f"[{'пользователь' if m['role'] == 'user' else 'агент'}] {m['content']}" for m in batch
        )
        started = time.perf_counter()
        try:
            raw = totals.add(self._call(
                [{"role": "system", "content": SUMMARY_SYSTEM.format(
                    name=self.profile.name, points=config.SUMMARY_POINTS)},
                 {"role": "user", "content": SUMMARY_USER.format(
                     summary=self._summary.text or "(пока пуста)", count=len(batch), messages=listing)}],
                None,
                temperature=config.SUMMARY_TEMPERATURE,
                max_tokens=config.SUMMARY_MAX_TOKENS,
            ))
        except AgentError as e:
            logger.warning("Агент «%s» [%s]: суммаризация не обновлена (%s) — очередь подождёт",
                           self.profile.name, self.id, e)
            return None
        text = (raw["content"] or "").strip()
        if not text:
            logger.warning("Агент «%s» [%s]: модель вернула пустую суммаризацию — оставляем прежнюю",
                           self.profile.name, self.id)
            return None

        before = tokens.measure_messages(batch, self.model)
        after = tokens.measure_text(text, self.model)
        self._summary = Summary(
            text=text,
            version=self._summary.version + 1,
            upto=max([self._summary.upto] + [m.get("id") or 0 for m in batch]),
            messages=self._summary.messages + len(batch),
            tokens=self._summary.tokens + before,
            at=time.time(),
        )
        taken = {id(m) for m in batch}
        self._pending_summary = [m for m in self._pending_summary if id(m) not in taken]
        if self.store is not None:
            self.store.save_summary(self.id, {
                "version": self._summary.version,
                "turn": self.turns,
                "upto": self._summary.upto,
                "folded_messages": self._summary.messages,
                "folded_tokens": self._summary.tokens,
                "summary_tokens": after,
                "content": text,
            }, self.branch)
        usage = raw.get("usage") or {}
        logger.info(
            "Агент «%s» [%s]: %d сообщ. ≈ %d токенов свёрнуты в суммаризацию №%d ≈ %d токенов "
            "(всего заменяет %d сообщ. ≈ %d токенов)",
            self.profile.name, self.id, len(batch), before, self._summary.version, after,
            self._summary.messages, self._summary.tokens,
        )
        return Compression(
            messages=len(batch),
            before=before,
            after=after,
            version=self._summary.version,
            text=text,
            elapsed_s=round(time.perf_counter() - started, 3),
            call_tokens=usage.get("total_tokens", 0),
        )

    def _update_facts(self, totals: _Totals) -> FactsUpdate | None:
        """Обновить блок фактов по новым сообщениям (стратегия facts).

        Каждый обмен «вопрос-ответ» уходит отдельным вызовом модели вместе с
        текущим блоком; обратно приходит блок целиком — так изменившийся факт
        заменяется, отменённый исчезает, а важная деталь остаётся в запросе, даже
        когда сообщение с ней давно выпало из окна. Очередь обрабатывается с
        начала порциями по `FACTS_BATCH`: включили стратегию на длинном разговоре —
        факты догоняют историю за несколько обращений.

        Сбой не роняет обращение: ответ уже получен, очередь подождёт следующего.
        """
        if self.strategy != "facts" or not self._pending_facts:
            return None
        batch = self._pending_facts[:config.FACTS_BATCH]
        listing = "\n".join(
            f"[{'пользователь' if m['role'] == 'user' else 'агент'}] {m['content']}" for m in batch
        )
        started = time.perf_counter()
        try:
            raw = totals.add(self._call(
                [{"role": "system", "content": FACTS_SYSTEM.format(
                    name=self.profile.name, limit=config.FACTS_LIMIT)},
                 {"role": "user", "content": FACTS_USER.format(
                     facts=json.dumps({"facts": self._facts.items}, ensure_ascii=False)
                     if self._facts else "(пока пусто)",
                     count=len(batch), messages=listing)}],
                None,
                temperature=config.FACTS_TEMPERATURE,
                max_tokens=config.FACTS_MAX_TOKENS,
            ))
        except AgentError as e:
            logger.warning("Агент «%s» [%s]: факты не обновлены (%s) — очередь подождёт",
                           self.profile.name, self.id, e)
            return None
        items = _parse_facts(raw["content"])
        if items is None or (not items and self._facts):
            # Не разобрать или пустой блок при непустых фактах: скорее сбой формата,
            # чем «все факты отменены». Прежний блок остаётся.
            logger.warning("Агент «%s» [%s]: блок фактов не разобрать — оставляем прежний",
                           self.profile.name, self.id)
            return None

        before = dict(self._facts.items)
        added = [f"{key}: {value}" for key, value in items.items() if key not in before]
        changed = [f"{key}: {before[key]} → {value}" for key, value in items.items()
                   if key in before and before[key] != value]
        removed = [key for key in before if key not in items]
        self._facts = Facts(items=items, version=self._facts.version + 1, upto=self._facts.upto, at=time.time())
        del self._pending_facts[:len(batch)]
        weight = tokens.measure_text(self._facts_block(), self.model)
        usage = raw.get("usage") or {}
        logger.info(
            "Агент «%s» [%s]: факты №%d — %d шт. ≈ %d токенов (+%d, ~%d, −%d) по %d сообщ.",
            self.profile.name, self.id, self._facts.version, len(items), weight,
            len(added), len(changed), len(removed), len(batch),
        )
        return FactsUpdate(
            messages=len(batch),
            version=self._facts.version,
            items=len(items),
            tokens=weight,
            added=added,
            changed=changed,
            removed=removed,
            text=self._facts.text(),
            elapsed_s=round(time.perf_counter() - started, 3),
            call_tokens=usage.get("total_tokens", 0),
        )

    def _shadow(
        self,
        plan: list[str],
        text: str,
        specs: list[dict] | None,
        totals: _Totals,
    ) -> Shadow:
        """Теневой ответ для сравнения: тот же вопрос, но вся история как есть.

        Блоки стратегии в запрос не идут, вместо них — все сообщения истории
        активной ветки, сколько влезает в окно модели. Ответ не запоминается и на
        разговор не влияет: это измерение, а не обращение. Стоит он по-настоящему,
        поэтому считается в расход обращения вместе с остальными вызовами.
        """
        history = self._full_history()
        prompt = self._system_prompt(plan, memory=history, blocks=False)
        room = self._context_limit() - self._answer_reserve()
        fixed = tokens.measure(prompt, [], text, specs, self.model).total
        window, dropped = list(history), 0
        while window and fixed + tokens.measure_messages(window, self.model) > room:
            del window[:2]
            dropped += 1
        breakdown = tokens.measure(prompt, window, text, specs, self.model)

        started = time.perf_counter()
        raw = totals.add(self._call(self._build_messages(prompt, window, text), specs))
        usage = raw.get("usage") or {}
        content = (raw["content"] or "").strip()
        if raw["tool_calls"] and not content:
            content = (
                "Модель попросила инструменты (" + ", ".join(c["name"] for c in raw["tool_calls"]) +
                "): теневой прогон их не исполняет, сравнивать здесь можно только контекст."
            )
        logger.info(
            "Агент «%s» [%s]: теневой ответ с полной историей — %d сообщ., %d→%d токенов",
            self.profile.name, self.id, len(window), usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
        return Shadow(
            text=content,
            messages=len(window),
            breakdown=breakdown,
            prompt_tokens=usage.get("prompt_tokens", breakdown.total),
            completion_tokens=usage.get("completion_tokens", 0),
            cost_usd=raw.get("cost_usd"),
            elapsed_s=round(time.perf_counter() - started, 3),
            trimmed_pairs=dropped,
            tool_calls=len(raw["tool_calls"]),
            finish_reason=raw["finish_reason"],
        )

    def _full_history(self) -> list[dict]:
        """Вся переписка активной ветки как есть — то, что ушло бы в модель без стратегии."""
        if self.store is not None:
            return [_slim(m) for m in self.store.messages(self.id, self.branch)]
        # Без хранилища свёрнутое уже не вернуть: есть только очередь и окно.
        return [_slim(m) for m in self._pending_summary + self._memory]

    def _beyond_window(self, history: list[dict] | None = None) -> tuple[int, int]:
        """Сколько сообщений истории в модель дословно не уходит — и сколько они весили бы.

        Это то, с чем работают стратегия и сжатие: скользящее окно эти сообщения
        отбрасывает, факты заменяют своим блоком, сжатие ставит их в очередь на
        суммаризацию. Уже свёрнутое не считается — оно в запросе представлено
        суммаризацией.
        """
        if self.store is None:
            rest = self._pending_summary if self.summarize else []
        else:
            history = self._full_history() if history is None else history
            in_window = {m["id"] for m in self._memory if m.get("id")}
            upto = self._summary.upto if self._summary_block() else 0
            rest = [m for m in history if (m.get("id") or 0) > upto and m.get("id") not in in_window]
        return len(rest), tokens.measure_messages(rest, self.model)

    def _load_memory(self) -> None:
        """Собрать окно контекста и обе очереди из истории активной ветки.

        Вызывается при восстановлении агента, при смене глубины памяти, стратегии,
        сжатия или ветки: увеличили глубину — агент дотягивает из истории то, что
        уже забыл; включили сжатие — всё, что за окном и ещё не в суммаризации,
        встаёт в очередь на неё; включили факты — в очередь встаёт всё, что в блоке
        фактов ещё не учтено. Свёрнутое в суммаризацию в окно не возвращается:
        платить за это дважды незачем.
        """
        if self.store is None:
            if not self.summarize:
                self._pending_summary.clear()
            if self.strategy != "facts":
                self._pending_facts.clear()
            self._trim_memory()
            return
        history = [_slim(m) for m in self.store.messages(self.id, self.branch)]
        if self.summarize:
            history = [m for m in history if m["id"] > self._summary.upto]
        keep = max(0, self.memory_turns) * 2
        window = history[-keep:] if keep else []
        rest = history[:len(history) - len(window)]
        self._memory = window
        self._pending_summary = rest if self.summarize else []
        self._pending_facts = (
            [m for m in history if m["id"] > self._facts.upto] if self.strategy == "facts" else []
        )

    def _load_summary(self) -> None:
        """Поднять действующую суммаризацию ветки из хранилища (последнюю версию)."""
        if self.store is None:
            return
        row = self.store.summary(self.id, self.branch)
        self._summary = Summary(
            text=row["content"],
            version=row["version"],
            upto=row["upto"],
            messages=row["folded_messages"],
            tokens=row["folded_tokens"],
            at=row["at"],
        ) if row else Summary()

    def _load_facts(self) -> None:
        """Поднять действующий блок фактов ветки из хранилища (последнюю версию)."""
        if self.store is None:
            return
        row = self.store.facts(self.id, self.branch)
        if not row:
            self._facts = Facts()
            return
        try:
            items = json.loads(row["content"] or "{}")
        except json.JSONDecodeError:
            items = {}
        self._facts = Facts(
            items={str(k): str(v) for k, v in items.items()} if isinstance(items, dict) else {},
            version=row["version"],
            upto=row["upto"],
            at=row["at"],
        )

    def _load_branch(self) -> None:
        """Найти описание активной ветки; нет такой — вернуться в основную."""
        if self.branch == MAIN_BRANCH or self.store is None:
            self.branch = MAIN_BRANCH
            self._branch = _main_branch()
            return
        row = next((b for b in self.store.branches(self.id) if b["id"] == self.branch), None)
        if row is None:
            logger.warning("Агент [%s]: ветки %d больше нет — открываем основную", self.id, self.branch)
            self.branch = MAIN_BRANCH
            self._branch = _main_branch()
            return
        self._branch = {"id": row["id"], "name": row["name"], "origin": row["origin"],
                        "shared": row["shared"], "fork_at": row["fork_at"]}

    def _last_seen(self) -> str:
        """Когда в памяти появилось последнее сообщение — для заметки модели."""
        moment = self.store.last_at(self.id, self.branch) if self.store is not None else None
        return time.strftime("%d.%m.%Y %H:%M", time.localtime(moment)) if moment else "недавно"

    # ------------------------------------------------------------ управление им --

    def configure(
        self,
        model: str | None = None,
        temperature: float | None = None,
        memory_turns: int | None = None,
        tools_enabled: bool | None = None,
        planning: bool | None = None,
        max_steps: int | None = None,
        max_tokens: int | None = None,
        strategy: str | None = None,
        summarize: bool | None = None,
        summary_every: int | None = None,
    ) -> None:
        """Изменить настройки агента.

        Проверки живут здесь, а не в интерфейсе: настройки — часть самого агента,
        и любой интерфейс поверх него получает их бесплатно.
        """
        if strategy is not None:
            if strategy not in config.STRATEGY_BY_CODE:
                raise AgentError(
                    f"Стратегия «{strategy}» неизвестна. Доступны: "
                    + ", ".join(s["code"] for s in config.STRATEGIES) + "."
                )
            self.strategy = strategy
            self._load_memory()  # стратегия сменилась: очереди собираются заново под её правила
        if summarize is not None:
            self.summarize = bool(summarize)
            # Включили сжатие на живом агенте — всё, что за окном и ещё не свёрнуто,
            # встаёт в очередь и свернётся после следующего ответа; выключили —
            # очередь расходится, а сама суммаризация остаётся в базе.
            self._load_memory()
        if summary_every is not None:
            if not SUMMARY_EVERY_MIN <= summary_every <= SUMMARY_EVERY_MAX:
                raise AgentError(
                    f"Суммаризация обновляется, когда за окном памяти накопится N сообщений: "
                    f"N должно быть от {SUMMARY_EVERY_MIN} до {SUMMARY_EVERY_MAX}."
                )
            self.summary_every = int(summary_every)
        if model is not None:
            if not config.is_allowed_model(model):
                # Чужая модель — риск реальных списаний после free-квоты: в реестре
                # config.MODELS только модели со Stop-on-Exhaust.
                raise AgentError(f"Модель «{model}» не в безопасном списке. Выберите модель из списка.")
            self.model = model
            ceiling = config.model_max_output(model)
            if self.max_tokens and self.max_tokens > ceiling:
                # У новой модели потолок генерации может быть ниже — иначе первый же
                # вызов вернул бы «Range of max_tokens should be [1, N]».
                logger.info("Лимит ответа %d выше потолка модели — уменьшен до %d", self.max_tokens, ceiling)
                self.max_tokens = ceiling
        if temperature is not None:
            if not 0 <= temperature < 2:
                raise AgentError("Температура должна быть в диапазоне [0, 2).")
            self.temperature = float(temperature)
        if memory_turns is not None:
            if not 0 <= memory_turns <= MEMORY_TURNS_MAX:
                raise AgentError(
                    f"Глубина памяти должна быть от 0 до {MEMORY_TURNS_MAX} пар сообщений."
                )
            self.memory_turns = int(memory_turns)
            self._load_memory()  # глубину увеличили — доберём забытое из истории
        if tools_enabled is not None:
            self.tools_enabled = bool(tools_enabled)
        if planning is not None:
            self.planning = bool(planning)
        if max_steps is not None:
            if not 1 <= max_steps <= 12:
                raise AgentError("Потолок шагов должен быть от 1 до 12.")
            self.max_steps = int(max_steps)
        if max_tokens is not None:
            ceiling = config.model_max_output(self.model)
            if max_tokens and not 1 <= max_tokens <= ceiling:
                raise AgentError(
                    f"Лимит ответа должен быть от 1 до {ceiling} токенов: столько модель "
                    f"«{config.model_label(self.model)}» способна сгенерировать за раз."
                )
            self.max_tokens = int(max_tokens) or None   # ноль означает «без ограничения»
        logger.info(
            "Агент «%s» [%s]: настройки — модель=%s, t°=%s, память=%d пар, инструменты=%s, "
            "план=%s, шагов=%d, лимит ответа=%s, стратегия=%s, сжатие=%s (суммаризация каждые %d сообщ.)",
            self.profile.name, self.id, self.model, self.temperature, self.memory_turns,
            self.tools_enabled, self.planning, self.max_steps, self.max_tokens or "без ограничения",
            self.strategy, "вкл" if self.summarize else "выкл", self.summary_every,
        )
        self.persist()  # настройки тоже переживают перезапуск

    def set_profile(
        self,
        name: str | None = None,
        role: str | None = None,
        instructions: str | None = None,
    ) -> None:
        """Сменить паспорт: имя, подпись роли и system-инструкцию.

        Память и история остаются — это тот же агент, просто теперь он ведёт себя
        иначе. Пустое поле означает «оставить как было», пустая инструкция —
        вернуться к инструкции по умолчанию.
        """
        current = self.profile
        self.profile = AgentProfile(
            name=(name if name is not None else current.name).strip()[:40] or current.name,
            role=(role if role is not None else current.role).strip() or current.role,
            instructions=(instructions if instructions is not None else current.instructions).strip()
            or DEFAULT_PROFILE.instructions,
        )
        self.persist()
        logger.info("Агент [%s]: паспорт обновлён — «%s», %s", self.id, self.profile.name, self.profile.role)

    def reset(self) -> None:
        """Забыть диалог — и в памяти, и в истории на диске, во всех ветках.

        Паспорт и настройки остаются: агент тот же самый, просто без прошлого.
        Стирать историю здесь важно, иначе после перезапуска забытое вернулось бы.
        """
        self._memory.clear()
        self._pending_summary.clear()
        self._pending_facts.clear()
        self._summary = Summary()
        self._facts = Facts()
        self.turns = 0
        self.branch = MAIN_BRANCH
        self._branch = _main_branch()
        if self.store is not None:
            self.store.forget(self.id)
        self.persist()
        logger.info("Агент «%s» [%s]: память и история очищены", self.profile.name, self.id)

    # ------------------------------------------------------------ ветки диалога --

    def checkpoint(self, name: str = "") -> dict:
        """Поставить точку ветвления после последнего сообщения активной ветки.

        Точка — место в истории, от которого создаются ветки. Обычно её ставит сам
        `fork()` в момент ветвления, и тогда она остаётся в истории как адрес: от неё
        можно отвести ещё одну ветку, когда разговор уже ушёл дальше. Разговор точка
        не меняет — только запоминает границу.
        """
        self._need_store("точек ветвления")
        upto = self.store.last_id(self.id, self.branch)
        if not upto:
            raise AgentError("Точка ветвления ставится в разговоре: в этой ветке пока нет сообщений.")
        count = self.store.count(self.id, self.branch)
        name = _clean_name(name) or f"Точка {len(self.checkpoints()) + 1}"
        number = self.store.add_checkpoint(self.id, self.branch, upto, count, name)
        logger.info("Агент «%s» [%s]: точка ветвления «%s» после %d сообщ. ветки «%s»",
                    self.profile.name, self.id, name, count, self._branch["name"])
        return {"id": number, "branch": self.branch, "upto": upto, "messages": count, "name": name}

    def fork(self, name: str = "", checkpoint: int | None = None) -> dict:
        """Создать ветку от точки ветвления и переключиться на неё.

        Без точки ветка отходит от текущего места: точка ставится тут же. Ветка
        получает копию общего начала разговора (сообщения до точки, а с ними
        суммаризацию и факты на тот момент) и дальше живёт независимо: её
        сообщения, суммаризации и факты в другие ветки не попадают.
        """
        self._need_store("веток")
        if checkpoint is None:
            point = self.checkpoint()
        else:
            point = next((c for c in self.checkpoints() if c["id"] == checkpoint), None)
            if point is None:
                raise AgentError("Такой точки ветвления нет — возможно, её ветка удалена.")
        name = _clean_name(name) or f"Ветка {len(self.store.branches(self.id)) + 1}"
        branch = self.store.fork(
            self.id, point["branch"], point["upto"], name, origin=point["name"], checkpoint=point["id"]
        )
        self.switch_branch(branch)
        logger.info("Агент «%s» [%s]: ветка «%s» [%d] от точки «%s» (%d общих сообщ.)",
                    self.profile.name, self.id, name, branch, point["name"], point["messages"])
        return {"id": branch, "name": name, "origin": point["name"], "shared": point["messages"]}

    def switch_branch(self, branch: int) -> None:
        """Переключиться на ветку: память, суммаризация и факты собираются из её истории."""
        self._need_store("веток")
        if not self.store.branch_exists(self.id, int(branch)):
            raise AgentError("Такой ветки нет.")
        self.branch = int(branch)
        self._load_branch()
        self._load_summary()
        self._load_facts()
        self._load_memory()
        self.persist()   # следующий запуск откроет ту же ветку
        logger.info("Агент «%s» [%s]: активна ветка «%s» — %d сообщ. в памяти из %d",
                    self.profile.name, self.id, self._branch["name"], len(self._memory), self.history_size)

    def delete_branch(self, branch: int) -> None:
        """Удалить ветку вместе с её перепиской; основную удалить нельзя."""
        self._need_store("веток")
        if int(branch) == MAIN_BRANCH:
            raise AgentError("Основную ветку удалить нельзя — это сама история агента.")
        if int(branch) == self.branch:
            self.switch_branch(MAIN_BRANCH)
        self.store.remove_branch(self.id, int(branch))

    def branches(self) -> list[dict]:
        """Все ветки агента, начиная с основной; у активной `active` = True."""
        main = _main_branch()
        main["checkpoint"] = 0
        main["at"] = self.created_at
        main["messages"] = self.history_size if self.branch == MAIN_BRANCH else (
            self.store.count(self.id, MAIN_BRANCH) if self.store is not None else 0
        )
        rows = [main] + (self.store.branches(self.id) if self.store is not None else [])
        for row in rows:
            row["active"] = row["id"] == self.branch
        return rows

    def checkpoints(self) -> list[dict]:
        """Точки ветвления агента во всех ветках."""
        return self.store.checkpoints(self.id) if self.store is not None else []

    def _need_store(self, what: str) -> None:
        if self.store is None:
            raise AgentError(f"Без хранилища нет {what}: ветки живут в истории на диске.")

    # ------------------------------------------------- состояние между запусками --

    def state(self) -> dict:
        """Всё, чем агент является, кроме переписки: её ведёт сама история.

        Это то, что уходит в хранилище. Обратная операция — `restore()`.
        """
        return {
            "id": self.id,
            "created_at": self.created_at,
            "turns": self.turns,
            "branch": self.branch,
            "profile": {
                "name": self.profile.name,
                "role": self.profile.role,
                "instructions": self.profile.instructions,
            },
            "settings": {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "memory_turns": self.memory_turns,
                "tools_enabled": self.tools_enabled,
                "planning": self.planning,
                "max_steps": self.max_steps,
                "strategy": self.strategy,
                "summarize": self.summarize,
                "summary_every": self.summary_every,
            },
        }

    @classmethod
    def restore(cls, state: dict, store: Store) -> "Agent":
        """Поднять агента из сохранённого состояния вместе с его памятью.

        К значениям из файла относимся как к чужим: чего нет — берём по
        умолчанию, модель вне безопасного реестра меняем на модель по умолчанию.
        Приложение должно запускаться даже с устаревшим или правленым файлом.
        """
        profile = state.get("profile") or {}
        settings = state.get("settings") or {}
        model = settings.get("model", config.DEFAULT_MODEL)
        if not config.is_allowed_model(model):
            logger.warning("В истории модель «%s» вне реестра — берём %s", model, config.DEFAULT_MODEL)
            model = config.DEFAULT_MODEL
        strategy, summarize = _strategy_and_summarize(settings)

        agent = cls(
            profile=AgentProfile(
                name=profile.get("name") or DEFAULT_PROFILE.name,
                role=profile.get("role") or DEFAULT_PROFILE.role,
                instructions=profile.get("instructions") or DEFAULT_PROFILE.instructions,
            ),
            model=model,
            temperature=float(settings.get("temperature", config.AGENT_TEMPERATURE)),
            max_tokens=settings.get("max_tokens", config.AGENT_MAX_TOKENS),
            memory_turns=int(settings.get("memory_turns", config.AGENT_MEMORY_TURNS)),
            tools_enabled=bool(settings.get("tools_enabled", config.AGENT_TOOLS)),
            planning=bool(settings.get("planning", config.AGENT_PLANNING)),
            max_steps=int(settings.get("max_steps", config.AGENT_MAX_STEPS)),
            strategy=strategy,
            summarize=summarize,
            summary_every=int(settings.get("summary_every") or config.AGENT_SUMMARY_EVERY),
            id=state.get("id") or uuid.uuid4().hex[:8],
            created_at=float(state.get("created_at") or time.time()),
            turns=int(state.get("turns") or 0),
            branch=int(state.get("branch") or MAIN_BRANCH),
            store=store,
            restored=True,
        )
        agent._load_branch()    # сначала ветка: остальное читается из её истории
        agent._load_summary()   # суммаризация раньше окна: от её границы зависит, что войдёт в окно
        agent._load_facts()
        agent._load_memory()
        logger.info(
            "Агент «%s» [%s] восстановлен: %d обращений, ветка «%s», %d сообщ. в памяти из %d в истории, "
            "стратегия %s, сжатие %s, суммаризация №%d заменяет %d сообщ., фактов %d, "
            "ждут суммаризации %d, ждут фактов %d",
            agent.profile.name, agent.id, agent.turns, agent._branch["name"], len(agent._memory),
            agent.history_size, agent.strategy, "вкл" if agent.summarize else "выкл",
            agent._summary.version, agent._summary.messages, len(agent._facts.items),
            len(agent._pending_summary), len(agent._pending_facts),
        )
        return agent

    def persist(self) -> None:
        """Записать состояние агента в хранилище (без хранилища — ничего не делаем)."""
        if self.store is not None:
            self.store.save_agent(self.state())

    def mark_active(self) -> None:
        """Запомнить, что разговор идёт с этим агентом: его и откроет следующий запуск."""
        if self.store is not None:
            self.store.set_active(self.id)

    def erase(self) -> None:
        """Убрать агента из хранилища: после перезапуска его не будет."""
        if self.store is not None:
            self.store.remove_agent(self.id)

    # --------------------------------------------------------- для интерфейса --

    @property
    def memory(self) -> list[dict]:
        """Копия памяти: интерфейс её показывает, но менять не может."""
        return [dict(m) for m in self._memory]

    @property
    def history_size(self) -> int:
        """Сколько сообщений активной ветки лежит в истории (в памяти — обычно меньше)."""
        return self.store.count(self.id, self.branch) if self.store is not None else len(self._memory)

    def transcript(self) -> list[dict]:
        """Вся переписка активной ветки: из истории, если она есть, иначе из памяти.

        Интерфейс рисует ленту именно отсюда, поэтому после перезапуска на экране
        оказывается весь прошлый разговор, а не только то, что уйдёт в модель.
        """
        if self.store is not None:
            return self.store.messages(self.id, self.branch)
        return self.memory

    def tokens_state(self) -> dict:
        """Во что обойдётся следующий запрос и сколько уже потрачено за всё время.

        Считается до всякого вызова: инструкция, блоки стратегии, окно памяти и
        схемы инструментов уже известны, а значит известен и вес контекста.
        Интерфейс показывает это полосой — видно, как разговор занимает окно
        модели и из чего он состоит. Отдельно считается ВСЯ переписка на диске:
        обычно она заметно больше контекста, и разница между «сохранено» и
        «уходит в модель» — это и есть цена памяти.
        """
        specs = tools.specs() if self.tools_enabled else None
        summary_block = self._summary_block()
        facts_block = self._facts_block()
        breakdown = tokens.measure(
            self._system_prompt([]), self._memory, "", specs, self.model,
            summary=summary_block, facts=facts_block,
        )
        limit = self._context_limit()
        history = self.transcript()
        fix = tokens.calibration(self.model)
        folded = self._summary.tokens if summary_block else 0
        beyond, beyond_tokens = self._beyond_window(history)
        return {
            "model": self.model,
            "breakdown": breakdown.to_dict(),
            "parts": breakdown.parts(),
            "context_tokens": breakdown.total,
            "limit": limit,
            "reserve": self._answer_reserve(),
            "max_output": config.model_max_output(self.model),
            "fill": breakdown.total / limit if limit else 0.0,
            "window_messages": len(self._memory),
            "history_messages": len(history),
            "history_tokens": tokens.measure_history(history, self.model),
            "calibration": {
                "factor": round(fix.factor, 3),
                "samples": fix.samples,
                "error_pct": fix.error_pct,
                "last_estimated": fix.last_estimated,
                "last_actual": fix.last_actual,
            },
            "spent": self.spent(),
            # Стратегия и сжатие: что заменяют их блоки, что ждёт очередей, что осталось
            # за окном и сколько весил бы тот же запрос с полной историей как есть.
            "strategy": self.strategy,
            "strategy_label": config.strategy_label(self.strategy),
            "strategy_en": config.strategy_en(self.strategy),
            "summarize": self.summarize,
            "branch": self.branch,
            "branch_name": self._branch["name"],
            "branch_origin": self._branch["origin"],
            "branch_shared": self._branch["shared"],
            "summary_every": self.summary_every,
            "summary": self._summary.to_dict(),
            "summary_active": bool(summary_block),
            "summary_tokens": breakdown.summary,
            "folded_messages": self._summary.messages if summary_block else 0,
            "folded_tokens": folded,
            "facts": self._facts.to_dict(),
            "facts_active": bool(facts_block),
            "facts_tokens": breakdown.facts,
            "facts_pending": len(self._pending_facts),
            "pending_messages": len(self._pending_summary),
            "pending_tokens": tokens.measure_messages(self._pending_summary, self.model),
            "dropped_messages": beyond,
            "dropped_tokens": beyond_tokens,
            "uncompressed": breakdown.total - breakdown.summary - breakdown.facts + folded + beyond_tokens,
        }

    def summaries(self) -> list[dict]:
        """Все версии суммаризации активной ветки: как она росла вместе с разговором."""
        return self.store.summaries(self.id, self.branch) if self.store is not None else []

    def facts_versions(self) -> list[dict]:
        """Все версии блока фактов активной ветки."""
        return self.store.facts_versions(self.id, self.branch) if self.store is not None else []

    def spent(self) -> dict:
        """Итог по расходу токенов за всё время жизни агента (из хранилища)."""
        if self.store is None:
            return {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "cost_usd": 0.0, "llm_calls": 0}
        return self.store.usage_totals(self.id)

    def usage_log(self) -> list[dict]:
        """Расход по обращениям, по строке на обращение: из этого растёт диаграмма."""
        return self.store.usage(self.id) if self.store is not None else []

    def passport(self) -> dict:
        """Всё состояние агента одним словарём — то, что рисует интерфейс."""
        return {
            "id": self.id,
            "name": self.profile.name,
            "role": self.profile.role,
            "instructions": self.profile.instructions,
            "model": self.model,
            "model_label": config.model_label(self.model),
            "tier": config.model_tier(self.model),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "memory_turns": self.memory_turns,
            "memory_messages": len(self._memory),
            "memory": self.memory,
            "turns": self.turns,
            "created_at": self.created_at,
            "tools_enabled": self.tools_enabled,
            "planning": self.planning,
            "max_steps": self.max_steps,
            "strategy": self.strategy,
            "strategy_label": config.strategy_label(self.strategy),
            "strategy_en": config.strategy_en(self.strategy),
            "summarize": self.summarize,
            "summary_every": self.summary_every,
            "summary": self._summary.to_dict(),
            "summary_active": bool(self._summary_block()),
            "facts": self._facts.to_dict(),
            "facts_active": bool(self._facts_block()),
            "facts_pending": len(self._pending_facts),
            "pending_messages": len(self._pending_summary),
            "branch": self.branch,
            "branch_name": self._branch["name"],
            "branch_origin": self._branch["origin"],
            "branch_shared": self._branch["shared"],
            "tools": tools.catalog(),
            "workspace": str(tools.workspace()),
            # Память между запусками: сколько сохранено, где лежит и когда говорили
            "history_messages": self.history_size,
            "history_file": str(self.store.path) if self.store is not None else None,
            "last_seen_at": self.store.last_at(self.id, self.branch) if self.store is not None else None,
            "restored": self.restored,
        }


def load_agents(store: Store | None = None) -> tuple[list[Agent], Agent]:
    """Поднять агентов прошлого запуска и того из них, с кем шёл разговор.

    Это единственное место, где решается, откуда берутся агенты при старте, —
    интерфейсу остаётся показать готовое. Истории нет (первый запуск, стёрли файл)
    — заводим одного агента по умолчанию и сразу закрепляем его в хранилище.
    """
    store = store if store is not None else Store()
    agents = [Agent.restore(state, store) for state in store.agents()]
    if not agents:
        fresh = Agent(store=store)
        fresh.persist()
        fresh.mark_active()
        agents = [fresh]
        logger.info("История пуста — создан агент «%s» [%s]", fresh.profile.name, fresh.id)

    active_id = store.active_id()
    active = next((a for a in agents if a.id == active_id), agents[0])
    return agents, active


def _strategy_and_summarize(settings: dict) -> tuple[str, bool]:
    """Стратегия и сжатие из сохранённых настроек, с оглядкой на прошлые версии базы.

    Настройка `summarize` появилась, когда суммаризация перестала быть стратегией и
    стала опцией поверх любой из них. Пока её в базе нет (None — колонку только что
    дописали), значение выводим из того, что там лежало: в базе прошлой версии
    суммаризация была одним из положений переключателя стратегий, а ещё раньше —
    отдельным тумблером `compression`. Поведение агента от этого не меняется: он
    продолжает делать то же, что делал до обновления.
    """
    strategy = settings.get("strategy") or ""
    summarize = settings.get("summarize")
    if summarize is None:
        summarize = (strategy == "summary") if strategy else settings.get("compression", True)
    if strategy in ("summary", ""):
        strategy = "window"   # суммаризация поверх скользящего окна — это она и была
    if strategy not in config.STRATEGY_BY_CODE:
        logger.warning("В истории стратегия «%s» неизвестна — берём %s", strategy, config.AGENT_STRATEGY)
        strategy = config.AGENT_STRATEGY
    return strategy, bool(summarize)


def _extract_json(text: str) -> dict | None:
    """Достать JSON из ответа модели (на случай ```-обрамления или текста вокруг)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _parse_facts(text: str) -> dict[str, str] | None:
    """Разобрать блок фактов из ответа модели: {"facts": {...}} или просто объект.

    Значения приводятся к коротким строкам, пустые ключи и значения отбрасываются,
    лишнее сверх потолка отрезается. None — если это вообще не JSON-объект.
    """
    data = _extract_json(text)
    if isinstance(data, dict) and isinstance(data.get("facts"), dict):
        data = data["facts"]
    if not isinstance(data, dict):
        return None
    items: dict[str, str] = {}
    for key, value in data.items():
        name = " ".join(str(key).split()).strip(" :-")[:60]
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        text_value = " ".join(str(value).split())[:200] if value is not None else ""
        if name and text_value:
            items[name] = text_value
    return dict(list(items.items())[:config.FACTS_LIMIT])


def _clean_name(name: str) -> str:
    """Имя ветки или точки: одна строка без лишних пробелов, не длиннее NAME_MAX."""
    return " ".join((name or "").split())[:NAME_MAX]


def _short(value: object, limit: int = 60) -> str:
    """Короткое представление аргумента для лога."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _slim(message: dict) -> dict:
    """Сообщение истории в виде для памяти: роль, текст и номер строки, если есть."""
    slim = {"role": message["role"], "content": message["content"]}
    if message.get("id"):
        slim["id"] = message["id"]
    return slim
