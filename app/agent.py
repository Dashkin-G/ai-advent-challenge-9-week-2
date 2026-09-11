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
       вопрос и итоговый ответ, без служебной переписки с инструментами)

Именно шаг 4 отличает агента от чата: одна фраза пользователя разворачивается в
последовательность действий, которые агент выбирает и выполняет сам.

Токены агент считает сам и до отправки (`app/tokens.py`): контекст — ресурс с
жёстким потолком, и знать его цену постфактум поздно. Оценка сверяется с
фактическим `usage` из ответа, расхождение уходит в калибровку, а расход каждого
обращения ложится в хранилище — по нему видно, как дорожает разговор.

Память двухуровневая. В `_memory` живёт окно контекста — последние `memory_turns`
пар «вопрос-ответ», ровно то, что уходит в модель. Полная переписка вместе с
паспортом и настройками пишется в хранилище (`store.py`), поэтому агент не
начинает с нуля после перезапуска приложения: `load_agents()` поднимает тех же
агентов с их историей, и разговор продолжается так, будто его не прерывали.

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
from .store import Store

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
class TokenReport:
    """Счёт за обращение в токенах: оценка до вызова и факт после.

    Оценку агент считает сам (`app/tokens.py`) ещё до того, как запрос ушёл в
    сеть, — иначе о цене и о переполнении контекста узнаёшь только по факту, то
    есть когда платить уже поздно. Факт приходит в `usage`, и разница между
    оценкой и фактом здесь же: по ней видно, можно ли счётчику верить.
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
            "request": self.request,
            "response": self.response,
        }


@dataclass
class Agent:
    """Экземпляр агента: паспорт + настройки + память + инструменты + счётчики.

    Память диалога хранится здесь, в самом агенте: интерфейс присылает только
    очередное сообщение, а контекст для модели агент собирает сам. Агентов может
    быть несколько — они независимы, память одного не видна другому.

    Если агенту дали хранилище (`store`), он сам записывает туда каждое обращение
    и каждую смену настроек — и переживает перезапуск приложения. Без хранилища
    агент полностью работоспособен, просто помнит разговор только до закрытия.
    """
    profile: AgentProfile = DEFAULT_PROFILE
    model: str = config.DEFAULT_MODEL
    temperature: float = config.AGENT_TEMPERATURE
    max_tokens: int | None = config.AGENT_MAX_TOKENS
    memory_turns: int = config.AGENT_MEMORY_TURNS
    tools_enabled: bool = config.AGENT_TOOLS      # давать ли модели инструменты
    planning: bool = config.AGENT_PLANNING        # писать ли план перед работой
    max_steps: int = config.AGENT_MAX_STEPS       # потолок шагов цикла за обращение

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)
    turns: int = 0                                # сколько запросов агент обработал
    store: Store | None = field(default=None, repr=False)   # куда класть состояние
    restored: bool = False                        # поднят из истории, а не создан заново
    _memory: list[dict] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------- вход --

    def ask(self, message: str) -> AgentReply:
        """Единственный публичный вход: принять запрос и вернуть ответ агента."""
        text = self._prepare(message)                       # 1. проверка входа
        started = time.perf_counter()
        totals = _Totals()
        steps: list[AgentStep] = []

        logger.info(
            "Агент «%s» [%s] ← запрос #%d (%d симв.) · инструменты=%s · план=%s",
            self.profile.name, self.id, self.turns + 1, len(text), self.tools_enabled, self.planning,
        )

        plan = self._make_plan(text, totals)                # 2. план действий
        specs = tools.specs() if self.tools_enabled else None
        prompt = self._system_prompt(plan)
        window, dropped = self._fit_context(prompt, text, specs)   # 3. бюджет контекста
        report = TokenReport(
            breakdown=tokens.measure(prompt, window, text, specs, self.model),
            limit=self._context_limit(),
            reserve=self._answer_reserve(),
            max_output=config.model_max_output(self.model),
            trimmed_pairs=dropped,
        )
        report.estimated = report.breakdown.total
        messages = self._build_messages(prompt, window, text)      #    сборка запроса
        logger.info(
            "Агент «%s» [%s]: в запрос уйдёт ≈%d токенов (инструкция %d + память %d + вопрос %d "
            "+ схемы %d) из окна %d",
            self.profile.name, self.id, report.estimated, report.breakdown.system,
            report.breakdown.memory, report.breakdown.question, report.breakdown.tools, report.limit,
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
        self._close_report(report, totals, raw, first_usage)
        self.turns += 1
        self._remember(text, answer, report, totals)        # 6. память, история, расход

        elapsed = time.perf_counter() - started
        logger.info(
            "Агент «%s» [%s] → ответ #%d за %.2f c · шагов=%d · вызовов модели=%d · в памяти %d сообщ. · "
            "токены: оценка %d → факт %d (%s%%), ответ %d, всего за обращение %d",
            self.profile.name, self.id, self.turns, elapsed, len(steps), totals.llm_calls,
            len(self._memory), report.estimated, report.prompt_tokens, report.error_pct,
            report.completion_tokens, report.total_prompt + report.total_completion,
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
        это память, уже подрезанная под окно контекста (см. `_fit_context`).
        """
        return (
            [{"role": "system", "content": prompt}]
            + list(window)
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

    def _system_prompt(self, plan: list[str]) -> str:
        """Роль агента, дополненная правилами про инструменты, память и план.

        Роль пишет пользователь, и полагаться на неё в этих вопросах нельзя: про
        инструменты и про собственную память агент рассказывает модели сам.
        """
        prompt = self.profile.instructions
        if self.tools_enabled:
            prompt += TOOLS_NOTE
            prompt += f"\n\nРабочая папка для файловых инструментов: {tools.workspace()}"
        if self._memory:
            prompt += MEMORY_NOTE.format(count=len(self._memory), when=self._last_seen())
        else:
            prompt += NO_MEMORY_NOTE
        if plan:
            listed = "\n".join(f"{i}. {step}" for i, step in enumerate(plan, 1))
            prompt += (
                "\n\nТы сам составил план на эту задачу:\n" + listed +
                "\nСледуй ему. Если по ходу дела план оказался неверным — скажи об этом в ответе."
            )
        return prompt

    def _call(self, messages: list[dict], specs: list[dict] | None) -> dict:
        """Один вызов модели через транспорт; сбой превращаем в ошибку агента."""
        try:
            return llm.chat(
                messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
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

    def _remember(
        self,
        question: str,
        answer: str,
        report: TokenReport | None = None,
        totals: _Totals | None = None,
    ) -> None:
        """Шаг 6. Запомнить обмен: в окно контекста, в историю и в расход токенов.

        В памяти держим только вопрос и итоговый ответ: служебная переписка с
        инструментами нужна внутри одного обращения, а в долгой памяти она бы
        быстро съела контекст и деньги.

        В хранилище уходит та же пара, обновлённое состояние агента и строка
        расхода — одной записью, чтобы история, счётчик обращений и токены не
        разъезжались. Строка расхода и превращает «сколько стоило» в наблюдаемую
        величину: по ней видно, как цена растёт от обращения к обращению.
        """
        pair = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        self._memory.extend(pair)
        self._trim_memory()
        if self.store is not None:
            self.store.save_turn(self.state(), pair, self._usage_row(report, totals))

    def _usage_row(self, report: TokenReport | None, totals: _Totals | None) -> dict | None:
        """Расход обращения одной строкой — то, из чего потом рисуется рост цены."""
        if report is None or totals is None:
            return None
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
        }

    def _trim_memory(self) -> None:
        """Оставить в памяти только последние `memory_turns` пар «вопрос-ответ»."""
        limit = max(0, self.memory_turns) * 2
        if len(self._memory) > limit:
            del self._memory[: len(self._memory) - limit]

    def _load_memory(self) -> None:
        """Собрать окно контекста из истории: последние `memory_turns` пар.

        Вызывается при восстановлении агента и при смене глубины памяти —
        увеличили глубину, и агент дотягивает из истории то, что уже забыл.
        """
        if self.store is None:
            self._trim_memory()
            return
        history = self.store.messages(self.id, limit=max(0, self.memory_turns) * 2)
        self._memory = [{"role": m["role"], "content": m["content"]} for m in history]

    def _last_seen(self) -> str:
        """Когда в памяти появилось последнее сообщение — для заметки модели."""
        moment = self.store.last_at(self.id) if self.store is not None else None
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
    ) -> None:
        """Изменить настройки агента.

        Проверки живут здесь, а не в интерфейсе: настройки — часть самого агента,
        и любой интерфейс поверх него получает их бесплатно.
        """
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
            "план=%s, шагов=%d, лимит ответа=%s",
            self.profile.name, self.id, self.model, self.temperature, self.memory_turns,
            self.tools_enabled, self.planning, self.max_steps, self.max_tokens or "без ограничения",
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
        """Забыть диалог — и в памяти, и в истории на диске.

        Паспорт и настройки остаются: агент тот же самый, просто без прошлого.
        Стирать историю здесь важно, иначе после перезапуска забытое вернулось бы.
        """
        self._memory.clear()
        self.turns = 0
        if self.store is not None:
            self.store.forget(self.id)
        self.persist()
        logger.info("Агент «%s» [%s]: память и история очищены", self.profile.name, self.id)

    # ------------------------------------------------- состояние между запусками --

    def state(self) -> dict:
        """Всё, чем агент является, кроме переписки: её ведёт сама история.

        Это то, что уходит в хранилище. Обратная операция — `restore()`.
        """
        return {
            "id": self.id,
            "created_at": self.created_at,
            "turns": self.turns,
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
            id=state.get("id") or uuid.uuid4().hex[:8],
            created_at=float(state.get("created_at") or time.time()),
            turns=int(state.get("turns") or 0),
            store=store,
            restored=True,
        )
        agent._load_memory()
        logger.info(
            "Агент «%s» [%s] восстановлен: %d обращений, %d сообщ. в памяти из %d в истории",
            agent.profile.name, agent.id, agent.turns, len(agent._memory), agent.history_size,
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
        """Сколько сообщений агента лежит в истории (в памяти — обычно меньше)."""
        return self.store.count(self.id) if self.store is not None else len(self._memory)

    def transcript(self) -> list[dict]:
        """Вся переписка агента: из истории, если она есть, иначе из памяти.

        Интерфейс рисует ленту именно отсюда, поэтому после перезапуска на экране
        оказывается весь прошлый разговор, а не только то, что уйдёт в модель.
        """
        if self.store is not None:
            return self.store.messages(self.id)
        return self.memory

    def tokens_state(self) -> dict:
        """Во что обойдётся следующий запрос и сколько уже потрачено за всё время.

        Считается до всякого вызова: инструкция, окно памяти и схемы инструментов
        уже известны, а значит известен и вес контекста. Интерфейс показывает это
        полосой — видно, как разговор занимает окно модели и из чего он состоит.
        Отдельно считается ВСЯ переписка на диске: обычно она заметно больше
        контекста, и разница между «сохранено» и «уходит в модель» — это и есть
        цена памяти.
        """
        specs = tools.specs() if self.tools_enabled else None
        breakdown = tokens.measure(self._system_prompt([]), self._memory, "", specs, self.model)
        limit = self._context_limit()
        history = self.transcript()
        fix = tokens.calibration(self.model)
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
        }

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
            "tools": tools.catalog(),
            "workspace": str(tools.workspace()),
            # Память между запусками: сколько сохранено, где лежит и когда говорили
            "history_messages": self.history_size,
            "history_file": str(self.store.path) if self.store is not None else None,
            "last_seen_at": self.store.last_at(self.id) if self.store is not None else None,
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


def _short(value: object, limit: int = 60) -> str:
    """Короткое представление аргумента для лога."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "…"
