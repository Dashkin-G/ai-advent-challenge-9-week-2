"""Интерфейс агента — десктопное окно: python -m app.gui

Окно ничего не знает про устройство агента: оно создаёт `Agent`, отдаёт ему строку
и показывает `reply.text` вместе с трассой шагов. Ни сборки запроса, ни памяти, ни
исполнения инструментов здесь нет — всё это внутри агента (app/agent.py), и в нём
нет ни одного импорта Qt. Проверить границу просто: любой новый интерфейс поверх
`Agent` не должен требовать правок в самом агенте.

Интерфейс на PySide6 (Qt): Qt рисует виджеты сам и стилизуется таблицей QSS —
поэтому тёмная тема получается полностью управляемой, вплоть до полос прокрутки.

Важная особенность: вызов модели блокирующий, а перерисовывает окно главный поток.
Поэтому обращение к агенту уходит в отдельный QThread, а результат возвращается
сигналом.
"""
import json
import logging
import re
import sys

from PySide6.QtCore import QEvent, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from . import config
from .agent import DEFAULT_PROFILE, Agent, AgentError, AgentProfile, AgentReply

# В окне нужен диалог, а не логи вызовов — оставляем только предупреждения.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")

# --- Палитра ---------------------------------------------------------------
BLACK = "#08090b"     # фон ленты
PANEL = "#0d0f13"     # боковая панель, шапка, полоса ввода
CARD = "#13161c"      # карточки и пузыри агента
LINE = "#1e232c"      # границы
TEXT = "#e8eaed"
MUTED = "#79818f"
ACCENT = "#4f8cff"
ACCENT2 = "#2b6bff"
ERR_BG = "#241417"
ERR_LINE = "#4a1f26"
ERR_TEXT = "#ff9d9d"
OK = "#2fbf6b"
WARN = "#e6b800"

BUBBLE_ID = {"user": "bubbleUser", "agent": "bubbleAgent", "error": "bubbleError"}

QSS = f"""
QWidget {{
    background: {BLACK};
    color: {TEXT};
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
}}
QFrame#sidebar {{ background: {PANEL}; border-right: 1px solid {LINE}; }}
QScrollArea#sidebarScroll, QWidget#sidebarInner {{ background: {PANEL}; }}
QWidget#sidebarInner QLabel {{ background: transparent; }}
QFrame#card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; }}
QFrame#header {{ background: {PANEL}; border-bottom: 1px solid {LINE}; }}
QFrame#composer {{ background: {PANEL}; border-top: 1px solid {LINE}; }}
QFrame#card QLabel, QFrame#header QLabel, QFrame#composer QLabel,
QFrame#sidebar QLabel {{ background: transparent; }}

QLabel#agentName {{ font-size: 17px; font-weight: 700; }}
QLabel#agentRole {{ color: {MUTED}; font-size: 12px; }}
QLabel#agentId {{ color: {MUTED}; font-family: Consolas, monospace; font-size: 11px; }}
QLabel#avatar {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ACCENT2}, stop:1 #7a5cff);
    border-radius: 22px; font-size: 20px;
}}
QLabel#section {{ color: {MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px; }}
QLabel#note {{ color: {MUTED}; font-size: 11px; }}
QLabel#meta {{ color: {MUTED}; font-family: Consolas, monospace; font-size: 11px; }}
QLabel#system {{ color: {MUTED}; font-size: 12px; }}
QLabel#statValue {{ font-size: 20px; font-weight: 700; }}
QLabel#statLabel {{ color: {MUTED}; font-size: 11px; }}
QLabel#title {{ font-size: 15px; font-weight: 600; }}
QLabel#subtitle {{ color: {MUTED}; font-size: 12px; }}
QLabel#status {{ color: {MUTED}; font-size: 12px; }}

QFrame#bubbleUser {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ACCENT2}, stop:1 {ACCENT});
    border-radius: 16px;
}}
QFrame#bubbleUser QLabel {{ background: transparent; color: #ffffff; }}
QFrame#bubbleAgent {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; }}
QFrame#bubbleAgent QLabel {{ background: transparent; }}
QFrame#bubbleError {{ background: {ERR_BG}; border: 1px solid {ERR_LINE}; border-radius: 16px; }}
QFrame#bubbleError QLabel {{ background: transparent; color: {ERR_TEXT}; }}
QFrame#stat {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; }}
QFrame#stat QLabel {{ background: transparent; }}

QPushButton#send {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ACCENT2}, stop:1 {ACCENT});
    color: #ffffff; border: 0; border-radius: 12px; padding: 0 26px;
    font-size: 14px; font-weight: 700;
}}
QPushButton#send:hover {{ background: {ACCENT}; }}
QPushButton#send:disabled {{ background: #1b2330; color: {MUTED}; }}
QPushButton#ghost {{
    background: transparent; color: {TEXT}; border: 1px solid {LINE};
    border-radius: 10px; padding: 9px 12px; font-size: 13px;
}}
QPushButton#ghost:hover {{ background: {CARD}; border-color: #2c3441; }}
QPushButton#link {{
    background: transparent; border: 0; color: {ACCENT};
    font-size: 11px; font-family: Consolas, monospace; padding: 0;
}}
QPushButton#link:hover {{ color: #82adff; }}

QCheckBox {{ font-size: 13px; spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 5px;
    border: 1px solid #2c3441; background: {CARD};
}}
QCheckBox::indicator:checked {{ background: {ACCENT2}; border-color: {ACCENT2}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}

QFrame#trace {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 14px; }}
QFrame#trace QLabel {{ background: transparent; }}
QLabel#traceLabel {{ color: {MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px; }}
QLabel#planItem {{ font-size: 13px; }}
QLabel#stepHead {{ color: {ACCENT}; font-family: Consolas, monospace; font-size: 12px; }}
QLabel#stepHeadErr {{ color: {ERR_TEXT}; font-family: Consolas, monospace; font-size: 12px; }}
QFrame#stepResult {{ background: {BLACK}; border: 1px solid {LINE}; border-radius: 8px; }}
QFrame#stepResult QLabel {{
    background: transparent; color: #a9c39a;
    font-family: Consolas, monospace; font-size: 11px;
}}

QComboBox, QSpinBox {{
    background: {CARD}; border: 1px solid {LINE}; border-radius: 10px;
    padding: 8px 10px; font-size: 13px; selection-background-color: {ACCENT2};
}}
QComboBox:hover, QSpinBox:hover {{ border-color: #2c3441; }}
QComboBox::drop-down {{ border: 0; width: 22px; }}
/* Стрелки спинбокса система рисует светлыми — прячем, поле правится с клавиатуры. */
QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: 0; }}
QComboBox QAbstractItemView {{
    background: {CARD}; border: 1px solid {LINE};
    selection-background-color: {ACCENT2}; outline: 0; padding: 4px;
}}
QTextEdit#input {{
    background: {CARD}; border: 1px solid {LINE}; border-radius: 12px;
    padding: 10px 12px; font-size: 14px; selection-background-color: {ACCENT2};
}}
QTextEdit#input:focus {{ border-color: {ACCENT2}; }}
QPlainTextEdit#raw {{
    background: {BLACK}; border: 1px solid {LINE}; border-radius: 10px;
    font-family: Consolas, monospace; font-size: 12px; color: #a9c39a;
}}
QTabWidget::pane {{ border: 0; }}
QTabBar::tab {{
    background: transparent; color: {MUTED}; padding: 7px 14px;
    border-bottom: 2px solid transparent; font-size: 12px;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom-color: {ACCENT2}; }}

QSlider::groove:horizontal {{ height: 4px; background: {LINE}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT2}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; background: {ACCENT};
}}
QSlider::handle:horizontal:hover {{ background: #7fb0ff; }}

QScrollArea {{ border: 0; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px; }}
QScrollBar::handle:vertical {{ background: #232a34; border-radius: 5px; min-height: 40px; }}
QScrollBar::handle:vertical:hover {{ background: #333c4a; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px 4px; }}
QScrollBar::handle:horizontal {{ background: #232a34; border-radius: 5px; min-width: 40px; }}
QScrollBar::handle:horizontal:hover {{ background: #333c4a; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}

QPushButton#newAgent {{
    background: {CARD}; color: {TEXT}; border: 1px solid {ACCENT2};
    border-radius: 10px; padding: 7px 14px; font-size: 13px; font-weight: 600;
}}
QPushButton#newAgent:hover {{ background: {ACCENT2}; }}

/* Полоса агентов над лентой: список растёт вбок, а не выдавливает панель вниз */
QWidget#agentBarRow {{ background: {PANEL}; border-bottom: 1px solid {LINE}; }}
QScrollArea#agentBar, QWidget#agentBarInner {{ background: {PANEL}; }}
QFrame#agentItem {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 10px; }}
QFrame#agentItem:hover {{ border-color: {ACCENT}; }}
QFrame#agentItemActive {{
    background: #182034; border: 1px solid {ACCENT2}; border-radius: 10px;
}}
QFrame#agentItem QLabel, QFrame#agentItemActive QLabel {{ background: transparent; }}
QLabel#agentItemName {{ font-size: 13px; font-weight: 600; }}
QLabel#agentItemSub {{ color: {MUTED}; font-size: 11px; }}
QPushButton#example {{
    background: {CARD}; color: {TEXT}; border: 1px solid {LINE};
    border-radius: 14px; padding: 6px 12px; font-size: 12px;
}}
QPushButton#example:hover {{ border-color: {ACCENT2}; background: #182034; }}
QLineEdit {{
    background: {CARD}; border: 1px solid {LINE}; border-radius: 10px;
    padding: 8px 10px; font-size: 13px; selection-background-color: {ACCENT2};
}}
QLineEdit:focus {{ border-color: {ACCENT2}; }}
"""

# Масштаб интерфейса: все размеры в QSS заданы в пикселях, поэтому Ctrl+колесо
# просто пересобирает таблицу стилей, умножая числа перед «px». Отдельной копии
# стилей под каждый масштаб не нужно.
MIN_SCALE, MAX_SCALE = 0.8, 2.0


def build_qss(scale: float = 1.0) -> str:
    if abs(scale - 1.0) < 0.01:
        return QSS
    return re.sub(r"(\d+)px", lambda m: f"{max(1, round(int(m.group(1)) * scale))}px", QSS)


class AskWorker(QThread):
    """Обращение к агенту в отдельном потоке: окно не должно замирать на вызове."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, agent: Agent, message: str) -> None:
        super().__init__()
        self.agent = agent
        self.message = message

    def run(self) -> None:
        try:
            self.done.emit(self.agent.ask(self.message))
        except AgentError as e:
            self.failed.emit(str(e))


class Composer(QTextEdit):
    """Поле ввода: Enter отправляет, Shift+Enter переносит строку."""

    submitted = Signal()

    def keyPressEvent(self, event) -> None:
        enter = event.key() in (Qt.Key_Return, Qt.Key_Enter)
        if enter and not event.modifiers() & Qt.ShiftModifier:
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class Bubble(QFrame):
    """Пузырь сообщения. Роль задаёт оформление: свой, агента или ошибка."""

    def __init__(self, text: str, role: str, max_width: int = 660) -> None:
        super().__init__()
        self.max_width = max_width
        self.setObjectName(BUBBLE_ID[role])
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.setMaximumWidth(max_width)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 11, 16, 12)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.label)
        self._fit(text)

    def _fit(self, text: str) -> None:
        """Подобрать ширину под текст.

        QLabel с переносом отдаёт нарочито узкий sizeHint, и пузырь схлопывается
        в колонку в пару слов. Поэтому считаем ширину самой длинной строки сами и
        ставим её минимальной, а перенос включается уже на потолке MAX_WIDTH.
        """
        fm = self.label.fontMetrics()
        longest = max((fm.horizontalAdvance(line) for line in text.split("\n")), default=0)
        ideal = int(longest * 1.1) + 44  # запас на отступы и разницу шрифта QSS
        self.setMinimumWidth(max(90, min(self.max_width, ideal)))

    def set_text(self, text: str) -> None:
        self.label.setText(text)
        self._fit(text)

    def set_role(self, role: str) -> None:
        self.setObjectName(BUBBLE_ID[role])
        self.style().unpolish(self)
        self.style().polish(self)


class ChatView(QScrollArea):
    """Лента диалога: пузыри и мета-строки, всегда прокрученная к последнему."""

    def __init__(self, bubble_max: int = 660) -> None:
        super().__init__()
        self.bubble_limit = bubble_max   # желаемый потолок (растёт с масштабом)
        self.bubble_max = bubble_max     # фактический, с учётом ширины ленты
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        self.lay = QVBoxLayout(body)
        self.lay.setContentsMargins(26, 22, 18, 22)
        self.lay.setSpacing(10)
        self.lay.addStretch(1)
        self.setWidget(body)

    def add(self, widget: QWidget, align=Qt.AlignLeft) -> QWidget:
        """Добавить виджет строкой ленты, прижав его к левому или правому краю."""
        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        if align == Qt.AlignRight:
            row_lay.addStretch(1)
            row_lay.addWidget(widget)
        else:
            row_lay.addWidget(widget)
            row_lay.addStretch(1)
        return self.add_row(row, widget)

    def add_row(self, row: QWidget, result: QWidget | None = None) -> QWidget:
        """Добавить готовую строку во всю ширину ленты (без прижатия к краю)."""
        self.lay.insertWidget(self.lay.count() - 1, row)  # перед распоркой в конце
        QTimer.singleShot(0, self._to_bottom)
        return result if result is not None else row

    def add_bubble(self, text: str, role: str) -> Bubble:
        bubble = Bubble(text, role, self.bubble_max)
        return self.add(bubble, Qt.AlignRight if role == "user" else Qt.AlignLeft)

    def to_top(self) -> None:
        """Показать начало ленты (в окне памяти это список, а не живой диалог)."""
        QTimer.singleShot(0, lambda: self.verticalScrollBar().setValue(0))

    def add_system(self, text: str) -> None:
        label = QLabel(text)
        label.setObjectName("system")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 6, 0, 6)
        row_lay.addWidget(label)   # во всю ширину: текст центрируется внутри метки
        self.add_row(row)

    def clear(self) -> None:
        while self.lay.count() > 1:
            item = self.lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.refit()

    def refit(self) -> None:
        """Подогнать ширину пузырей под ленту.

        Потолок растёт вместе с масштабом интерфейса, но пузырь не должен быть
        шире самой ленты — иначе на большом масштабе текст уезжает за край.
        """
        available = max(240, self.viewport().width() - 90)
        limit = min(self.bubble_limit, available)
        if limit == self.bubble_max:
            return
        self.bubble_max = limit
        for bubble in self.findChildren(Bubble):
            bubble.max_width = limit
            bubble.setMaximumWidth(limit)
            bubble._fit(bubble.label.text())

    def _to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())


class RawDialog(QDialog):
    """Сырой обмен: тело запроса, собранное агентом, и ответ модели как есть."""

    def __init__(self, reply: AgentReply, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Сырой обмен · обращение #{reply.turn}")
        self.resize(780, 560)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        note = QLabel("Окно отправило одну строку — вот что из неё собрал агент.")
        note.setObjectName("note")
        lay.addWidget(note)
        tabs = QTabWidget()
        for title, data in (("→ запрос", reply.request), ("← ответ модели", reply.response)):
            view = QPlainTextEdit(json.dumps(data, ensure_ascii=False, indent=2))
            view.setObjectName("raw")
            view.setReadOnly(True)
            view.setLineWrapMode(QPlainTextEdit.NoWrap)
            tabs.addTab(view, title)
        lay.addWidget(tabs)


class MemoryDialog(QDialog):
    """Память агента: то, что он подставит в следующий запрос."""

    def __init__(self, agent: Agent, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Память агента «{agent.profile.name}»")
        self.resize(680, 560)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)

        memory = agent.memory
        head = QLabel(
            f"{len(memory)} сообщ. — агент подставит их в следующий запрос"
            if memory else "Память пуста — агент ещё ничего не запомнил."
        )
        head.setObjectName("note")
        lay.addWidget(head)

        chat = ChatView(bubble_max=480)
        for m in memory:
            chat.add_bubble(m["content"], "user" if m["role"] == "user" else "agent")
        chat.to_top()
        lay.addWidget(chat)


class AgentItem(QFrame):
    """Карточка агента в списке: клик выбирает, крестик удаляет."""

    chosen = Signal()
    removed = Signal()

    def __init__(self, agent: Agent, active: bool, deletable: bool) -> None:
        super().__init__()
        self.setObjectName("agentItemActive" if active else "agentItem")
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setToolTip(
            f"{agent.profile.name} — {agent.profile.role}\n"
            f"{config.model_label(agent.model)} · обращений: {agent.turns} · "
            f"в памяти: {len(agent.memory)} сообщ."
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 8, 7)
        lay.setSpacing(7)

        name = QLabel(agent.profile.name)
        name.setObjectName("agentItemName")
        lay.addWidget(name)

        counter = QLabel(f"{agent.turns} обр. · {len(agent.memory)} в памяти")
        counter.setObjectName("agentItemSub")
        lay.addWidget(counter)

        if deletable:
            remove = QPushButton("✕")
            remove.setObjectName("link")
            remove.setCursor(Qt.PointingHandCursor)
            remove.setToolTip("Удалить агента вместе с его памятью")
            remove.clicked.connect(self.removed.emit)
            lay.addWidget(remove, 0, Qt.AlignTop)

    def mousePressEvent(self, event) -> None:
        self.chosen.emit()
        super().mousePressEvent(event)


class NewAgentDialog(QDialog):
    """Новый агент: имя, заготовка профиля, инструкция и модель."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Новый агент")
        self.resize(560, 460)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        note = QLabel("У каждого агента своя роль, своя память и свои настройки.")
        note.setObjectName("note")
        lay.addWidget(note)

        lay.addWidget(_section("ЗАГОТОВКА ПРОФИЛЯ"))
        self.preset = QComboBox()
        for profile in config.PROFILE_PRESETS:
            self.preset.addItem(f"{profile['name']} — {profile['role']}", profile)
        self.preset.currentIndexChanged.connect(self._fill_from_preset)
        lay.addWidget(self.preset)

        lay.addWidget(_section("ИМЯ"))
        self.name = QLineEdit()
        self.name.setMaxLength(40)
        lay.addWidget(self.name)

        lay.addWidget(_section("ИНСТРУКЦИЯ (РОЛЬ)"))
        self.instructions = QPlainTextEdit()
        self.instructions.setPlaceholderText("Пусто — возьмётся инструкция агента по умолчанию")
        lay.addWidget(self.instructions, 1)

        lay.addWidget(_section("МОДЕЛЬ"))
        self.model = QComboBox()
        for item in config.MODELS:
            self.model.addItem(item["label"], item["code"])
        self.model.setCurrentIndex(max(0, self.model.findData(config.DEFAULT_MODEL)))
        lay.addWidget(self.model)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        create = QPushButton("Создать")
        create.setObjectName("send")
        create.setFixedHeight(40)
        create.setCursor(Qt.PointingHandCursor)
        create.clicked.connect(self.accept)
        cancel = _ghost("Отмена")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        buttons.addWidget(create, 1)
        lay.addLayout(buttons)

        self._fill_from_preset()

    def _fill_from_preset(self) -> None:
        profile = self.preset.currentData()
        self.name.setText(profile["name"])
        self.instructions.setPlainText(profile["instructions"])

    def values(self) -> dict:
        return {
            "name": self.name.text().strip() or "Агент",
            "role": self.preset.currentData()["role"],
            "instructions": self.instructions.toPlainText().strip(),
            "model": self.model.currentData(),
        }


class ToolsDialog(QDialog):
    """Чем агент умеет действовать: список инструментов из его паспорта."""

    def __init__(self, agent: Agent, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Инструменты агента")
        self.resize(600, 520)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)

        passport = agent.passport()
        head = QLabel(
            f"{len(passport['tools'])} инструментов · рабочая папка: {passport['workspace']}"
            if passport["tools_enabled"]
            else "Инструменты сейчас выключены — агент может только отвечать текстом."
        )
        head.setObjectName("note")
        head.setWordWrap(True)
        lay.addWidget(head)

        area = QScrollArea()
        area.setWidgetResizable(True)
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(0, 8, 8, 0)
        body_lay.setSpacing(10)
        for tool in passport["tools"]:
            name = QLabel(tool["name"])
            name.setObjectName("stepHead")
            description = QLabel(tool["description"])
            description.setObjectName("agentRole")
            description.setWordWrap(True)
            body_lay.addWidget(name)
            body_lay.addWidget(description)
        body_lay.addStretch(1)
        area.setWidget(body)
        lay.addWidget(area)


class AgentWindow(QMainWindow):
    """Окно диалога: слева паспорт и настройки агента, справа лента и ввод."""

    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings("AI Advent", "Agent")   # масштаб переживает перезапуск
        self.scale = float(self.settings.value("ui/scale", 1.0))
        self.agents: list[Agent] = [Agent()]
        self.agent = self.agents[0]
        self.busy = False
        self.worker: AskWorker | None = None
        self._pending: Bubble | None = None
        self._loading = False  # чтобы программная установка контролов не била в агента

        self.setWindowTitle(f"Агент «{self.agent.profile.name}» — AI Advent")
        self.resize(1060, 720)
        self.setMinimumSize(880, 600)

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._build_sidebar())
        lay.addWidget(self._build_main(), 1)
        self.setCentralWidget(central)

        # Масштаб: Ctrl+колесо ловим фильтром на приложении (иначе событие
        # съедает лента), плюс привычные Ctrl+= / Ctrl+- / Ctrl+0.
        QApplication.instance().installEventFilter(self)
        QShortcut(QKeySequence.ZoomIn, self, activated=lambda: self._zoom(0.1))
        QShortcut(QKeySequence("Ctrl+="), self, activated=lambda: self._zoom(0.1))
        QShortcut(QKeySequence.ZoomOut, self, activated=lambda: self._zoom(-0.1))
        QShortcut(QKeySequence("Ctrl+0"), self, activated=lambda: self._set_scale(1.0))

        self._refresh()
        self.input.setFocus()

    # ------------------------------------------------------------ сборка окна --

    def _build_sidebar(self) -> QWidget:
        """Паспорт агента и его настройки: сущность видна как объект с состоянием."""
        panel = QFrame()
        panel.setObjectName("sidebar")
        self.sidebar = panel
        panel.setFixedWidth(round(300 * self.scale))
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)

        # Панель прокручивается: контролов много, и на большом масштабе они
        # перестают помещаться по высоте — без прокрутки Qt сжимал бы карточки.
        scroll = QScrollArea()
        scroll.setObjectName("sidebarScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.viewport().setStyleSheet(f"background: {PANEL};")
        inner = QWidget()
        inner.setObjectName("sidebarInner")
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        lay = QVBoxLayout(inner)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(13)

        card = QFrame()
        card.setObjectName("card")
        card_lay = QHBoxLayout(card)
        card_lay.setContentsMargins(14, 14, 14, 14)
        card_lay.setSpacing(12)
        avatar = QLabel("🤖")
        self.avatar = avatar
        avatar.setObjectName("avatar")
        avatar.setFixedSize(round(44 * self.scale), round(44 * self.scale))
        avatar.setAlignment(Qt.AlignCenter)
        card_lay.addWidget(avatar, 0, Qt.AlignTop)
        who = QVBoxLayout()
        who.setSpacing(2)
        self.name_label = QLabel()
        self.name_label.setObjectName("agentName")
        self.role_label = QLabel()
        self.role_label.setObjectName("agentRole")
        self.role_label.setWordWrap(True)
        self.id_label = QLabel()
        self.id_label.setObjectName("agentId")
        who.addWidget(self.name_label)
        who.addWidget(self.role_label)
        who.addWidget(self.id_label)
        card_lay.addLayout(who, 1)
        lay.addWidget(card)

        lay.addWidget(_section("МОДЕЛЬ"))
        self.model_box = QComboBox()
        for m in config.MODELS:
            self.model_box.addItem(m["label"], m["code"])
        self.model_box.currentIndexChanged.connect(self._on_model)
        lay.addWidget(self.model_box)

        lay.addWidget(_section("ТЕМПЕРАТУРА"))
        temp_row = QHBoxLayout()
        temp_row.setSpacing(10)
        self.temp = QSlider(Qt.Horizontal)
        self.temp.setRange(0, 195)  # шкала в сотых: диапазон Qwen — [0, 2)
        self.temp.valueChanged.connect(lambda v: self.temp_value.setText(f"{v / 100:.2f}"))
        self.temp.sliderReleased.connect(lambda: self._apply(temperature=self.temp.value() / 100))
        self.temp_value = QLabel()
        self.temp_value.setObjectName("agentId")
        self.temp_value.setFixedWidth(32)
        temp_row.addWidget(self.temp, 1)
        temp_row.addWidget(self.temp_value)
        lay.addLayout(temp_row)

        lay.addWidget(_section("ГЛУБИНА ПАМЯТИ, ПАР"))
        self.mem_box = QSpinBox()
        self.mem_box.setRange(0, 50)
        self.mem_box.editingFinished.connect(lambda: self._apply(memory_turns=self.mem_box.value()))
        lay.addWidget(self.mem_box)

        lay.addWidget(_section("КАК АГЕНТ РАБОТАЕТ"))
        self.tools_box = QCheckBox("Инструменты")
        self.tools_box.clicked.connect(lambda on: self._apply(tools_enabled=on))
        lay.addWidget(self.tools_box)
        self.plan_box = QCheckBox("Планировать действия")
        self.plan_box.clicked.connect(lambda on: self._apply(planning=on))
        lay.addWidget(self.plan_box)

        steps_row = QHBoxLayout()
        steps_row.setSpacing(8)
        steps_label = QLabel("потолок шагов")
        steps_label.setObjectName("agentRole")
        self.steps_box = QSpinBox()
        self.steps_box.setRange(1, 12)
        self.steps_box.setFixedWidth(64)
        self.steps_box.editingFinished.connect(lambda: self._apply(max_steps=self.steps_box.value()))
        steps_row.addWidget(steps_label, 1)
        steps_row.addWidget(self.steps_box)
        lay.addLayout(steps_row)

        tools_btn = _ghost("Чем агент умеет действовать")
        tools_btn.clicked.connect(lambda: ToolsDialog(self.agent, self).exec())
        lay.addWidget(tools_btn)

        lay.addStretch(1)

        stats = QHBoxLayout()
        stats.setSpacing(10)
        self.turns_stat, turns_card = _stat("обращений")
        self.mem_stat, mem_card = _stat("в памяти")
        stats.addWidget(turns_card)
        stats.addWidget(mem_card)
        lay.addLayout(stats)

        memory_btn = _ghost("Память агента")
        memory_btn.clicked.connect(lambda: MemoryDialog(self.agent, self).exec())
        lay.addWidget(memory_btn)

        reset_btn = _ghost("Сбросить память")
        reset_btn.clicked.connect(self._reset)
        lay.addWidget(reset_btn)

        note = QLabel("Память живёт внутри агента, а не в окне: интерфейс присылает только новое сообщение.")
        note.setObjectName("note")
        note.setWordWrap(True)
        lay.addWidget(note)
        return panel

    def _build_main(self) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Полоса агентов: переключение вынесено наверх, чтобы длинный список
        # уходил вбок и не выдавливал настройки в боковой панели.
        self.agent_bar = QWidget()
        self.agent_bar.setObjectName("agentBarRow")
        self.agent_bar.setFixedHeight(round(52 * self.scale))
        bar_row = QHBoxLayout(self.agent_bar)
        bar_row.setContentsMargins(0, 0, 16, 0)
        bar_row.setSpacing(10)

        tabs = QScrollArea()
        tabs.setObjectName("agentBar")
        tabs.setWidgetResizable(True)
        tabs.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        bar_inner = QWidget()
        bar_inner.setObjectName("agentBarInner")
        self.agent_list = QHBoxLayout(bar_inner)
        self.agent_list.setContentsMargins(16, 8, 8, 8)
        self.agent_list.setSpacing(8)
        tabs.setWidget(bar_inner)
        bar_row.addWidget(tabs, 1)

        # Кнопка вне прокрутки: сколько бы ни было агентов, она всегда на виду.
        new_agent = QPushButton("+ Новый агент")
        new_agent.setObjectName("newAgent")
        new_agent.setCursor(Qt.PointingHandCursor)
        new_agent.clicked.connect(self._create_agent)
        bar_row.addWidget(new_agent)

        lay.addWidget(self.agent_bar)

        header = QFrame()
        header.setObjectName("header")
        head_lay = QHBoxLayout(header)
        head_lay.setContentsMargins(24, 14, 20, 14)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.title = QLabel()
        self.title.setObjectName("title")
        titles.addWidget(self.title)
        head_lay.addLayout(titles, 1)
        self.dot = QLabel()
        self.dot.setFixedSize(9, 9)
        self.status = QLabel()
        self.status.setObjectName("status")
        head_lay.addWidget(self.dot)
        head_lay.addSpacing(7)
        head_lay.addWidget(self.status)
        lay.addWidget(header)

        self.chat = ChatView()
        lay.addWidget(self.chat, 1)

        self.examples = QScrollArea()
        self.examples.setWidgetResizable(True)
        self.examples.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.examples.setFixedHeight(round(52 * self.scale))
        strip = QWidget()
        strip_lay = QHBoxLayout(strip)
        strip_lay.setContentsMargins(24, 8, 20, 8)
        strip_lay.setSpacing(8)
        for example in config.EXAMPLES:
            chip = QPushButton(example["title"])
            chip.setObjectName("example")
            chip.setCursor(Qt.PointingHandCursor)
            chip.setToolTip(example["prompt"] + "\n\nинструменты: " + ", ".join(example["tools"]))
            chip.clicked.connect(lambda _, text=example["prompt"]: self._use_example(text))
            strip_lay.addWidget(chip)
        strip_lay.addStretch(1)
        self.examples.setWidget(strip)
        lay.addWidget(self.examples)

        composer = QFrame()
        composer.setObjectName("composer")
        comp_lay = QHBoxLayout(composer)
        comp_lay.setContentsMargins(24, 16, 20, 16)
        comp_lay.setSpacing(12)
        self.input = Composer()
        self.input.setObjectName("input")
        self.input.setPlaceholderText("Сообщение агенту…")
        self.input.setFixedHeight(56)
        self.input.submitted.connect(self._send)
        self.send_btn = QPushButton("Отправить")
        self.send_btn.setObjectName("send")
        self.send_btn.setFixedHeight(56)
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.clicked.connect(self._send)
        comp_lay.addWidget(self.input, 1)
        comp_lay.addWidget(self.send_btn)
        lay.addWidget(composer)

        self._set_status(OK, "готов")
        return wrap

    # ----------------------------------------------------------------- вывод --

    def _set_status(self, color: str, text: str) -> None:
        self._status_color = color
        radius = max(2, round(4 * self.scale))
        self.dot.setStyleSheet(f"background: {color}; border-radius: {radius}px;")
        self.status.setText(text)

    def _refresh(self) -> None:
        """Перечитать паспорт активного агента и обновить панель и список агентов."""
        self._render_agents()
        p = self.agent.passport()
        self.name_label.setText(p["name"])
        self.role_label.setText(p["role"])
        self.id_label.setText(f"id {p['id']}")
        self.title.setText(f"Диалог с агентом «{p['name']}»")
        self.turns_stat.setText(str(p["turns"]))
        self.mem_stat.setText(str(p["memory_messages"]))

        self._loading = True
        self.model_box.setCurrentIndex(max(0, self.model_box.findData(p["model"])))
        self.temp.setValue(round(p["temperature"] * 100))
        self.temp_value.setText(f"{p['temperature']:.2f}")
        self.mem_box.setValue(p["memory_turns"])
        self.tools_box.setChecked(p["tools_enabled"])
        self.plan_box.setChecked(p["planning"])
        self.steps_box.setValue(p["max_steps"])
        self._loading = False

    def _show_meta(self, reply: AgentReply) -> None:
        """Строка под ответом: метрики и ссылка на сырой обмен."""
        u = reply.usage or {}
        cost = f" · ${reply.cost_usd:.6f} (free-квота)" if reply.cost_usd is not None else ""
        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(6, 0, 0, 4)
        row_lay.setSpacing(10)
        name = config.model_label(reply.model).split(" · ")[0]
        meta = QLabel(
            f"{name} · {reply.elapsed_s} c · "
            f"{u.get('prompt_tokens', '?')}→{u.get('completion_tokens', '?')} токенов{cost} · "
            f"обращение #{reply.turn} · "
            f"{f'{len(reply.steps)} шаг(ов) инструментами' if reply.steps else 'без инструментов'} · "
            f"{reply.llm_calls} вызов(ов) модели"
        )
        meta.setObjectName("meta")
        meta.setWordWrap(True)
        link = QPushButton("сырой обмен →")
        link.setObjectName("link")
        link.setCursor(Qt.PointingHandCursor)
        link.clicked.connect(lambda: RawDialog(reply, self).exec())
        row_lay.addWidget(meta, 1)      # метрики занимают всю строку,
        row_lay.addWidget(link, 0)      # ссылка прижата к правому краю
        self.chat.add_row(row)

    def _show_trace(self, reply: AgentReply) -> None:
        """План агента и выполненные шаги — то, чего в чате не бывает."""
        if not reply.plan and not reply.steps:
            return
        card = QFrame()
        card.setObjectName("trace")
        card.setMinimumWidth(min(560, self.chat.bubble_max))
        card.setMaximumWidth(self.chat.bubble_max + 60)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 12, 16, 14)
        lay.setSpacing(8)

        if reply.plan:
            lay.addWidget(_trace_label("ПЛАН АГЕНТА"))
            for number, item in enumerate(reply.plan, 1):
                line = QLabel(f"{number}. {item}")
                line.setObjectName("planItem")
                line.setWordWrap(True)
                lay.addWidget(line)

        if reply.steps:
            lay.addWidget(_trace_label("ВЫПОЛНЕННЫЕ ШАГИ"))
            for step in reply.steps:
                arguments = ", ".join(
                    f"{k}={json.dumps(v, ensure_ascii=False)[:60]}" for k, v in step.arguments.items()
                )
                head = QLabel(
                    f"#{step.number}  {step.tool}({arguments})  ·  {step.title}"
                    f"{'' if step.ok else '  ·  ошибка'}  ·  {step.elapsed_s} c"
                )
                head.setObjectName("stepHead" if step.ok else "stepHeadErr")
                head.setWordWrap(True)
                lay.addWidget(head)

                box = QFrame()
                box.setObjectName("stepResult")
                box_lay = QVBoxLayout(box)
                box_lay.setContentsMargins(10, 7, 10, 8)
                result = QLabel(_clip(step.result, 700))
                result.setWordWrap(True)
                result.setTextInteractionFlags(Qt.TextSelectableByMouse)
                box_lay.addWidget(result)
                lay.addWidget(box)

        self.chat.add(card, Qt.AlignLeft)

    # ------------------------------------------------------- несколько агентов --

    def _render_agents(self) -> None:
        """Перерисовать список агентов сессии (активный подсвечен)."""
        while self.agent_list.count():
            item = self.agent_list.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # setParent(None) убирает карточку из окна сразу; без него старые
                # карточки живут до следующего прохода цикла событий.
                widget.setParent(None)
                widget.deleteLater()
        for agent in self.agents:
            tab = AgentItem(agent, agent is self.agent, deletable=len(self.agents) > 1)
            tab.chosen.connect(lambda a=agent: self._select_agent(a))
            tab.removed.connect(lambda a=agent: self._delete_agent(a))
            self.agent_list.addWidget(tab)
        self.agent_list.addStretch(1)

    def _create_agent(self) -> None:
        """Завести нового агента: своё имя, своя инструкция, своя память."""
        dialog = NewAgentDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        values = dialog.values()
        profile = AgentProfile(
            name=values["name"],
            role=values["role"],
            instructions=values["instructions"] or DEFAULT_PROFILE.instructions,
        )
        agent = Agent(profile=profile, model=values["model"])
        self.agents.append(agent)
        self._select_agent(agent)

    def _select_agent(self, agent: Agent) -> None:
        """Переключиться на другого агента и показать его собственную переписку."""
        if self.busy or agent is self.agent:
            return
        self.agent = agent
        self.setWindowTitle(f"Агент «{agent.profile.name}» — AI Advent")
        self._refresh()
        self._show_transcript()

    def _delete_agent(self, agent: Agent) -> None:
        """Удалить агента вместе с его памятью (последнего удалить нельзя)."""
        if self.busy or len(self.agents) == 1:
            return
        self.agents.remove(agent)
        if agent is self.agent:
            self.agent = self.agents[-1]
            self._show_transcript()
        self._refresh()

    def _show_transcript(self) -> None:
        """Перерисовать ленту из памяти агента: она хранится в нём, а не в окне."""
        self.chat.clear()
        memory = self.agent.memory
        if not memory:
            self.chat.add_system(
                f"Агент «{self.agent.profile.name}» активен. Память пуста — "
                "начните разговор или возьмите пример под полем ввода."
            )
            return
        for message in memory:
            self.chat.add_bubble(message["content"], "user" if message["role"] == "user" else "agent")
        self.chat.add_system(
            f"Показана память агента «{self.agent.profile.name}» "
            f"({len(memory)} сообщ.) — она хранится в самом агенте."
        )

    def _use_example(self, text: str) -> None:
        self.input.setPlainText(text)
        self.input.setFocus()

    # ------------------------------------------------------- масштаб интерфейса --

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
            self._zoom(0.1 if event.angleDelta().y() > 0 else -0.1)
            return True
        return super().eventFilter(obj, event)

    def _zoom(self, delta: float) -> None:
        self._set_scale(self.scale + delta)

    def _set_scale(self, scale: float) -> None:
        """Применить масштаб к стилям и к размерам, заданным в коде."""
        scale = round(min(MAX_SCALE, max(MIN_SCALE, scale)), 2)
        if abs(scale - self.scale) < 0.01:
            return
        self.scale = scale
        app = QApplication.instance()
        app.setStyleSheet(build_qss(scale))
        app.setFont(QFont("Segoe UI", max(7, round(10 * scale))))

        self.sidebar.setFixedWidth(round(300 * scale))
        self.avatar.setFixedSize(round(44 * scale), round(44 * scale))
        self.input.setFixedHeight(round(56 * scale))
        self.send_btn.setFixedHeight(round(56 * scale))
        self.examples.setFixedHeight(round(52 * scale))
        self.agent_bar.setFixedHeight(round(52 * scale))
        self.dot.setFixedSize(round(9 * scale), round(9 * scale))
        self._set_status(self._status_color, self.status.text())

        # Ширину пузырей считает сам пузырь по шрифту — после смены масштаба
        # пересчитываем, иначе текст остаётся в старой колонке.
        self.chat.bubble_limit = round(660 * scale)
        self.chat.refit()

        self.settings.setValue("ui/scale", scale)

    # --------------------------------------------------------------- действия --

    def _send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or self.busy:
            return
        self.input.clear()
        self.chat.add_bubble(text, "user")
        self._pending = self.chat.add_bubble("…", "agent")

        self.busy = True
        self.send_btn.setEnabled(False)
        self._set_status(WARN, "агент думает…")

        # Пока обращение идёт, кнопка выключена — двух одновременных не бывает.
        self.worker = AskWorker(self.agent, text)
        self.worker.done.connect(self._on_reply)
        self.worker.failed.connect(self._on_error)
        self.worker.start()

    def _on_reply(self, reply: AgentReply) -> None:
        self._pending.set_text(reply.text)
        self._show_meta(reply)
        self._show_trace(reply)
        self._set_status(OK, "готов")
        self._finish()

    def _on_error(self, message: str) -> None:
        self._pending.set_role("error")
        self._pending.set_text(message)
        self._set_status(ERR_TEXT, "ошибка")
        self._finish()

    def _finish(self) -> None:
        self.busy = False
        self.send_btn.setEnabled(True)
        self._refresh()
        self.input.setFocus()

    def _on_model(self) -> None:
        if not self._loading:
            self._apply(model=self.model_box.currentData())

    def _apply(self, **settings) -> None:
        """Настройки проверяет сам агент — окно только показывает результат."""
        try:
            self.agent.configure(**settings)
        except AgentError as e:
            self.chat.add_bubble(str(e), "error")
        self._refresh()  # при отказе контролы вернутся к значениям агента

    def _reset(self) -> None:
        self.agent.reset()
        self.chat.clear()
        self.chat.add_system("Память агента очищена — он снова не знает, о чём был разговор.")
        self._refresh()

    def apply_saved_scale(self) -> None:
        """Применить масштаб, сохранённый с прошлого запуска."""
        saved, self.scale = self.scale, 1.0
        self._set_scale(saved)

    def closeEvent(self, event) -> None:
        # Даём текущему обращению завершиться, чтобы поток не умер на полуслове.
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(3000)
        super().closeEvent(event)


def _trace_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("traceLabel")
    return label


def _clip(text: str, limit: int) -> str:
    """Обрезать длинный результат инструмента: полный текст есть в сыром обмене."""
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"\n…[ещё {len(text) - limit} символов]"


def _section(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("section")
    return label


def _ghost(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("ghost")
    button.setCursor(Qt.PointingHandCursor)
    return button


def _stat(caption: str) -> tuple[QLabel, QFrame]:
    """Карточка счётчика: крупное число и подпись под ним."""
    card = QFrame()
    card.setObjectName("stat")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(12, 8, 12, 9)
    lay.setSpacing(0)
    value = QLabel("0")
    value.setObjectName("statValue")
    label = QLabel(caption)
    label.setObjectName("statLabel")
    lay.addWidget(value)
    lay.addWidget(label)
    return value, card


def _dark_titlebar(window: QWidget) -> None:
    """Тёмная рамка окна на Windows — иначе светлый заголовок бьётся с темой."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            int(window.winId()), 20, ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int)
        )
    except Exception:
        pass


def build_app() -> tuple[QApplication, "AgentWindow"]:
    """Собрать приложение и окно (отдельно от main, чтобы окно можно было проверять)."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")  # предсказуемая база под QSS, одинаковая на всех системах
    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(BLACK))
    palette.setColor(QPalette.Base, QColor(CARD))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor(ACCENT2))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(QSS)
    return app, AgentWindow()


def main() -> None:
    app, window = build_app()
    window.show()
    window.apply_saved_scale()
    _dark_titlebar(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
