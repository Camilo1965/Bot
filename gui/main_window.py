"""
gui.main_window
~~~~~~~~~~~~~~~

ClawdBot institutional desktop dashboard built with PyQt6 and pyqtgraph.

Layout (horizontal QSplitter)
──────────────────────────────
Left  │ Watchlist panel (QListWidget) + status cards
Center│ Candlestick / price chart (pyqtgraph PlotWidget)
Right │ Live log viewer (QTextEdit, read-only)

The window spawns a :class:`~gui.db_reader.DBReaderThread` in the background
that polls TimescaleDB every 2 seconds and delivers data via Qt signals.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.db_reader import DBReaderThread

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WATCHLIST: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
APP_TITLE = "ClawdBot - Institutional Dashboard"
DEFAULT_SYMBOL = WATCHLIST[0]

# Minimum rendered height for a doji (zero-body) candle, in price units
_MIN_CANDLE_HEIGHT: float = 0.01

# ---------------------------------------------------------------------------
# Dark-theme QSS stylesheet
# ---------------------------------------------------------------------------

_QSS = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
    font-size: 13px;
}

QSplitter::handle {
    background-color: #21262d;
    width: 2px;
    height: 2px;
}

/* ── Left panel ─────────────────────────────────────────── */
QListWidget {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    outline: none;
    padding: 4px;
}

QListWidget::item {
    padding: 8px 12px;
    border-radius: 4px;
    color: #c9d1d9;
}

QListWidget::item:selected {
    background-color: #1f6feb;
    color: #ffffff;
}

QListWidget::item:hover:!selected {
    background-color: #21262d;
}

/* ── Status cards ───────────────────────────────────────── */
QFrame#card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
}

/* ── Log viewer ─────────────────────────────────────────── */
QTextEdit {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    color: #8b949e;
    font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
    font-size: 12px;
    padding: 6px;
}

/* ── Labels ─────────────────────────────────────────────── */
QLabel#section_title {
    color: #58a6ff;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

QLabel#card_value {
    color: #e6edf3;
    font-size: 20px;
    font-weight: bold;
}

QLabel#card_label {
    color: #8b949e;
    font-size: 11px;
}

/* ── Scrollbars ─────────────────────────────────────────── */
QScrollBar:vertical {
    background: #0d1117;
    width: 8px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #30363d;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("section_title")
    return lbl


def _make_card(label: str, initial_value: str = "—") -> tuple[QFrame, QLabel]:
    """Return a dark card frame and its mutable value label."""
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 8, 12, 8)
    layout.setSpacing(2)

    val_lbl = QLabel(initial_value)
    val_lbl.setObjectName("card_value")

    cap_lbl = QLabel(label)
    cap_lbl.setObjectName("card_label")

    layout.addWidget(val_lbl)
    layout.addWidget(cap_lbl)
    return frame, val_lbl


# ---------------------------------------------------------------------------
# CandlestickItem  (custom pyqtgraph graphics item)
# ---------------------------------------------------------------------------


class CandlestickItem(pg.GraphicsObject):
    """Draws OHLCV candlesticks on a pyqtgraph plot.

    Parameters
    ----------
    data:   list of dicts with keys: ``t`` (float epoch), ``open``, ``high``,
            ``low``, ``close``.
    """

    def __init__(self, data: list[dict[str, float]] | None = None) -> None:
        super().__init__()
        self._data: list[dict[str, float]] = data or []
        self._picture: pg.QtGui.QPicture | None = None
        self.generatePicture()

    def setData(self, data: list[dict[str, float]]) -> None:  # noqa: N802
        self._data = data
        self._picture = None
        self.generatePicture()
        self.informViewBoundsChanged()

    def generatePicture(self) -> None:  # noqa: N802
        self._picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self._picture)
        p.setPen(pg.mkPen("w", width=1))
        w = 0.4  # half-width of candle body in x-units (minutes)
        for c in self._data:
            t = c["t"]
            # Wick
            if c["close"] >= c["open"]:
                p.setPen(pg.mkPen("#3fb950", width=1))
                p.setBrush(pg.mkBrush("#3fb950"))
            else:
                p.setPen(pg.mkPen("#f85149", width=1))
                p.setBrush(pg.mkBrush("#f85149"))
            p.drawLine(
                pg.QtCore.QPointF(t, c["low"]),
                pg.QtCore.QPointF(t, c["high"]),
            )
            p.drawRect(
                pg.QtCore.QRectF(
                    t - w,
                    min(c["open"], c["close"]),
                    2 * w,
                    abs(c["close"] - c["open"]) or _MIN_CANDLE_HEIGHT,
                )
            )
        p.end()

    def paint(  # noqa: D102
        self,
        p: pg.QtGui.QPainter,
        *args: Any,
    ) -> None:
        if self._picture is not None:
            p.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> pg.QtCore.QRectF:  # noqa: N802, D102
        if not self._data:
            return pg.QtCore.QRectF()
        xs = [c["t"] for c in self._data]
        ys = [c["low"] for c in self._data] + [c["high"] for c in self._data]
        return pg.QtCore.QRectF(
            min(xs) - 1,
            min(ys),
            max(xs) - min(xs) + 2,
            max(ys) - min(ys),
        )


# ---------------------------------------------------------------------------
# Left pane – watchlist
# ---------------------------------------------------------------------------


class WatchlistPane(QWidget):
    """Vertical panel showing the symbol list and live status cards."""

    def __init__(self, symbols: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(_make_section_label("Watchlist"))

        self.symbol_list = QListWidget()
        for sym in symbols:
            item = QListWidgetItem(sym)
            item.setFont(QFont("Segoe UI", 12))
            self.symbol_list.addItem(item)
        self.symbol_list.setCurrentRow(0)
        layout.addWidget(self.symbol_list)

        layout.addWidget(_make_section_label("Status"))

        _, self.pnl_label = _make_card("Total PnL (USDT)", "0.00")
        layout.addWidget(self.pnl_label.parent())

        _, self.trades_label = _make_card("Open Trades", "0")
        layout.addWidget(self.trades_label.parent())

        _, self.sentiment_label = _make_card("Sentiment Score", "—")
        layout.addWidget(self.sentiment_label.parent())

        layout.addStretch()

    # ------------------------------------------------------------------ #

    def update_status(
        self,
        total_pnl: float,
        active_trades: list[dict[str, Any]],
        sentiment: float | None,
    ) -> None:
        sign = "+" if total_pnl >= 0 else ""
        self.pnl_label.setText(f"{sign}{total_pnl:.2f}")
        self.pnl_label.setStyleSheet(
            "color: #3fb950;" if total_pnl >= 0 else "color: #f85149;"
        )

        self.trades_label.setText(str(len(active_trades)))

        if sentiment is not None:
            self.sentiment_label.setText(f"{sentiment:+.4f}")
            if sentiment >= 0.05:
                self.sentiment_label.setStyleSheet("color: #3fb950;")
            elif sentiment <= -0.05:
                self.sentiment_label.setStyleSheet("color: #f85149;")
            else:
                self.sentiment_label.setStyleSheet("color: #e6edf3;")
        else:
            self.sentiment_label.setText("—")


# ---------------------------------------------------------------------------
# Centre pane – candlestick chart
# ---------------------------------------------------------------------------


class ChartPane(QWidget):
    """pyqtgraph-based candlestick chart with volume bars."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(4)

        self._title_lbl = _make_section_label("BTC/USDT  •  15m")
        layout.addWidget(self._title_lbl)

        # ── pyqtgraph layout ─────────────────────────────────────────────
        pg.setConfigOption("background", "#0d1117")
        pg.setConfigOption("foreground", "#8b949e")

        self._graphics_layout = pg.GraphicsLayoutWidget()
        layout.addWidget(self._graphics_layout)

        # Price chart
        self._price_plot: pg.PlotItem = self._graphics_layout.addPlot(
            row=0, col=0, title=""
        )
        self._price_plot.showGrid(x=True, y=True, alpha=0.15)
        self._price_plot.getAxis("left").setStyle(tickFont=QFont("Consolas", 9))
        self._price_plot.getAxis("bottom").setStyle(tickFont=QFont("Consolas", 9))

        self._candles = CandlestickItem()
        self._price_plot.addItem(self._candles)

        # Volume chart (smaller, below price)
        self._graphics_layout.nextRow()
        self._vol_plot: pg.PlotItem = self._graphics_layout.addPlot(row=1, col=0)
        self._vol_plot.showGrid(x=False, y=True, alpha=0.10)
        self._vol_plot.setMaximumHeight(80)
        self._vol_bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush="#1f6feb")
        self._vol_plot.addItem(self._vol_bars)

        # Link x-axes
        self._vol_plot.setXLink(self._price_plot)

        # Empty-state annotation
        self._no_data_text = pg.TextItem(
            "No data – waiting for TimescaleDB…",
            anchor=(0.5, 0.5),
            color="#8b949e",
        )
        self._price_plot.addItem(self._no_data_text)
        self._no_data_text.setPos(0, 0)

    def set_symbol(self, symbol: str) -> None:
        self._title_lbl.setText(f"{symbol}  •  15m".upper())

    def update_chart(self, ohlcv: list[dict[str, Any]]) -> None:
        if not ohlcv:
            self._no_data_text.setVisible(True)
            return

        self._no_data_text.setVisible(False)

        # Convert bucket datetimes to epoch minutes for x-axis
        candle_data: list[dict[str, float]] = []
        xs: list[float] = []
        vols: list[float] = []

        for row in ohlcv:
            bucket: datetime = row["bucket"]
            if isinstance(bucket, datetime):
                t = bucket.timestamp() / 60.0  # minutes since epoch
            else:
                t = float(bucket)
            candle_data.append(
                {
                    "t": t,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                }
            )
            xs.append(t)
            vols.append(float(row.get("volume", 0)))

        self._candles.setData(candle_data)

        if xs:
            self._vol_bars.setOpts(
                x=np.array(xs),
                height=np.array(vols),
                width=0.6,
                brush="#1f6feb",
            )

        self._price_plot.autoRange()


# ---------------------------------------------------------------------------
# Right pane – log viewer
# ---------------------------------------------------------------------------


class LogPane(QWidget):
    """Read-only log display that auto-scrolls to the latest entry."""

    MAX_LINES = 500

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        layout.addWidget(_make_section_label("Live Log"))

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        layout.addWidget(self._text)

    def append(self, message: str) -> None:
        self._text.append(message)
        # Trim to MAX_LINES
        doc = self._text.document()
        while doc.blockCount() > self.MAX_LINES:
            cursor = self._text.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.select(cursor.SelectionType.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()  # remove trailing newline
        # Auto-scroll
        self._text.verticalScrollBar().setValue(
            self._text.verticalScrollBar().maximum()
        )

    def append_error(self, message: str) -> None:
        self._text.append(f'<span style="color:#f85149;">{message}</span>')
        self._text.verticalScrollBar().setValue(
            self._text.verticalScrollBar().maximum()
        )


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """ClawdBot institutional trading dashboard."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 800)
        self.setStyleSheet(_QSS)

        self._build_ui()
        self._start_db_reader()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        root_layout.addWidget(splitter)

        # Left pane
        self._watchlist_pane = WatchlistPane(WATCHLIST)
        self._watchlist_pane.setMinimumWidth(180)
        self._watchlist_pane.setMaximumWidth(260)
        splitter.addWidget(self._watchlist_pane)

        # Centre pane
        self._chart_pane = ChartPane()
        splitter.addWidget(self._chart_pane)

        # Right pane
        self._log_pane = LogPane()
        self._log_pane.setMinimumWidth(260)
        self._log_pane.setMaximumWidth(420)
        splitter.addWidget(self._log_pane)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        # Symbol selection
        self._watchlist_pane.symbol_list.currentTextChanged.connect(
            self._on_symbol_changed
        )

    # ------------------------------------------------------------------
    # DB reader
    # ------------------------------------------------------------------

    def _start_db_reader(self) -> None:
        self._db_reader = DBReaderThread(symbol=DEFAULT_SYMBOL, parent=self)
        self._db_reader.data_ready.connect(self._on_data_ready)
        self._db_reader.error.connect(self._log_pane.append_error)
        self._db_reader.log_message.connect(self._log_pane.append)
        self._db_reader.start()

        ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        self._log_pane.append(f"[{ts}] Dashboard started – connecting to TimescaleDB…")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_data_ready(self, payload: dict[str, Any]) -> None:
        self._watchlist_pane.update_status(
            total_pnl=payload["total_pnl"],
            active_trades=payload["active_trades"],
            sentiment=payload["sentiment"],
        )
        self._chart_pane.update_chart(payload["ohlcv"])

    def _on_symbol_changed(self, symbol: str) -> None:
        if symbol:
            self._db_reader.set_symbol(symbol)
            self._chart_pane.set_symbol(symbol)
            ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
            self._log_pane.append(f"[{ts}] Symbol switched to {symbol}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        self._db_reader.stop()
        self._db_reader.wait(3000)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry-point helper (also called by run_gui.py)
# ---------------------------------------------------------------------------


def launch() -> None:
    """Create the QApplication and show the main window."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
