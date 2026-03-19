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

import math
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtProperty
from PyQt6.QtGui import QColor, QFont, QLinearGradient
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
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

# Fractional price offset (0.1 %) applied when placing trade markers above/below candles
_MARKER_OFFSET_PCT: float = 0.001

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

/* ── Emergency stop button ──────────────────────────────── */
QPushButton#emergency_stop {
    background-color: #b91c1c;
    color: #ffffff;
    border: 2px solid #ef4444;
    border-radius: 6px;
    font-size: 13px;
    font-weight: bold;
    padding: 10px 8px;
    letter-spacing: 0.02em;
}

QPushButton#emergency_stop:hover {
    background-color: #dc2626;
    border-color: #fca5a5;
}

QPushButton#emergency_stop:pressed {
    background-color: #991b1b;
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
# TimeAxisItem – HH:MM labels for the X axis
# ---------------------------------------------------------------------------


class TimeAxisItem(pg.AxisItem):
    """Custom bottom axis that formats UNIX-minutes as ``HH:MM`` strings.

    The chart stores X values as ``epoch_seconds / 60`` (minutes since epoch).
    ``tickStrings`` converts those back to a wall-clock ``HH:MM`` in UTC.
    """

    def tickStrings(  # noqa: N802
        self,
        values: list[float],
        scale: float,
        spacing: float,
    ) -> list[str]:
        result: list[str] = []
        for v in values:
            try:
                dt = datetime.fromtimestamp(v * 60.0, tz=timezone.utc)
                result.append(dt.strftime("%H:%M"))
            except (OSError, OverflowError, ValueError):
                result.append("")
        return result


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
        p.setRenderHint(pg.QtGui.QPainter.RenderHint.Antialiasing)
        w = 0.4  # half-width of candle body in x-units (minutes)
        for c in self._data:
            t = c["t"]
            is_bull = c["close"] >= c["open"]

            # ── Color theme ───────────────────────────────────────────────
            if is_bull:
                wick_color = QColor("#00FFFF")   # neon cyan
                glow_color = QColor(0, 255, 255)
                body_hi_color = QColor("#00FFFF")
                body_lo_color = QColor("#006080")
            else:
                wick_color = QColor("#DC143C")   # crimson
                glow_color = QColor(220, 20, 60)
                body_hi_color = QColor("#FF44AA")  # magenta-glow
                body_lo_color = QColor("#6B0000")  # deep crimson

            # ── Wick glow (multi-pass softlight) ─────────────────────────
            for gw, alpha in ((6, 18), (4, 40), (2, 80)):
                gc = QColor(glow_color)
                gc.setAlpha(alpha)
                p.setPen(pg.mkPen(gc, width=gw))
                p.drawLine(
                    pg.QtCore.QPointF(t, c["low"]),
                    pg.QtCore.QPointF(t, c["high"]),
                )

            # Main wick
            p.setPen(pg.mkPen(wick_color, width=1))
            p.drawLine(
                pg.QtCore.QPointF(t, c["low"]),
                pg.QtCore.QPointF(t, c["high"]),
            )

            # ── Candle body with gradient fill ───────────────────────────
            body_top = max(c["open"], c["close"])
            body_bottom = min(c["open"], c["close"])
            body_height = body_top - body_bottom or _MIN_CANDLE_HEIGHT
            body_rect = pg.QtCore.QRectF(t - w, body_bottom, 2 * w, body_height)

            # Gradient runs from lo colour (visual bottom/lower-price) to hi colour
            # (visual top/higher-price). ObjectMode maps (0,0)=rect top-left to
            # (1,1)=rect bottom-right; because pyqtgraph replays the picture in a
            # y-up data space, the rect's "bottom" (ObjectMode y=1) is the
            # higher-priced (visually upper) end.
            gradient = QLinearGradient(0.5, 0.0, 0.5, 1.0)
            gradient.setCoordinateMode(QLinearGradient.CoordinateMode.ObjectMode)
            gradient.setColorAt(0.0, body_lo_color)
            gradient.setColorAt(1.0, body_hi_color)

            p.setPen(pg.mkPen(wick_color, width=1))
            p.setBrush(pg.QtGui.QBrush(gradient))
            p.drawRect(body_rect)
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
# PnLLabel – QLabel with a pyqtProperty-driven text colour
# ---------------------------------------------------------------------------


class PnLLabel(QLabel):
    """QLabel subclass that exposes a :class:`~PyQt6.QtGui.QColor` Qt property.

    The ``textColor`` property drives the displayed text color so that Qt's
    stylesheet engine can target it directly.  ``setPnlValue`` updates both the
    displayed text **and** the colour atomically via the property setter.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._text_color: QColor = QColor("#e6edf3")
        self._apply_color()

    # ── Qt property ──────────────────────────────────────────────────────

    def _get_text_color(self) -> QColor:
        return self._text_color

    def _set_text_color(self, color: QColor) -> None:
        self._text_color = color
        self._apply_color()

    textColor: QColor = pyqtProperty(  # type: ignore[assignment]
        QColor, fget=_get_text_color, fset=_set_text_color
    )

    def _apply_color(self) -> None:
        r, g, b = self._text_color.red(), self._text_color.green(), self._text_color.blue()
        self.setStyleSheet(f"color: rgb({r},{g},{b});")

    # ── Convenience ──────────────────────────────────────────────────────

    def setPnlValue(self, value: float) -> None:  # noqa: N802
        """Update text and colour for a PnL value (neon-green ≥ 0, crimson < 0)."""
        sign = "+" if value >= 0 else ""
        self.setText(f"{sign}{value:.2f}")
        self.textColor = QColor("#00FF88") if value >= 0 else QColor("#DC143C")


def _make_pnl_card(label: str, initial_value: str = "0.00") -> tuple[QFrame, PnLLabel]:
    """Return a dark card frame with a :class:`PnLLabel` value widget."""
    frame = QFrame()
    frame.setObjectName("card")
    card_layout = QVBoxLayout(frame)
    card_layout.setContentsMargins(12, 8, 12, 8)
    card_layout.setSpacing(2)

    val_lbl = PnLLabel(initial_value)
    val_lbl.setObjectName("card_value")

    cap_lbl = QLabel(label)
    cap_lbl.setObjectName("card_label")

    card_layout.addWidget(val_lbl)
    card_layout.addWidget(cap_lbl)
    return frame, val_lbl


# ---------------------------------------------------------------------------
# SentimentGaugeItem – arc-gauge / speedometer (pg.GraphicsObject)
# ---------------------------------------------------------------------------


class SentimentGaugeItem(pg.GraphicsObject):
    """Dynamic arc-gauge / speedometer for sentiment score in the range [-1, +1].

    The gauge is drawn as a semicircle (top half of a unit circle centred at
    the origin) with three colour-coded zones:

    * 0° – 60°   → green  (positive sentiment)
    * 60° – 120° → yellow (neutral)
    * 120° – 180° → red   (negative sentiment)

    A glowing needle rotates from 180° (most negative) through 90° (neutral)
    to 0° (most positive).
    """

    def __init__(self) -> None:
        super().__init__()
        self._value: float = 0.0
        self._picture: pg.QtGui.QPicture | None = None
        self._rebuild()

    # ── Public API ───────────────────────────────────────────────────────

    def setValue(self, value: float) -> None:  # noqa: N802
        self._value = max(-1.0, min(1.0, float(value)))
        self._picture = None
        self._rebuild()
        self.update()

    # ── Internal drawing ─────────────────────────────────────────────────

    def _rebuild(self) -> None:
        self._picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self._picture)
        p.setRenderHint(pg.QtGui.QPainter.RenderHint.Antialiasing)

        r = 1.0
        arc_rect = pg.QtCore.QRectF(-r, -r, 2 * r, 2 * r)

        # ── Background track (dark, wide) ─────────────────────────────
        p.setPen(pg.mkPen(QColor(40, 50, 60), width=8))
        p.setBrush(pg.QtGui.QBrush(Qt.BrushStyle.NoBrush))
        # Negative span = clockwise, which in pyqtgraph's y-up data space traces
        # through y = +r (the visually-upper semicircle).
        p.drawArc(arc_rect, 0 * 16, -180 * 16)

        # ── Colour zones (slightly narrower, drawn on top) ─────────────
        # Angles correspond to the arc's mathematical angle convention (y-up):
        #   0° = right (most positive), 90° = top (neutral), 180° = left (most negative)
        # In Qt clockwise terms, these map to: 0°→0°, 60°→-60°, 120°→-120°, 180°→-180°
        p.setPen(pg.mkPen(QColor(0, 200, 80, 180), width=5))    # green  0°–60°
        p.drawArc(arc_rect, 0 * 16, -60 * 16)

        p.setPen(pg.mkPen(QColor(255, 220, 0, 180), width=5))   # yellow 60°–120°
        p.drawArc(arc_rect, -60 * 16, -60 * 16)

        p.setPen(pg.mkPen(QColor(220, 50, 50, 180), width=5))   # red    120°–180°
        p.drawArc(arc_rect, -120 * 16, -60 * 16)

        # ── Needle ────────────────────────────────────────────────────
        # Map value [-1,+1] → angle [180°, 0°] in Qt counter-clockwise space
        needle_deg = 90.0 * (1.0 - self._value)
        needle_rad = math.radians(needle_deg)
        nx = math.cos(needle_rad) * 0.78
        ny = math.sin(needle_rad) * 0.78

        needle_color = QColor(0, 255, 136) if self._value >= 0 else QColor(220, 20, 60)

        # Glow passes (outer → inner)
        for gw, alpha in ((12, 20), (8, 45), (5, 90)):
            gc = QColor(needle_color)
            gc.setAlpha(alpha)
            p.setPen(pg.mkPen(gc, width=gw))
            p.drawLine(pg.QtCore.QPointF(0.0, 0.0), pg.QtCore.QPointF(nx, ny))

        # Solid needle
        p.setPen(pg.mkPen(needle_color, width=2))
        p.drawLine(pg.QtCore.QPointF(0.0, 0.0), pg.QtCore.QPointF(nx, ny))

        # Pivot dot
        piv = 0.07
        p.setPen(pg.mkPen(QColor(200, 210, 225), width=1))
        p.setBrush(pg.mkBrush(QColor(200, 210, 225)))
        p.drawEllipse(pg.QtCore.QRectF(-piv, -piv, 2 * piv, 2 * piv))

        p.end()

    # ── QGraphicsObject interface ─────────────────────────────────────────

    def paint(  # noqa: D102
        self,
        p: pg.QtGui.QPainter,
        *args: Any,
    ) -> None:
        if self._picture is not None:
            p.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> pg.QtCore.QRectF:  # noqa: N802, D102
        return pg.QtCore.QRectF(-1.15, -0.15, 2.3, 1.25)


# ---------------------------------------------------------------------------
# SentimentGaugeWidget – thin QWidget wrapper embedding the gauge in a card
# ---------------------------------------------------------------------------


class SentimentGaugeWidget(QWidget):
    """Compact card-style widget that hosts a :class:`SentimentGaugeItem`."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("card")
        outer.addWidget(frame)

        inner = QVBoxLayout(frame)
        inner.setContentsMargins(12, 8, 12, 8)
        inner.setSpacing(2)

        cap = QLabel("Sentiment Score")
        cap.setObjectName("card_label")
        inner.addWidget(cap)

        # pyqtgraph graphics container
        self._gfx = pg.GraphicsLayoutWidget()
        self._gfx.setBackground("#161b22")
        self._gfx.setFixedHeight(130)
        inner.addWidget(self._gfx)

        self._plot = self._gfx.addPlot()
        self._plot.hideAxis("left")
        self._plot.hideAxis("bottom")
        self._plot.setMenuEnabled(False)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setDefaultPadding(0.0)
        self._plot.setRange(xRange=(-1.35, 1.35), yRange=(-0.4, 1.25), padding=0)

        self._gauge = SentimentGaugeItem()
        self._plot.addItem(self._gauge)

        self._value_text = pg.TextItem(text="—", anchor=(0.5, 0.5), color="#8b949e")
        self._value_text.setFont(QFont("Consolas", 9))
        self._value_text.setPos(0.0, -0.22)
        self._plot.addItem(self._value_text)

    # ── Public API ───────────────────────────────────────────────────────

    def setValue(self, value: float | None) -> None:  # noqa: N802
        if value is None:
            self._gauge.setValue(0.0)
            self._value_text.setText("—")
            self._value_text.setColor("#8b949e")
        else:
            self._gauge.setValue(value)
            sign = "+" if value >= 0 else ""
            self._value_text.setText(f"{sign}{value:.4f}")
            if value >= 0.05:
                self._value_text.setColor("#00FF88")
            elif value <= -0.05:
                self._value_text.setColor("#DC143C")
            else:
                self._value_text.setColor("#e6edf3")


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

        _pnl_frame, self.pnl_label = _make_pnl_card("Total PnL (USDT)", "0.00")
        layout.addWidget(_pnl_frame)

        _, self.trades_label = _make_card("Open Trades", "0")
        layout.addWidget(self.trades_label.parent())

        self.sentiment_gauge = SentimentGaugeWidget()
        layout.addWidget(self.sentiment_gauge)

        layout.addStretch()

        # Emergency kill switch
        self.emergency_btn = QPushButton("🛑  EMERGENCY STOP / LIQUIDATE")
        self.emergency_btn.setObjectName("emergency_stop")
        self.emergency_btn.setMinimumHeight(48)
        layout.addWidget(self.emergency_btn)

    # ------------------------------------------------------------------ #

    def update_status(
        self,
        total_pnl: float,
        active_trades: list[dict[str, Any]],
        sentiment: float | None,
    ) -> None:
        self.pnl_label.setPnlValue(total_pnl)

        self.trades_label.setText(str(len(active_trades)))

        self.sentiment_gauge.setValue(sentiment)


# ---------------------------------------------------------------------------
# Centre pane – candlestick chart
# ---------------------------------------------------------------------------


class ChartPane(QWidget):
    """pyqtgraph-based candlestick chart with volume bars and trade markers."""

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

        # Price chart – use a custom TimeAxisItem for the bottom axis.
        time_axis = TimeAxisItem(orientation="bottom")
        time_axis.setStyle(tickFont=QFont("Consolas", 9))
        self._price_plot: pg.PlotItem = self._graphics_layout.addPlot(
            row=0, col=0, title="", axisItems={"bottom": time_axis}
        )
        self._price_plot.showGrid(x=True, y=True, alpha=0.15)
        self._price_plot.getAxis("left").setStyle(tickFont=QFont("Consolas", 9))

        self._candles = CandlestickItem()
        # Subtle neon glow on the whole candlestick item
        _candle_glow = QGraphicsDropShadowEffect()
        _candle_glow.setBlurRadius(14)
        _candle_glow.setColor(QColor(0, 200, 220, 130))
        _candle_glow.setOffset(0, 0)
        self._candles.setGraphicsEffect(_candle_glow)
        self._price_plot.addItem(self._candles)

        # Sniper-point scatter items: BUY (green ▲) and SELL (red ▼).
        self._buy_markers = pg.ScatterPlotItem(
            symbol="t1",  # upward-pointing triangle
            size=14,
            pen=pg.mkPen(None),
            brush=pg.mkBrush("#00ff88"),
        )
        self._sell_markers = pg.ScatterPlotItem(
            symbol="t",  # downward-pointing triangle
            size=14,
            pen=pg.mkPen(None),
            brush=pg.mkBrush("#ff4444"),
        )
        self._price_plot.addItem(self._buy_markers)
        self._price_plot.addItem(self._sell_markers)

        # Volume chart (smaller, below price) – use TimeAxisItem for HH:MM labels
        self._graphics_layout.nextRow()
        vol_time_axis = TimeAxisItem(orientation="bottom")
        vol_time_axis.setStyle(tickFont=QFont("Consolas", 9))
        self._vol_plot: pg.PlotItem = self._graphics_layout.addPlot(
            row=1, col=0, axisItems={"bottom": vol_time_axis}
        )
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

    def update_chart(
        self,
        ohlcv: list[dict[str, Any]],
        trades: list[dict[str, Any]] | None = None,
    ) -> None:
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

        # ── Trade marker (sniper-point) overlay ──────────────────────────
        self._update_trade_markers(trades or [], candle_data)

        self._price_plot.autoRange()

    def _update_trade_markers(
        self,
        trades: list[dict[str, Any]],
        candle_data: list[dict[str, float]],
    ) -> None:
        """Plot BUY (▲) and SELL (▼) markers on the price chart.

        A green upward triangle is drawn *below* the candle low at the BUY
        entry time.  A red downward triangle is drawn *above* the candle high
        at the SELL / trailing-stop exit time (or above the entry candle for
        open trades whose exit_time is not yet known).
        """
        if not trades:
            self._buy_markers.setData([], [])
            self._sell_markers.setData([], [])
            return

        # Build a lookup from nearest candle t-value to its low/high.
        candle_map: dict[float, dict[str, float]] = {c["t"]: c for c in candle_data}
        candle_ts = sorted(candle_map.keys())

        def _nearest_candle(time_val: Any) -> dict[str, float] | None:
            if time_val is None:
                return None
            if isinstance(time_val, datetime):
                t = time_val.timestamp() / 60.0
            else:
                t = float(time_val)
            if not candle_ts:
                return None
            # Find the closest candle by absolute distance.
            closest = min(candle_ts, key=lambda ct: abs(ct - t))
            return candle_map[closest]

        buy_xs: list[float] = []
        buy_ys: list[float] = []
        sell_xs: list[float] = []
        sell_ys: list[float] = []

        for trade in trades:
            entry_time_val = trade.get("entry_time")
            exit_time_val = trade.get("exit_time")
            exit_price = trade.get("exit_price")

            # BUY marker – below the candle low at entry time.
            entry_candle = _nearest_candle(entry_time_val)
            if entry_candle is not None:
                buy_xs.append(entry_candle["t"])
                buy_ys.append(entry_candle["low"] * (1.0 - _MARKER_OFFSET_PCT))

            # SELL / trailing-stop marker – above the candle high at exit time.
            if exit_time_val is not None and exit_price is not None:
                exit_candle = _nearest_candle(exit_time_val)
                if exit_candle is not None:
                    sell_xs.append(exit_candle["t"])
                    sell_ys.append(exit_candle["high"] * (1.0 + _MARKER_OFFSET_PCT))

        self._buy_markers.setData(
            x=np.array(buy_xs, dtype=float),
            y=np.array(buy_ys, dtype=float),
        )
        self._sell_markers.setData(
            x=np.array(sell_xs, dtype=float),
            y=np.array(sell_ys, dtype=float),
        )


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

        # Emergency kill switch
        self._watchlist_pane.emergency_btn.clicked.connect(
            self._on_emergency_stop
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
        self._chart_pane.update_chart(
            ohlcv=payload["ohlcv"],
            trades=payload.get("trades", []),
        )

    def _on_symbol_changed(self, symbol: str) -> None:
        if symbol:
            self._db_reader.set_symbol(symbol)
            self._chart_pane.set_symbol(symbol)
            ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
            self._log_pane.append(f"[{ts}] Symbol switched to {symbol}")

    def _on_emergency_stop(self) -> None:
        """Show a confirmation dialog; if confirmed, flag emergency stop in the DB."""
        reply = QMessageBox.warning(
            self,
            "⚠  Emergency Stop Confirmation",
            "This will LIQUIDATE ALL ACTIVE POSITIONS and halt the trading engine.\n\n"
            "Are you absolutely sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._db_reader.request_emergency_stop()
            ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
            self._log_pane.append_error(
                f"[{ts}] ⚠ EMERGENCY STOP issued – liquidating all positions…"
            )

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
