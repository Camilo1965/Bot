"""
strategy.pair_trader
~~~~~~~~~~~~~~~~~~~~

BTC/ETH ratio market-neutral pair trading strategy.

Tests for Engle-Granger cointegration, computes a rolling z-score of the
BTC/ETH price ratio, and emits entry/exit signals based on z-score thresholds.

Usage (programmatic)::

    from strategy.pair_trader import PairTrader

    pt = PairTrader(window=60, z_entry=2.0, z_exit=0.5, z_stop=4.0)
    if pt.fit(btc_close, eth_close):
        signal = pt.signal(btc_price, eth_price)

See also: scripts/pair_trade_backtest.py
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Signal = Literal["BUY_ETH_SELL_BTC", "BUY_BTC_SELL_ETH", "CLOSE", "HOLD"]

# ── Optional statsmodels import ───────────────────────────────────────────────
try:
    from statsmodels.tsa.stattools import coint as _sm_coint

    _STATSMODELS_AVAILABLE = True
except ImportError:
    _sm_coint = None  # type: ignore[assignment]
    _STATSMODELS_AVAILABLE = False
    logger.warning(
        "statsmodels not installed — cointegration test will be skipped. "
        "Install with: pip install statsmodels"
    )


class PairTrader:
    """
    Market-neutral BTC/ETH pair trader based on ratio z-score mean reversion.

    Parameters
    ----------
    window : int
        Rolling window (in bars) for z-score computation.
    z_entry : float
        Absolute z-score threshold to open a position.
    z_exit : float
        Absolute z-score threshold to close a position (mean reversion).
    z_stop : float
        Absolute z-score threshold for stop-loss (pair diverged too far).
    coint_pvalue : float
        Maximum p-value to consider the pair cointegrated.
    """

    def __init__(
        self,
        window: int = 60,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        z_stop: float = 4.0,
        coint_pvalue: float = 0.05,
    ) -> None:
        self.window = window
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.coint_pvalue = coint_pvalue

        self._is_coint: bool = False
        self._ratio_series: pd.Series | None = None
        self._zscore_series: pd.Series | None = None
        self._coint_pvalue_result: float | None = None
        self._cointegration_tested: bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, btc_close: pd.Series, eth_close: pd.Series) -> bool:
        """
        Fit on historical price series.

        Tests Engle-Granger cointegration (if statsmodels available) and
        computes the ratio series + rolling z-score.

        Parameters
        ----------
        btc_close : pd.Series
            BTC/USDT close prices (aligned with eth_close).
        eth_close : pd.Series
            ETH/USDT close prices.

        Returns
        -------
        bool
            True if cointegrated (or if statsmodels unavailable, always True).
        """
        if len(btc_close) != len(eth_close):
            raise ValueError(
                f"Price series length mismatch: BTC={len(btc_close)}, ETH={len(eth_close)}"
            )
        if len(btc_close) < self.window + 10:
            logger.warning(
                "fit(): series too short (%d rows) for window=%d", len(btc_close), self.window
            )
            self._is_coint = False
            return False

        # Align index
        btc = btc_close.reset_index(drop=True).astype(float)
        eth = eth_close.reset_index(drop=True).astype(float)

        # Cointegration test
        if _STATSMODELS_AVAILABLE:
            try:
                _, pvalue, _ = _sm_coint(btc, eth)
                self._coint_pvalue_result = float(pvalue)
                self._is_coint = pvalue < self.coint_pvalue
                self._cointegration_tested = True
                logger.info(
                    "Engle-Granger coint p=%.4f → %s",
                    pvalue,
                    "COINTEGRATED" if self._is_coint else "NOT cointegrated",
                )
            except Exception as exc:
                logger.warning("Cointegration test failed: %s — treating as cointegrated", exc)
                self._is_coint = True
                self._cointegration_tested = False
        else:
            logger.info("statsmodels unavailable — skipping cointegration test (treating as True)")
            self._is_coint = True
            self._cointegration_tested = False

        # Compute ratio and rolling z-score regardless of coint result
        ratio = btc / eth.replace(0, np.nan)
        rolling_mean = ratio.rolling(window=self.window, min_periods=self.window // 2).mean()
        rolling_std = ratio.rolling(window=self.window, min_periods=self.window // 2).std()
        zscore = (ratio - rolling_mean) / rolling_std.replace(0, np.nan)

        self._ratio_series = ratio
        self._zscore_series = zscore

        return self._is_coint

    def signal(self, btc_price: float, eth_price: float) -> Signal:
        """
        Compute current signal from live prices.

        Appends the new ratio observation to the fitted series, recomputes
        the rolling z-score, and returns a signal.

        Returns
        -------
        Signal
            "BUY_ETH_SELL_BTC" — z high (BTC over-valued vs ETH)
            "BUY_BTC_SELL_ETH" — z low (ETH over-valued vs BTC)
            "CLOSE"            — z reverted inside z_exit band
            "HOLD"             — no action
        """
        if self._ratio_series is None or self._zscore_series is None:
            logger.warning("signal() called before fit()")
            return "HOLD"

        if eth_price <= 0:
            return "HOLD"

        new_ratio = btc_price / eth_price
        new_ratio_s = pd.Series([new_ratio], index=[len(self._ratio_series)])
        extended = pd.concat([self._ratio_series, new_ratio_s])

        rolling_mean = extended.rolling(window=self.window, min_periods=self.window // 2).mean()
        rolling_std = extended.rolling(window=self.window, min_periods=self.window // 2).std()
        zscore_extended = (extended - rolling_mean) / rolling_std.replace(0, np.nan)

        current_z = float(zscore_extended.iloc[-1]) if not zscore_extended.empty else float("nan")

        if np.isnan(current_z):
            return "HOLD"

        # Stop-loss: z too extreme — exit
        if abs(current_z) >= self.z_stop:
            return "CLOSE"

        # Entry: z exceeds entry threshold
        if current_z >= self.z_entry:
            # BTC/ETH ratio is high → BTC overvalued → short BTC, long ETH
            return "BUY_ETH_SELL_BTC"
        if current_z <= -self.z_entry:
            # BTC/ETH ratio is low → ETH overvalued → long BTC, short ETH
            return "BUY_BTC_SELL_ETH"

        # Exit: z reverted inside the exit band
        if abs(current_z) <= self.z_exit:
            return "CLOSE"

        return "HOLD"

    def current_zscore(self) -> float | None:
        """Return the most recent fitted z-score, or None if not fitted."""
        if self._zscore_series is None or self._zscore_series.empty:
            return None
        val = self._zscore_series.iloc[-1]
        return float(val) if not np.isnan(val) else None

    def is_cointegrated(self) -> bool:
        """True if the last fit() call found the pair cointegrated."""
        return self._is_coint

    def cointegration_tested(self) -> bool:
        """True if statsmodels was available and coint test ran successfully."""
        return self._cointegration_tested

    def get_ratio_series(self) -> pd.Series | None:
        """Return the BTC/ETH ratio series from the last fit() call."""
        return self._ratio_series

    def get_zscore_series(self) -> pd.Series | None:
        """Return the rolling z-score series from the last fit() call."""
        return self._zscore_series

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        coint_str = (
            f"coint={self._is_coint} p={self._coint_pvalue_result:.4f}"
            if self._coint_pvalue_result is not None
            else "not_fitted"
        )
        return (
            f"PairTrader(window={self.window}, z_entry={self.z_entry}, "
            f"z_exit={self.z_exit}, z_stop={self.z_stop}, {coint_str})"
        )
