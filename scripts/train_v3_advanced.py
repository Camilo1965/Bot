#!/usr/bin/env python3
"""
scripts/train_v3_advanced.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pipeline ULTIMO v3:
- 100k velas ETH 15m
- Features avanzadas: multi-timeframe (1h), microstructure, momentum divergence
- Ensemble: 3 XGBoost + LogisticRegression meta-classifier
- Label: 1.5% threshold (R:R 1:2 con SL 1.5%, TP 3%)
- Backtest walk-forward OOS
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
from strategy.quant_features import QUANT_FEATURE_COLS, add_quant_features

# ── Configuration ────────────────────────────────────────────────────────────
SYMBOL = "ETH/USDT"
TIMEFRAME = "15m"
LIMIT = 100_000
HORIZON = 8
LABEL_THRESHOLD = 0.015  # 1.5%
TRAIN_RATIO = 0.70

# ── Advanced Feature Engineering ─────────────────────────────────────────────

def add_advanced_features(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Add microstructure + multi-timeframe + divergence features."""
    df = df_15m.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # 1. Microstructure proxies
    df["range_pct"] = (high - low) / close
    df["body_pct"] = (close - df["open"].astype(float)).abs() / (high - low).replace(0, np.nan)
    df["upper_wick"] = (high - close) / (high - low).replace(0, np.nan)
    df["lower_wick"] = (close - low) / (high - low).replace(0, np.nan)
    df["volume_zscore"] = ((volume - volume.rolling(20).mean()) / volume.rolling(20).std().replace(0, np.nan)).fillna(0)
    df["vol_of_vol"] = df["log_ret_1"].rolling(20).std().fillna(0)

    # 2. Multi-timeframe: aggregate 15m to 1h and project back
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.RangeIndex(len(df))
    if "timestamp" in df.columns:
        df["hour"] = ts.dt.hour
        df["is_asia"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(float)
        df["is_london"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(float)
        df["is_ny"] = ((df["hour"] >= 16) & (df["hour"] < 24)).astype(float)

        # 1h aggregation
        df_temp = df.copy()
        df_temp["timestamp"] = ts
        df_temp.set_index("timestamp", inplace=True)
        
        htf = df_temp.resample("1h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        # RSI 1h
        htf_rsi = _htf_rsi(htf["close"], 14)
        htf_rsi = htf_rsi.reindex(df_temp.index, method="ffill")
        df["htf_rsi"] = htf_rsi.fillna(50).values
        
        # MACD 1h
        htf_macd_line, htf_macd_sig, _ = _htf_macd(htf["close"])
        htf_macd_line = htf_macd_line.reindex(df_temp.index, method="ffill").fillna(0).values
        htf_macd_sig = htf_macd_sig.reindex(df_temp.index, method="ffill").fillna(0).values
        df["htf_macd_line"] = htf_macd_line
        df["htf_macd_signal"] = htf_macd_sig
        
        # Trend strength: close vs 20-period 1h mean
        htf_sma20 = htf["close"].rolling(20).mean()
        htf_sma20_aligned = htf_sma20.reindex(df_temp.index, method="ffill").fillna(close.iloc[0]).to_numpy()
        df["htf_trend_strength"] = (close.to_numpy() / htf_sma20_aligned - 1.0)
    else:
        df["hour"] = 0
        df["is_asia"] = 0.0
        df["is_london"] = 0.0
        df["is_ny"] = 0.0
        df["htf_rsi"] = 50.0
        df["htf_macd_line"] = 0.0
        df["htf_macd_signal"] = 0.0
        df["htf_trend_strength"] = 0.0

    # 3. Momentum divergence: price vs RSI direction
    df["price_direction"] = np.sign(close.diff())
    df["rsi_direction"] = np.sign(df["rsi"].diff())
    df["rsi_divergence"] = (df["price_direction"] != df["rsi_direction"]).astype(float)

    # 4. Volatility regime
    df["vol_regime"] = pd.cut(df["atr"] / close, bins=[0, 0.005, 0.015, 1.0], labels=[0, 1, 2]).astype(float)

    return df.fillna(0)


def _htf_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _htf_macd(series: pd.Series):
    ema_f = series.ewm(span=12, adjust=False).mean()
    ema_s = series.ewm(span=26, adjust=False).mean()
    line = ema_f - ema_s
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist


# ── Feature columns ──────────────────────────────────────────────────────────
BASE_FEATURES = list(QUANT_FEATURE_COLS)
ADVANCED_FEATURES = [
    "range_pct", "body_pct", "upper_wick", "lower_wick",
    "volume_zscore", "vol_of_vol",
    "hour", "is_asia", "is_london", "is_ny",
    "htf_rsi", "htf_macd_line", "htf_macd_signal", "htf_trend_strength",
    "rsi_divergence", "vol_regime",
]
ALL_FEATURES = BASE_FEATURES + ADVANCED_FEATURES


# ── Data preparation ─────────────────────────────────────────────────────────

def prepare_data(limit: int = LIMIT) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    print(f"Fetching {limit} {TIMEFRAME} candles for {SYMBOL}...")
    raw = fetch_ohlcv_ccxt(SYMBOL, timeframe=TIMEFRAME, limit=limit)
    
    feat = add_quant_features(raw.copy(), base_timeframe_min=15)
    feat = add_advanced_features(feat)
    
    # Label: 1.5% forward return
    fwd = feat["close"].shift(-HORIZON) / feat["close"] - 1.0
    feat["label"] = (fwd > LABEL_THRESHOLD).astype(int)
    
    feat = feat.dropna(subset=ALL_FEATURES + ["label"]).reset_index(drop=True)
    
    split = int(len(feat) * TRAIN_RATIO)
    train = feat.iloc[:split]
    test = feat.iloc[split:]
    
    return train, test, feat["label"]


# ── Ensemble + Meta-classifier ───────────────────────────────────────────────

def train_ensemble(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """Train 3 XGBoost models + logistic regression meta-classifier."""
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0
    
    models = []
    preds = []
    
    configs = [
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "seed": 42},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "seed": 123},
        {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.06, "subsample": 0.85, "colsample_bytree": 0.85, "seed": 999},
    ]
    
    for cfg in configs:
        model = XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=cfg["seed"],
            n_jobs=-1,
            early_stopping_rounds=20,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        models.append(model)
        preds.append(model.predict_proba(X_val)[:, 1])
    
    # Meta-features: probabilities from each model
    meta_X = np.column_stack(preds)
    
    # Scale for logistic regression
    scaler = StandardScaler()
    meta_X_scaled = scaler.fit_transform(meta_X)
    
    meta_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    meta_clf.fit(meta_X_scaled, y_val)
    
    return models, meta_clf, scaler


def predict_ensemble(models, meta_clf, scaler, X):
    preds = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    preds_scaled = scaler.transform(preds)
    return meta_clf.predict_proba(preds_scaled)[:, 1]


# ── Backtest ─────────────────────────────────────────────────────────────────

def backtest(test_df: pd.DataFrame, models, meta_clf, scaler, threshold: float = 0.55):
    X_test = test_df[ALL_FEATURES].to_numpy(dtype=np.float32)
    probs = predict_ensemble(models, meta_clf, scaler, X_test)
    
    balance = 10000.0
    trades = 0
    wins = 0
    gross_wins = 0.0
    gross_losses = 0.0
    max_dd = 0.0
    peak = balance
    
    closes = test_df["close"].to_numpy()
    highs = test_df["high"].to_numpy()
    lows = test_df["low"].to_numpy()
    
    SL_PCT = 0.015
    TP_PCT = 0.030
    RISK_PCT = 0.015
    MAX_POS_PCT = 0.30  # max 30% of balance in one trade
    SLIPPAGE = 0.0005   # 0.05% slippage each side
    COMMISSION = 0.0008  # 0.08% taker fee
    
    in_trade = False
    cooldown = 0
    
    for i in range(len(test_df) - HORIZON):
        if cooldown > 0:
            cooldown -= 1
            continue
        
        if probs[i] < threshold:
            continue
        
        trades += 1
        entry_raw = float(closes[i])
        entry = entry_raw * (1.0 + SLIPPAGE)  # buy slippage
        sl_price = entry * (1.0 - SL_PCT)
        tp_price = entry * (1.0 + TP_PCT)
        
        # Position sizing: risk 1.5% of balance, but cap at 30% of balance
        raw_size = balance * RISK_PCT / SL_PCT
        position_size = min(raw_size, balance * MAX_POS_PCT)
        
        # Find exit
        future_highs = highs[i+1:i+1+HORIZON]
        future_lows = lows[i+1:i+1+HORIZON]
        
        tp_idx = np.where(future_highs >= tp_price)[0]
        sl_idx = np.where(future_lows <= sl_price)[0]
        
        if len(tp_idx) > 0 and (len(sl_idx) == 0 or tp_idx[0] < sl_idx[0]):
            exit_raw = tp_price
            is_win = True
            exit_bar = tp_idx[0] + 1
        elif len(sl_idx) > 0:
            exit_raw = sl_price
            is_win = False
            exit_bar = sl_idx[0] + 1
        else:
            exit_raw = float(closes[i+HORIZON])
            is_win = exit_raw > entry_raw
            exit_bar = HORIZON
        
        exit_p = exit_raw * (1.0 - SLIPPAGE)  # sell slippage
        pnl_pct = (exit_p - entry) / entry
        pnl_usd = pnl_pct * position_size
        
        # Costs
        commission_usd = position_size * COMMISSION * 2  # open + close
        pnl_usd -= commission_usd
        
        balance += pnl_usd
        if pnl_usd > 0:
            wins += 1
            gross_wins += pnl_usd
        else:
            gross_losses += abs(pnl_usd)
        
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak
        if dd > max_dd:
            max_dd = dd
        
        # Cooldown: can't trade until this trade would have closed
        cooldown = exit_bar
    
    wr = wins / trades if trades > 0 else 0
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    pnl_pct = (balance - 10000) / 100
    test_days = len(test_df) * 15 / 1440
    
    return {
        "trades": trades,
        "win_rate": round(wr, 3),
        "pnl_pct": round(pnl_pct, 2),
        "max_dd": round(max_dd * 100, 2),
        "profit_factor": round(pf, 2),
        "trades_per_week": round(trades / max(test_days / 7, 1), 1),
        "final_balance": round(balance, 2),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    train_df, test_df, labels = prepare_data(LIMIT)
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}, Pos rate: {labels.mean():.3f}")
    
    X_train = train_df[ALL_FEATURES].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int32)
    
    # Validation set for early stopping
    val_split = int(len(X_train) * 0.85)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]
    
    print("Training ensemble...")
    models, meta_clf, scaler = train_ensemble(X_tr, y_tr, X_val, y_val)
    
    # Evaluate on test
    X_test = test_df[ALL_FEATURES].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int32)
    test_probs = predict_ensemble(models, meta_clf, scaler, X_test)
    test_preds = (test_probs >= 0.55).astype(int)
    
    print(f"\nTest AUC: {roc_auc_score(y_test, test_probs):.4f}")
    print(f"Test ACC: {accuracy_score(y_test, test_preds):.4f}")
    print(f"Test F1:  {f1_score(y_test, test_preds):.4f}")
    
    # Backtest with different thresholds
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS (v3 Advanced)")
    print("=" * 70)
    print(f"{'Threshold':<12} | {'Trades':<8} | {'Win%':<6} | {'PnL%':<8} | {'DD%':<6} | {'PF':<6} | {'Trd/Wk':<8}")
    print("-" * 70)
    
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        res = backtest(test_df, models, meta_clf, scaler, threshold=thresh)
        print(
            f"{thresh:<12.2f} | {res['trades']:<8} | {res['win_rate']*100:<6.1f} | "
            f"{res['pnl_pct']:<+8.2f} | {res['max_dd']:<6.2f} | {res['profit_factor']:<6.2f} | {res['trades_per_week']:<8.1f}"
        )
    
    print("\n" + "=" * 70)
    print("PROJECTION (based on threshold 0.60)")
    print("=" * 70)
    res = backtest(test_df, models, meta_clf, scaler, threshold=0.60)
    weekly_return = res["pnl_pct"] / max(res["trades_per_week"] * (len(test_df) * 15 / 1440 / 7), 1)
    print(f"Trades per week:     {res['trades_per_week']}")
    print(f"Win rate:            {res['win_rate']*100:.1f}%")
    print(f"Total return (test): {res['pnl_pct']:+.2f}%")
    print(f"Max drawdown:        {res['max_dd']:.2f}%")
    print(f"Profit factor:       {res['profit_factor']:.2f}")
    print(f"\nWeekly estimate:     ~{weekly_return*7:+.2f}%")
    print(f"Monthly estimate:    ~{weekly_return*30:+.2f}%")
    print(f"Final balance:       ${res['final_balance']:,.2f}")


if __name__ == "__main__":
    main()
