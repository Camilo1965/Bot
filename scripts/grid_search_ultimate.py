#!/usr/bin/env python3
"""
scripts/grid_search_ultimate.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Grid search EXHAUSTIVO sobre:
- Feature sets (5)
- Modelos (3 tipos × 3 configs = 9)
- Horizons (3)
- SL/TP combos (3)
- Thresholds (4)

Total: 5 × 9 × 3 × 3 × 4 = 1,620 backtests
Salida: CSV + TOP 10 ranking
"""

import os
import sys
import time
import warnings
from pathlib import Path
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
from strategy.quant_features import QUANT_FEATURE_COLS, add_quant_features

# ── Config ───────────────────────────────────────────────────────────────────
SYMBOL = "ETH/USDT"
TIMEFRAME = "15m"
LIMIT = 100_000
TRAIN_RATIO = 0.70

FEATURE_SETS = {
    "base": list(QUANT_FEATURE_COLS),
    "base+micro": list(QUANT_FEATURE_COLS) + ["range_pct", "body_pct", "upper_wick", "lower_wick", "volume_zscore", "vol_of_vol"],
    "base+htf": list(QUANT_FEATURE_COLS) + ["htf_rsi", "htf_macd_line", "htf_macd_signal", "htf_trend_strength", "is_asia", "is_london", "is_ny"],
    "base+div": list(QUANT_FEATURE_COLS) + ["rsi_divergence", "vol_regime"],
    "all": list(QUANT_FEATURE_COLS) + [
        "range_pct", "body_pct", "upper_wick", "lower_wick", "volume_zscore", "vol_of_vol",
        "htf_rsi", "htf_macd_line", "htf_macd_signal", "htf_trend_strength", "is_asia", "is_london", "is_ny",
        "rsi_divergence", "vol_regime",
    ],
}

MODEL_CONFIGS = {
    "xgb_light": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.10, "subsample": 0.8, "colsample_bytree": 0.8},
    "xgb_std": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.08, "subsample": 0.9, "colsample_bytree": 0.9},
    "xgb_deep": {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.06, "subsample": 0.85, "colsample_bytree": 0.85},
    "rf_light": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 50},
    "rf_std": {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 30},
    "rf_deep": {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 20},
}

HORIZONS = [6, 8, 12]
SL_TP_COMBOS = [(0.01, 0.02), (0.015, 0.03), (0.02, 0.04)]
THRESHOLDS = [0.55, 0.60, 0.65, 0.70]

# ── Feature engineering ──────────────────────────────────────────────────────

def add_all_advanced(df_15m: pd.DataFrame) -> pd.DataFrame:
    df = df_15m.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # Microstructure
    df["range_pct"] = (high - low) / close
    df["body_pct"] = (close - df["open"].astype(float)).abs() / (high - low).replace(0, np.nan)
    df["upper_wick"] = (high - close) / (high - low).replace(0, np.nan)
    df["lower_wick"] = (close - low) / (high - low).replace(0, np.nan)
    df["volume_zscore"] = ((volume - volume.rolling(20).mean()) / volume.rolling(20).std().replace(0, np.nan)).fillna(0)
    df["vol_of_vol"] = df["log_ret_1"].rolling(20).std().fillna(0)

    # Session + HTF (NO lookahead bias: each 15m bar only sees completed hourly data)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        df["hour"] = ts.dt.hour
        df["is_asia"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(float)
        df["is_london"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(float)
        df["is_ny"] = ((df["hour"] >= 16) & (df["hour"] < 24)).astype(float)

        # Hour index: each group of 4 bars = 1 hour
        df["hour_idx"] = np.arange(len(df)) // 4
        hourly_close = df.groupby("hour_idx")["close"].last()

        # Calculate HTF indicators on completed hours
        htf_rsi = _htf_rsi(hourly_close, 14)
        htf_macd_line, htf_macd_sig, _ = _htf_macd(hourly_close)
        htf_sma20 = hourly_close.rolling(20).mean()

        # Map to 15m bars using PREVIOUS completed hour (hour_idx - 1)
        df["htf_rsi"] = htf_rsi.reindex(df["hour_idx"] - 1).fillna(50).to_numpy()
        df["htf_macd_line"] = htf_macd_line.reindex(df["hour_idx"] - 1).fillna(0).to_numpy()
        df["htf_macd_signal"] = htf_macd_sig.reindex(df["hour_idx"] - 1).fillna(0).to_numpy()
        prev_sma = htf_sma20.reindex(df["hour_idx"] - 1).fillna(close.iloc[0]).to_numpy()
        df["htf_trend_strength"] = (close.to_numpy() / prev_sma - 1.0)
    else:
        for c in ["hour", "is_asia", "is_london", "is_ny", "htf_rsi", "htf_macd_line", "htf_macd_signal", "htf_trend_strength"]:
            df[c] = 0.0
        df["htf_rsi"] = 50.0

    # Divergence
    df["price_direction"] = np.sign(close.diff())
    df["rsi_direction"] = np.sign(df["rsi"].diff())
    df["rsi_divergence"] = (df["price_direction"] != df["rsi_direction"]).astype(float)
    df["vol_regime"] = pd.cut(df["atr"] / close, bins=[0, 0.005, 0.015, 1.0], labels=[0, 1, 2]).astype(float)

    return df.fillna(0)


def _htf_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(com=p-1, min_periods=p).mean()
    al = l.ewm(com=p-1, min_periods=p).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _htf_macd(s):
    ef = s.ewm(span=12, adjust=False).mean()
    es = s.ewm(span=26, adjust=False).mean()
    line = ef - es
    sig = line.ewm(span=9, adjust=False).mean()
    return line, sig, line - sig


# ── Data prep ────────────────────────────────────────────────────────────────

def prepare_data():
    print(f"Fetching {LIMIT} {TIMEFRAME} candles for {SYMBOL}...")
    raw = fetch_ohlcv_ccxt(SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)
    feat = add_quant_features(raw.copy(), base_timeframe_min=15)
    feat = add_all_advanced(feat)
    return feat


# ── Model training ───────────────────────────────────────────────────────────

def train_model(X_train, y_train, model_type, cfg):
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0

    if model_type == "xgb":
        m = XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        m.fit(X_train, y_train)
        return m
    elif model_type == "xgb_ensemble":
        models = []
        preds = []
        seeds = [42, 123, 999]
        for i, seed in enumerate(seeds):
            m = XGBClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"],
                subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                scale_pos_weight=spw,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            )
            m.fit(X_train, y_train)
            models.append(m)
            preds.append(m.predict_proba(X_train)[:, 1])
        meta_X = np.column_stack(preds)
        scaler = StandardScaler()
        meta_X_s = scaler.fit_transform(meta_X)
        meta = LogisticRegression(max_iter=1000, class_weight="balanced")
        meta.fit(meta_X_s, y_train)
        return models, meta, scaler
    else:  # rf
        m = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        m.fit(X_train, y_train)
        return m


def predict_model(model_obj, X, model_type):
    if isinstance(model_obj, tuple):
        models, meta, scaler = model_obj
        preds = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
        return meta.predict_proba(scaler.transform(preds))[:, 1]
    else:
        return model_obj.predict_proba(X)[:, 1]


# ── Backtest ─────────────────────────────────────────────────────────────────

def backtest_vectorized(closes, highs, lows, probs, threshold, horizon, sl, tp, risk_pct=0.015, max_pos_pct=0.30, slippage=0.0005, commission=0.0008):
    signals = np.where(probs >= threshold)[0]
    if len(signals) == 0:
        return {"trades": 0, "win_rate": 0, "pnl_pct": 0, "max_dd": 0, "profit_factor": 0, "trades_per_week": 0, "sharpe": 0}

    N = len(closes)
    balance = 10000.0
    trades = 0
    wins = 0
    gross_wins = 0.0
    gross_losses = 0.0
    peak = balance
    max_dd = 0.0
    pnls = []
    cooldown = 0

    for i in signals:
        if i + horizon >= N or i < cooldown:
            continue

        trades += 1
        entry_raw = float(closes[i])
        entry = entry_raw * (1.0 + slippage)
        sl_price = entry * (1.0 - sl)
        tp_price = entry * (1.0 + tp)

        raw_size = balance * risk_pct / sl
        position_size = min(raw_size, balance * max_pos_pct)

        f_highs = highs[i+1:i+1+horizon]
        f_lows = lows[i+1:i+1+horizon]

        tp_idx = np.where(f_highs >= tp_price)[0]
        sl_idx = np.where(f_lows <= sl_price)[0]

        if len(tp_idx) > 0 and (len(sl_idx) == 0 or tp_idx[0] < sl_idx[0]):
            exit_raw = tp_price
            exit_bar = tp_idx[0] + 1
        elif len(sl_idx) > 0:
            exit_raw = sl_price
            exit_bar = sl_idx[0] + 1
        else:
            exit_raw = float(closes[i+horizon])
            exit_bar = horizon

        exit_p = exit_raw * (1.0 - slippage)
        pnl_pct = (exit_p - entry) / entry
        pnl_usd = pnl_pct * position_size - (position_size * commission * 2)
        balance += pnl_usd
        pnls.append(pnl_usd)

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

        cooldown = i + exit_bar

    wr = wins / trades if trades > 0 else 0
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    pnl_pct = (balance - 10000) / 100
    test_days = len(closes) * 15 / 1440
    tpw = trades / max(test_days / 7, 1)

    # Sharpe aproximado (asumiendo 0 risk-free)
    if len(pnls) > 1:
        returns = np.array(pnls) / 10000
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(tpw * 52)
    else:
        sharpe = 0

    return {
        "trades": trades,
        "win_rate": round(wr, 3),
        "pnl_pct": round(pnl_pct, 2),
        "max_dd": round(max_dd * 100, 2),
        "profit_factor": round(pf, 2),
        "trades_per_week": round(tpw, 1),
        "sharpe": round(sharpe, 2),
        "final_balance": round(balance, 2),
    }


# ── Main grid search ─────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    feat = prepare_data()
    split = int(len(feat) * TRAIN_RATIO)
    train_df = feat.iloc[:split]
    test_df = feat.iloc[split:]

    closes = test_df["close"].to_numpy()
    highs = test_df["high"].to_numpy()
    lows = test_df["low"].to_numpy()

    results = []

    # Redefine model loop cleanly
    model_runs = []
    for fs_name, fs_cols in FEATURE_SETS.items():
        missing = [c for c in fs_cols if c not in train_df.columns]
        if missing:
            print(f"SKIP {fs_name}: missing {missing}")
            continue
        for mtype, cfg_name, cfg in [
            ("xgb", "xgb_light", MODEL_CONFIGS["xgb_light"]),
            ("xgb", "xgb_std", MODEL_CONFIGS["xgb_std"]),
            ("xgb", "xgb_deep", MODEL_CONFIGS["xgb_deep"]),
            ("xgb_ensemble", "xgb_ens_std", MODEL_CONFIGS["xgb_std"]),
            ("rf", "rf_light", MODEL_CONFIGS["rf_light"]),
            ("rf", "rf_std", MODEL_CONFIGS["rf_std"]),
            ("rf", "rf_deep", MODEL_CONFIGS["rf_deep"]),
        ]:
            model_runs.append((fs_name, fs_cols, mtype, cfg_name, cfg))

    total = len(model_runs) * len(HORIZONS) * len(SL_TP_COMBOS) * len(THRESHOLDS)
    done = 0

    print(f"\n{'='*80}")
    print(f"GRID SEARCH ULTIMATE — {total} combinations")
    print(f"{'='*80}\n")
    print(f"Models to train: {len(model_runs)}\n")

    for fs_name, fs_cols, mtype, cfg_name, cfg in model_runs:
        print(f"Training {mtype} ({cfg_name}) on {fs_name}...")
        X_test = test_df[fs_cols].to_numpy(dtype=np.float32)

        for horizon in HORIZONS:
            # Label depends on horizon
            train_fwd = train_df["close"].shift(-horizon) / train_df["close"] - 1.0
            train_label = (train_fwd > 0.015).astype(int)
            valid = ~train_label.isna() & train_df[fs_cols].notna().all(axis=1)
            X_tr = train_df.loc[valid, fs_cols].to_numpy(dtype=np.float32)
            y_tr = train_label[valid].to_numpy(dtype=np.int32)

            if len(y_tr) < 1000 or y_tr.sum() < 10:
                print(f"  SKIP h={horizon}: insufficient data/positives")
                done += len(SL_TP_COMBOS) * len(THRESHOLDS)
                continue

            t0 = time.time()
            model_obj = train_model(X_tr, y_tr, mtype, cfg)
            probs = predict_model(model_obj, X_test, mtype)
            print(f"  h={horizon} trained in {time.time()-t0:.1f}s | AUC={roc_auc_score((test_df['close'].shift(-horizon)/test_df['close']-1.0 > 0.015).fillna(0).astype(int), probs):.3f}")

            for (sl, tp) in SL_TP_COMBOS:
                for thresh in THRESHOLDS:
                    done += 1
                    res = backtest_vectorized(closes, highs, lows, probs, thresh, horizon, sl, tp)
                    results.append({
                        "feature_set": fs_name,
                        "model": cfg_name,
                        "model_type": mtype,
                        "horizon": horizon,
                        "sl": sl,
                        "tp": tp,
                        "threshold": thresh,
                        **res,
                    })
                    if done % 50 == 0:
                        elapsed = time.time() - start_time
                        eta = (elapsed / done) * (total - done)
                        print(f"  {done}/{total} done | ETA {eta/60:.1f}min")

    df_res = pd.DataFrame(results)
    df_res.to_csv("scripts/grid_search_results.csv", index=False)

    # Ranking
    df_res["score"] = df_res["pnl_pct"] / (df_res["max_dd"] + 1)
    df_res["score2"] = df_res["sharpe"]

    print(f"\n{'='*80}")
    print("TOP 10 — By Return/Drawdown")
    print(f"{'='*80}")
    top = df_res.nlargest(10, "score")[["feature_set", "model", "horizon", "sl", "tp", "threshold", "trades", "win_rate", "pnl_pct", "max_dd", "profit_factor", "sharpe", "score"]]
    print(top.to_string(index=False))

    print(f"\n{'='*80}")
    print("TOP 10 — By Sharpe")
    print(f"{'='*80}")
    top2 = df_res.nlargest(10, "score2")[["feature_set", "model", "horizon", "sl", "tp", "threshold", "trades", "win_rate", "pnl_pct", "max_dd", "profit_factor", "sharpe", "score2"]]
    print(top2.to_string(index=False))

    print(f"\n{'='*80}")
    print(f"Total time: {(time.time()-start_time)/60:.1f} minutes")
    print(f"Results saved to scripts/grid_search_results.csv")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
