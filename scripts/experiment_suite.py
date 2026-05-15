#!/usr/bin/env python3
import sys
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from rich.console import Console
from rich.table import Table

# Add repo root to path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from strategy.quant_features import add_quant_features, QUANT_FEATURE_COLS, forward_return_label, DEFAULT_TAKER_FEE_RATE
from scripts.quant_sweep import fetch_ohlcv_ccxt

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("experiment_suite")
console = Console()

# --- Configuration ---
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
TIMEFRAMES = ["5m", "15m", "30m"]
HORIZONS = [4, 8, 12] # Bars to look ahead
COST = 2.0 * DEFAULT_TAKER_FEE_RATE + 0.0005 # Round-trip cost estimate

def run_experiment(symbol, timeframe, horizon, model_params):
    logger.info(f"Experiment: {symbol} | {timeframe} | Horizon: {horizon}")
    
    try:
        # 1. Fetch Data
        df = fetch_ohlcv_ccxt(symbol, timeframe=timeframe, limit=5000)
        
        # 2. Build Dataset
        feat = add_quant_features(df, base_timeframe_min=int(timeframe[:-1]))
        feat["label"] = forward_return_label(feat["close"], horizon, COST)
        feat = feat.dropna(subset=QUANT_FEATURE_COLS + ["label"]).reset_index(drop=True)
        
        if len(feat) < 500:
            return {"error": "Insufficient data after processing"}
            
        # 3. Train/Test Split (Walk-forward style)
        train_ratio = 0.75
        split_i = int(len(feat) * train_ratio)
        train_df = feat.iloc[:split_i]
        test_df = feat.iloc[split_i:]
        
        X_train = train_df[QUANT_FEATURE_COLS].to_numpy()
        y_train = train_df["label"].to_numpy()
        X_test = test_df[QUANT_FEATURE_COLS].to_numpy()
        y_test = test_df["label"].to_numpy()
        
        # 4. Train Model
        model = XGBClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # 5. Evaluate Performance (Simulation)
        proba = model.predict_proba(X_test)[:, 1]
        
        results = []
        for th in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            mask = proba >= th
            n_signals = int(mask.sum())
            
            if n_signals < 5:
                results.append({
                    "threshold": th,
                    "pf": 0.0,
                    "wr": 0.0,
                    "signals": n_signals,
                    "exp": 0.0
                })
                continue
                
            # Forward returns for signals
            fwd_ret = test_df["close"].shift(-horizon).values / test_df["close"].values - 1.0
            sig_pnl = fwd_ret[mask] - COST
            sig_pnl = sig_pnl[~np.isnan(sig_pnl)]
            
            wins = sig_pnl[sig_pnl > 0]
            losses = sig_pnl[sig_pnl <= 0]
            
            gp = float(wins.sum())
            gl = float(abs(losses.sum()))
            pf = gp / gl if gl > 0 else float('inf')
            wr = len(wins) / len(sig_pnl) if len(sig_pnl) > 0 else 0
            exp = float(sig_pnl.mean())
            
            results.append({
                "threshold": th,
                "pf": pf,
                "wr": wr,
                "signals": n_signals,
                "exp": exp
            })
            
        return {
            "symbol": symbol,
            "tf": timeframe,
            "horizon": horizon,
            "metrics": results
        }
        
    except Exception as e:
        logger.error(f"Failed {symbol} {timeframe}: {e}")
        return {"error": str(e)}

def main():
    xgb_params = dict(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=1,
        random_state=42
    )
    
    all_results = []
    
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            for hor in HORIZONS:
                res = run_experiment(sym, tf, hor, xgb_params)
                if "error" not in res:
                    all_results.append(res)
    
    # Analyze and Output Table
    table = Table(title="Top Performance per Symbol (Ordered by Profit Factor)")
    table.add_column("Symbol")
    table.add_column("TF")
    table.add_column("Horiz")
    table.add_column("Thresh")
    table.add_column("PF")
    table.add_column("WinRate")
    table.add_column("Signals")
    table.add_column("Expectancy")
    
    # Selection logic for best per combination
    best_overall = []
    for res in all_results:
        for m in res["metrics"]:
            if m["signals"] >= 10: # Minimum significance
                best_overall.append({
                    "symbol": res["symbol"],
                    "tf": res["tf"],
                    "horizon": res["horizon"],
                    **m
                })
                
    best_overall.sort(key=lambda x: x["pf"], reverse=True)
    
    for b in best_overall[:20]:
        table.add_row(
            b["symbol"],
            b["tf"],
            str(b["horizon"]),
            f"{b['threshold']:.2f}",
            f"{b['pf']:.2f}",
            f"{b['wr']*100:.1f}%",
            str(b["signals"]),
            f"{b['exp']:.6f}"
        )
        
    console.print(table)
    
    # Save raw results
    out_file = _REPO / "logs" / "experiment_suite_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(best_overall, f, indent=2)
    logger.info(f"Full results saved to {out_file}")

if __name__ == "__main__":
    main()
