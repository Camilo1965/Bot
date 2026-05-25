"""
Retrain direction models using TP/SL-race binary label.

Label = 1  if price hits entry*(1+tp_pct) before entry*(1-sl_pct) within horizon candles.
Label = 0  if SL hit first, both hit same candle (conservative), or timeout.

This aligns training objective with actual trading outcome — unlike forward_return
which ignores whether SL was hit on the way to TP.
"""
import os, sys, time, json, warnings
os.environ.setdefault('DATABASE_URL', 'postgres://x:x@localhost/x')
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from strategy.quant_features import add_quant_features
import ccxt

exchange = ccxt.binance({'enableRateLimit': True})

FEATURES = [
    'rsi', 'macd_line', 'macd_signal', 'macd_hist', 'atr', 'bb_pct_b', 'bb_width',
    'vol_delta_norm', 'log_ret_1', 'log_ret_5', 'log_ret_10', 'log_ret_20',
    'close_vs_sma200_1h', 'vol_rel', 'adx', 'stoch_rsi_k', 'stoch_rsi_d',
    'williams_r', 'obv', 'cmf', 'ret_skew_20', 'ret_kurt_20', 'roll_sharpe_20',
    'hour_sin', 'hour_cos', 'dow',
]

TRAIN_CONFIG = {
    'ETH/USDT': {
        'tf': '15m',
        'tp_pct': 0.075,   # 7.5% TP — optimal from exhaustive search
        'sl_pct': 0.020,   # 2.0% SL
        'horizon': 36,     # 9h hold window
        'n_candles': 50000,
        'model_out': 'models/ETH_USDT_v3.json',
        'max_spw': 8.0,
    },
    'XRP/USDT': {
        'tf': '15m',
        'tp_pct': 0.100,   # 10% TP — optimal from exhaustive search
        'sl_pct': 0.015,   # 1.5% SL
        'horizon': 36,     # 9h hold window
        'n_candles': 50000,
        'model_out': 'models/XRP_USDT_v3.json',
        'max_spw': 8.0,
    },
}


def fetch_paginated(sym, tf, n=50000):
    tf_ms = {'15m': 15 * 60 * 1000, '30m': 30 * 60 * 1000}[tf]
    since = exchange.milliseconds() - n * tf_ms
    all_ohlcv = []
    while len(all_ohlcv) < n:
        chunk = exchange.fetch_ohlcv(sym, tf, since=since, limit=1000)
        if not chunk:
            break
        all_ohlcv.extend(chunk)
        since = chunk[-1][0] + tf_ms
        if len(chunk) < 1000:
            break
        time.sleep(0.15)
    df = pd.DataFrame(all_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df.drop_duplicates('ts').set_index('ts').sort_index()


def make_tpsl_label(df, tp_pct, sl_pct, horizon):
    """
    For each candle i: entry = close[i].
    Scan next `horizon` candles. First to be breached wins.
    Returns array of 0/1 with -1 for last `horizon` candles (invalid).
    """
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    n = len(closes)
    labels = np.full(n, -1, dtype=np.int8)

    for i in range(n - horizon):
        ep = closes[i]
        tp_px = ep * (1.0 + tp_pct)
        sl_px = ep * (1.0 - sl_pct)
        result = 0  # default: timeout = loss
        for j in range(i + 1, i + horizon + 1):
            th, tl = highs[j], lows[j]
            if th >= tp_px and tl <= sl_px:
                result = 0  # both same candle: SL first (conservative)
                break
            elif th >= tp_px:
                result = 1  # TP hit
                break
            elif tl <= sl_px:
                result = 0  # SL hit
                break
        labels[i] = result

    return labels


def train_symbol(sym, cfg):
    print(f'\n{"="*60}')
    print(f'Symbol: {sym}  TP={cfg["tp_pct"]*100:.1f}%  SL={cfg["sl_pct"]*100:.1f}%  horizon={cfg["horizon"]}')
    print(f'{"="*60}')

    print('Fetching data...')
    raw = fetch_paginated(sym, cfg['tf'], cfg['n_candles'])
    print(f'  Raw candles: {len(raw)}  ({raw.index[0].date()} to {raw.index[-1].date()})')

    print('Computing features...')
    df = add_quant_features(raw)
    df.dropna(subset=FEATURES, inplace=True)
    print(f'  After feature dropna: {len(df)} candles')

    print('Computing TP/SL-race labels...')
    t0 = time.time()
    labels = make_tpsl_label(df, cfg['tp_pct'], cfg['sl_pct'], cfg['horizon'])
    print(f'  Done in {time.time()-t0:.1f}s')

    # Attach labels, drop invalid rows
    df = df.copy()
    df['label'] = labels
    df = df[df['label'] >= 0].copy()

    n_pos = int((df['label'] == 1).sum())
    n_neg = int((df['label'] == 0).sum())
    pos_rate = n_pos / len(df)
    print(f'  Labeled: {len(df)} candles | pos={n_pos} ({pos_rate:.2%}) | neg={n_neg}')

    if n_pos < 50:
        print('ERROR: Too few positives (<50). Check TP/SL/horizon config.')
        return

    X = df[FEATURES].values
    y = df['label'].values

    # Scale pos weight (capped)
    raw_spw = n_neg / n_pos
    spw = min(raw_spw, cfg['max_spw'])
    print(f'  Raw SPW={raw_spw:.1f}x  capped at {spw:.1f}x')

    # TimeSeriesSplit CV
    tscv = TimeSeriesSplit(n_splits=5)
    cv_aucs = []
    print('  CV folds:')
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        if y_va.sum() < 5:
            continue
        m = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric='auc',
            early_stopping_rounds=30,
            verbosity=0,
            random_state=42,
        )
        n_pos_fold = int(y_va.sum())
        n_neg_fold = int((y_va == 0).sum())
        spw_fold = min(n_neg_fold / max(n_pos_fold, 1), cfg['max_spw'])
        m.set_params(scale_pos_weight=spw_fold)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        probs_va = m.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, probs_va)
        cv_aucs.append(auc)
        print(f'    fold {fold+1}: AUC={auc:.4f}  pos_va={n_pos_fold}')

    cv_mean = float(np.mean(cv_aucs)) if cv_aucs else 0.0
    print(f'  CV AUC mean: {cv_mean:.4f}')

    # Final model on 80% train, 20% holdout
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_ho, y_ho = X[split:], y[split:]

    n_pos_tr = int(y_tr.sum())
    n_neg_tr = int((y_tr == 0).sum())
    spw_final = min(n_neg_tr / max(n_pos_tr, 1), cfg['max_spw'])

    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw_final,
        eval_metric='auc',
        early_stopping_rounds=30,
        verbosity=0,
        random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_ho, y_ho)], verbose=False)

    probs_ho = model.predict_proba(X_ho)[:, 1]
    auc_ho = roc_auc_score(y_ho, probs_ho) if y_ho.sum() > 0 else 0.0

    print(f'\n  Holdout AUC: {auc_ho:.4f}')
    print(f'  Prob distribution on holdout:')
    for p in [50, 75, 90, 95, 99]:
        print(f'    p{p} = {np.percentile(probs_ho, p):.4f}')

    # Threshold sweep on holdout
    print(f'\n  Threshold sweep (TP={cfg["tp_pct"]*100:.0f}% SL={cfg["sl_pct"]*100:.1f}%):')
    print(f'  {"thresh":>7} {"fires":>6} {"fire%":>6} {"WR":>6} {"EV":>8}')
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        mask = probs_ho >= t
        nf = mask.sum()
        if nf < 3:
            continue
        wr = y_ho[mask].mean()
        ev = wr * cfg['tp_pct'] - (1 - wr) * cfg['sl_pct']
        print(f'  {t:>7.2f} {nf:>6} {mask.mean()*100:>5.1f}% {wr*100:>5.1f}% {ev*100:>+7.3f}%')

    # Save model
    model.save_model(cfg['model_out'])

    # Feature importance
    fi = dict(zip(FEATURES, model.feature_importances_.tolist()))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    meta = {
        'symbol': sym,
        'timeframe': cfg['tf'],
        'label': 'tpsl_race',
        'tp_pct': cfg['tp_pct'],
        'sl_pct': cfg['sl_pct'],
        'horizon': cfg['horizon'],
        'max_spw': cfg['max_spw'],
        'actual_spw': round(float(spw_final), 4),
        'rows_total': int(len(df)),
        'rows_train': int(len(X_tr)),
        'rows_holdout': int(len(X_ho)),
        'train_pos': int(n_pos_tr),
        'train_neg': int(n_neg_tr),
        'pos_rate': round(float(pos_rate), 6),
        'cv_auc_mean': round(float(cv_mean), 6),
        'holdout_auc': round(float(auc_ho), 6),
        'feature_importance': {k: round(float(v), 6) for k, v in fi_sorted.items()},
        'features': FEATURES,
    }
    meta_path = cfg['model_out'].replace('.json', '.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n  Model saved: {cfg["model_out"]}')
    print(f'  Meta saved : {meta_path}')
    print(f'\n  Top 5 features: {list(fi_sorted.keys())[:5]}')

    return auc_ho, cv_mean


def main():
    print('Retrain with TP/SL-race label\n')
    results = {}
    for sym, cfg in TRAIN_CONFIG.items():
        try:
            r = train_symbol(sym, cfg)
            if r:
                results[sym] = r
        except Exception as e:
            import traceback
            print(f'\nERROR {sym}: {e}')
            traceback.print_exc()

    print('\n\n=== SUMMARY ===')
    for sym, (auc_ho, cv_mean) in results.items():
        print(f'{sym}: CV_AUC={cv_mean:.4f}  Holdout_AUC={auc_ho:.4f}')

    print('\nNext step: update SYMBOL_CONFIG prob_threshold based on threshold sweep above.')
    print('Then update model_json_path_for_symbol() to load v3 models.')


if __name__ == '__main__':
    main()
