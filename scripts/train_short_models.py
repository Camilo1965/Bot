"""
Train SHORT direction models for ETH/USDT and XRP/USDT.

Label: forward_return < -LABEL_THRESHOLD (price falls more than 1.5% in horizon candles).
SHORT model fires when price is BELOW SMA200 1h (downtrend confirmation).

Symmetric to the LONG model training but with inverted label.
Saves as: models/ETH_USDT_short_v2.json, models/XRP_USDT_short_v2.json
"""
import os, sys, time, json, warnings
os.environ.setdefault('DATABASE_URL', 'postgres://x:x@localhost/x')
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
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
    'ETH/USDT': {'tf': '15m', 'horizon': 8,  'n_candles': 50000, 'label_threshold': 0.015, 'max_spw': 8.0, 'model_out': 'models/ETH_USDT_short_v2.json'},
    'XRP/USDT': {'tf': '15m', 'horizon': 8,  'n_candles': 50000, 'label_threshold': 0.015, 'max_spw': 8.0, 'model_out': 'models/XRP_USDT_short_v2.json'},
}


def fetch_paginated(sym, tf, n=50000):
    tf_ms = {'15m': 15*60*1000, '30m': 30*60*1000}[tf]
    since = exchange.milliseconds() - n * tf_ms
    all_ohlcv = []
    while len(all_ohlcv) < n:
        chunk = exchange.fetch_ohlcv(sym, tf, since=since, limit=1000)
        if not chunk: break
        all_ohlcv.extend(chunk)
        since = chunk[-1][0] + tf_ms
        if len(chunk) < 1000: break
        time.sleep(0.15)
    df = pd.DataFrame(all_ohlcv, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df.drop_duplicates('ts').set_index('ts').sort_index()


def make_short_label(close_series, horizon, threshold):
    """Label=1 if forward return < -threshold (price drops more than threshold in horizon bars)."""
    fwd = close_series.shift(-horizon) / close_series - 1.0
    return (fwd < -threshold).astype(int)


def train_symbol(sym, cfg):
    print(f'\n{"="*60}')
    print(f'SHORT model: {sym}  horizon={cfg["horizon"]}  threshold={cfg["label_threshold"]*100:.1f}%')
    print(f'{"="*60}')

    print('Fetching data...')
    raw = fetch_paginated(sym, cfg['tf'], cfg['n_candles'])
    print(f'  {len(raw)} candles ({raw.index[0].date()} to {raw.index[-1].date()})')

    print('Computing features...')
    df = add_quant_features(raw)
    df.dropna(subset=FEATURES, inplace=True)

    print('Computing SHORT labels...')
    df['label'] = make_short_label(df['close'], cfg['horizon'], cfg['label_threshold'])
    df.dropna(subset=['label'], inplace=True)

    # Exclude last horizon rows (no valid forward return)
    df = df.iloc[:-cfg['horizon']].copy()

    n_pos = int((df['label'] == 1).sum())
    n_neg = int((df['label'] == 0).sum())
    pos_rate = n_pos / len(df)
    print(f'  {len(df)} rows | pos(SHORT)={n_pos} ({pos_rate:.2%}) | neg={n_neg}')

    if n_pos < 100:
        print(f'ERROR: only {n_pos} short positives. Reduce threshold or horizon.')
        return None

    X = df[FEATURES].values
    y = df['label'].values

    raw_spw = n_neg / n_pos
    spw = min(raw_spw, cfg['max_spw'])
    print(f'  SPW: {raw_spw:.1f}x capped at {spw:.1f}x')

    # CV
    tscv = TimeSeriesSplit(n_splits=5)
    cv_aucs = []
    print('  CV folds:')
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        if y_va.sum() < 5: continue
        n_pos_f = int(y_tr.sum())
        n_neg_f = int((y_tr == 0).sum())
        spw_f = min(n_neg_f / max(n_pos_f, 1), cfg['max_spw'])
        m = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw_f,
                          eval_metric='auc', early_stopping_rounds=30, verbosity=0, random_state=42)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        auc = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])
        cv_aucs.append(auc)
        print(f'    fold {fold+1}: AUC={auc:.4f}  pos_va={int(y_va.sum())}')

    cv_mean = float(np.mean(cv_aucs)) if cv_aucs else 0.0
    print(f'  CV AUC mean: {cv_mean:.4f}')

    # Final model
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_ho, y_ho = X[split:], y[split:]

    n_pos_tr = int(y_tr.sum())
    n_neg_tr = int((y_tr == 0).sum())
    spw_final = min(n_neg_tr / max(n_pos_tr, 1), cfg['max_spw'])

    model = XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw_final,
                          eval_metric='auc', early_stopping_rounds=30, verbosity=0, random_state=42)
    model.fit(X_tr, y_tr, eval_set=[(X_ho, y_ho)], verbose=False)

    probs_ho = model.predict_proba(X_ho)[:, 1]
    auc_ho = roc_auc_score(y_ho, probs_ho) if y_ho.sum() > 0 else 0.0

    print(f'\n  Holdout AUC: {auc_ho:.4f}')
    print(f'  Prob distribution:')
    for p in [50, 75, 90, 95, 99]:
        print(f'    p{p} = {np.percentile(probs_ho, p):.4f}')

    # Threshold sweep
    print(f'\n  Threshold sweep (SHORT: down {cfg["label_threshold"]*100:.1f}% in {cfg["horizon"]} bars):')
    print(f'  {"thresh":>7} {"fires":>6} {"fire%":>6} {"WR":>6} {"EV":>8}')
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        mask = probs_ho >= t
        nf = mask.sum()
        if nf < 3: continue
        wr = y_ho[mask].mean()
        tp, sl = cfg['label_threshold'] * 2.5, cfg['label_threshold']  # same as LONG config
        ev = wr * tp - (1 - wr) * sl
        print(f'  {t:>7.2f} {nf:>6} {mask.mean()*100:>5.1f}% {wr*100:>5.1f}% {ev*100:>+7.3f}%')

    model.save_model(cfg['model_out'])

    fi = dict(zip(FEATURES, model.feature_importances_.tolist()))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    meta = {
        'symbol': sym,
        'direction': 'short',
        'timeframe': cfg['tf'],
        'horizon': cfg['horizon'],
        'label': 'short_forward_return',
        'label_threshold': cfg['label_threshold'],
        'max_spw': cfg['max_spw'],
        'actual_spw': round(float(spw_final), 4),
        'rows_total': int(len(df)),
        'pos_rate': round(float(pos_rate), 6),
        'cv_auc_mean': round(cv_mean, 6),
        'holdout_auc': round(auc_ho, 6),
        'features': FEATURES,
        'feature_importance': {k: round(float(v), 6) for k, v in fi_sorted.items()},
    }
    meta_path = cfg['model_out'].replace('.json', '.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n  Saved: {cfg["model_out"]}')
    print(f'  Top 5 features: {list(fi_sorted.keys())[:5]}')
    return auc_ho, cv_mean


def main():
    print('Training SHORT direction models\n')
    for sym, cfg in TRAIN_CONFIG.items():
        try:
            train_symbol(sym, cfg)
        except Exception as e:
            import traceback
            print(f'ERROR {sym}: {e}')
            traceback.print_exc()


if __name__ == '__main__':
    main()
