"""
Exhaustive combination search: threshold x TP x SL x horizon
Uses real path simulation (high/low race) instead of forward-return label WR.
"""
import os, sys, warnings, itertools, time
os.environ.setdefault('DATABASE_URL', 'postgres://x:x@localhost/x')
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
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

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
TP_LIST    = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100]
SL_LIST    = [0.008, 0.010, 0.015, 0.020, 0.025]
HORIZONS   = [4, 6, 8, 12, 16, 20, 24, 36, 48]

SYM_CONFIGS = [
    ('BTC/USDT', '30m', 'models/BTC_USDT_v2.json'),
    ('ETH/USDT', '15m', 'models/ETH_USDT_v2.json'),
    ('XRP/USDT', '15m', 'models/XRP_USDT_v2.json'),
]


def fetch_paginated(sym, tf, n=15000):
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
        time.sleep(0.2)
    df = pd.DataFrame(all_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df.drop_duplicates('ts').set_index('ts').sort_index()


def simulate_tp_sl_path(highs, lows, closes, fire_indices, entry_prices, tp_pct, sl_pct, horizon):
    wins = losses = timeouts = 0
    timeout_pnl = 0.0
    n = len(highs)
    for idx, ep in zip(fire_indices, entry_prices):
        tp_px = ep * (1.0 + tp_pct)
        sl_px = ep * (1.0 - sl_pct)
        result = None
        for j in range(idx + 1, min(idx + horizon + 1, n)):
            tp_hit = highs[j] >= tp_px
            sl_hit = lows[j]  <= sl_px
            if tp_hit and sl_hit:
                result = 'sl'  # conservative: assume SL first
                break
            elif tp_hit:
                result = 'tp'
                break
            elif sl_hit:
                result = 'sl'
                break
        if result == 'tp':
            wins += 1
        elif result == 'sl':
            losses += 1
        else:
            timeouts += 1
            exit_px = closes[min(idx + horizon, n - 1)]
            timeout_pnl += (exit_px - ep) / ep

    total = wins + losses + timeouts
    if total == 0:
        return None
    ev = (wins * tp_pct - losses * sl_pct + timeout_pnl) / total
    wr = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    return {
        'total': total, 'wins': wins, 'losses': losses, 'timeouts': timeouts,
        'wr': wr, 'ev': ev,
    }


def main():
    all_results = []

    for sym, tf, model_path in SYM_CONFIGS:
        print(f'\n=== {sym} ===')
        raw = fetch_paginated(sym, tf, 15000)
        df = add_quant_features(raw)
        df.dropna(subset=FEATURES, inplace=True)

        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        indices = np.arange(len(df))

        m = XGBClassifier()
        m.load_model(model_path)
        X = df[FEATURES].values
        probs = m.predict_proba(X)[:, 1]

        p50 = np.percentile(probs, 50)
        p90 = np.percentile(probs, 90)
        p99 = np.percentile(probs, 99)
        print(f'  {len(df)} candles | p50={p50:.3f}  p90={p90:.3f}  p99={p99:.3f}')

        cpd = {'15m': 96, '30m': 48}[tf]
        count = 0

        for thresh in THRESHOLDS:
            fire_mask = probs >= thresh
            n_fires = int(fire_mask.sum())
            if n_fires < 3:
                continue
            fire_idx = indices[fire_mask]
            entry_prices = closes[fire_mask]

            for tp, sl, hor in itertools.product(TP_LIST, SL_LIST, HORIZONS):
                if tp <= sl:
                    continue  # only R:R >= 1

                res = simulate_tp_sl_path(highs, lows, closes, fire_idx, entry_prices, tp, sl, hor)
                if res is None or res['total'] < 3:
                    continue

                fire_pct = n_fires / len(df)
                trades_per_month = fire_pct * cpd * 30
                monthly_ev = res['ev'] * trades_per_month

                all_results.append({
                    'sym': sym, 'tf': tf,
                    'thresh': thresh,
                    'tp': tp, 'sl': sl, 'horizon': hor,
                    'rr': round(tp / sl, 1),
                    'fires': n_fires,
                    'fire_pct': round(fire_pct * 100, 2),
                    'trades_mo': round(trades_per_month, 1),
                    'wr': round(res['wr'] * 100, 1),
                    'ev_trade': round(res['ev'] * 100, 3),
                    'monthly_ev': round(monthly_ev * 100, 3),
                    'wins': res['wins'],
                    'losses': res['losses'],
                    'timeouts': res['timeouts'],
                })
                count += 1

        print(f'  {count} combos tested')

    print(f'\nTotal combinations evaluated: {len(all_results)}')
    df_res = pd.DataFrame(all_results)

    pos = df_res[df_res['ev_trade'] > 0.0].copy()
    print(f'Combos with EV > 0 per trade: {len(pos)}')

    hdr = f'{"SYM":<10} {"THR":>5} {"TP":>6} {"SL":>6} {"HOR":>4} {"R:R":>4} {"WR":>6} {"EV/tr":>7} {"tr/mo":>6} {"mo_EV%":>8} {"fires":>6}'
    sep = '-' * 90

    if len(pos) > 0:
        pos_freq = pos[pos['trades_mo'] >= 2.0].sort_values('monthly_ev', ascending=False)
        print(f'\n=== TOP 30 by monthly EV (min 2 trades/mo) ===')
        print(hdr); print(sep)
        for _, r in pos_freq.head(30).iterrows():
            print(
                f'{r.sym:<10} {r.thresh:>5.2f} {r.tp*100:>5.1f}% {r.sl*100:>5.1f}% '
                f'{int(r.horizon):>4} {r.rr:>4.1f} {r.wr:>5.1f}% {r.ev_trade:>+6.3f}% '
                f'{r.trades_mo:>6.1f} {r.monthly_ev:>+7.3f}% {int(r.fires):>6}'
            )

        for sym in ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']:
            sub = pos[pos['sym'] == sym].sort_values('monthly_ev', ascending=False)
            print(f'\n--- {sym} best configs ---')
            print(hdr); print(sep)
            for _, r in sub.head(10).iterrows():
                print(
                    f'{r.sym:<10} {r.thresh:>5.2f} {r.tp*100:>5.1f}% {r.sl*100:>5.1f}% '
                    f'{int(r.horizon):>4} {r.rr:>4.1f} {r.wr:>5.1f}% {r.ev_trade:>+6.3f}% '
                    f'{r.trades_mo:>6.1f} {r.monthly_ev:>+7.3f}% {int(r.fires):>6}'
                )
    else:
        print('\nNO combination with positive EV. Showing least-negative:')
        best = df_res[df_res['trades_mo'] >= 2.0].sort_values('ev_trade', ascending=False).head(25)
        print(hdr); print(sep)
        for _, r in best.iterrows():
            print(
                f'{r.sym:<10} {r.thresh:>5.2f} {r.tp*100:>5.1f}% {r.sl*100:>5.1f}% '
                f'{int(r.horizon):>4} {r.rr:>4.1f} {r.wr:>5.1f}% {r.ev_trade:>+6.3f}% '
                f'{r.trades_mo:>6.1f} {r.monthly_ev:>+7.3f}% {int(r.fires):>6}'
            )

    # Save full results
    df_res.to_csv('scripts/combo_search_results.csv', index=False)
    print(f'\nFull results saved to scripts/combo_search_results.csv')


if __name__ == '__main__':
    main()
