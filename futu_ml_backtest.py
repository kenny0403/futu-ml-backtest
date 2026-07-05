#!/home/kenny/.hermes/hermes-agent/venv/bin/python3
"""
Futu ML Backtest 📈
===================
用 Futu OpenD 日K數據 (request_history_kline) 做 ML backtest：
  - SVM(rbf) 為主模型 + RandomForest 輔助 (feature importance)
  - 10-fold Walk-Forward 驗證 (TimeSeriesSplit)
  - 42 個技術因子 + Fundamental pre-filter (PE/PB/ROE/ProfitMargin/MktCap)
  - 每隻股 equity curve chart (dark theme, 標明買入/賣出位置)
  - Cantonese PDF report (NotoSansSC font, 內嵌 chart)
  - JSON results

Output:
  cache/backtest/charts/{stock}_equity.png
  cache/backtest/futu_ml_report.pdf
  cache/backtest/futu_ml_results.json

Usage:
  python futu_ml_backtest.py            # 全部股票
  python futu_ml_backtest.py US.AAPL    # 指定股票
  python futu_ml_backtest.py --no-filter  # 跳過 fundamental pre-filter
"""
import sys, os, json, math, warnings, traceback
warnings.filterwarnings('ignore')

# ─── Redirect stdout BEFORE futu import (避免 futu banner 污染 stdout) ───
_saved_stdout = sys.stdout
sys.stdout = sys.stderr
import logging
logging.basicConfig(stream=sys.stderr, level=logging.ERROR, force=True)

sys.path.insert(0, '/home/kenny/.hermes/hermes-agent/venv/lib/python3.11/site-packages')
from futu import OpenQuoteContext, KLType, AuType, RET_OK, KL_FIELD
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.metrics import accuracy_score

# matplotlib (Agg — headless)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

# reportlab
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, HRFlowable, Image)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

sys.stdout = _saved_stdout

# Fundamental factors module (optional)
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fundamental_factors import get_fundamentals, FUNDAMENTAL_FEATURES
    _HAS_FUNDU = True
except Exception as _e:
    print(f"  ⚠️  fundamental_factors 未可用: {_e}", file=sys.stderr)
    _HAS_FUNDU = False

# ============================================================
# CONFIG
# ============================================================
HKT = timezone(timedelta(hours=8))
HOST = os.environ.get('FUTU_OPEND_HOST') or '172.17.96.1'
PORT = 11111

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'cache', 'backtest')
CHART_DIR = os.path.join(OUTPUT_DIR, 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

LOOKBACK_YEARS = 5
WALK_FORWARD_SPLITS = 10        # 10-fold walk-forward CV
PREDICT_HORIZON = 1             # 預測下一日方向
START_CAPITAL = 100000.0         # equity curve 起始資金 $100K
POSITION_SIZE = 0.5             # 50% 倉位

# Stocks: (futu_code, display_name, type)
STOCKS = [
    ('US.GOOGL', 'GOOGL', '美股'),
    ('US.NVDA',  'NVDA',  '美股'),
    ('US.MSFT',  'MSFT',  '美股'),
    ('US.MU',    'MU',    '美股'),
    ('US.META',  'META',  '美股'),
    ('HK.00388', '0388港交所', '港股'),
    ('HK.00700', '0700騰訊',   '港股'),
    ('HK.03690', '3690美團',   '港股'),
    ('HK.09988', '9988阿里',   '港股'),
    ('HK.07709', '7709湛江',   '港股'),
    ('HK.03317', '3317訊策',   '港股'),
]

# 42 technical features
FEATURE_LIST = [
    'ret_1d','ret_5d','ret_10d','ret_20d',
    'ma5','ma10','ma20','ma50',
    'close_ma5','close_ma10','close_ma20','close_ma50','ma5_ma20','ma10_ma50',
    'ema12','ema26','ema_diff',
    'macd','macd_signal','macd_hist',
    'rsi_14','rsi_7',
    'bb_upper','bb_lower','bb_width','bb_position',
    'atr_14','atr_pct',
    'stoch_k',
    'vol_ma5','vol_ratio_5','vol_ratio_20',
    'obv','obv_ma5','obv_ratio',
    'volatility_5d','volatility_20d',
    'range_pct','body_pct','upper_shadow','lower_shadow',
]

# Fonts
FONT_PATH = '/home/kenny/.hermes/profiles/trading/home/.fonts/NotoSansSC-Regular.ttf'
pdfmetrics.registerFont(TTFont('NotoSansSC', FONT_PATH))
# matplotlib CJK font
try:
    fm.fontManager.addfont(FONT_PATH)
    _CJK_FONT = fm.FontProperties(fname=FONT_PATH)
except Exception:
    _CJK_FONT = None

# Dark theme for matplotlib
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#333355',
    'axes.labelcolor': '#cccccc',
    'text.color': '#ffffff',
    'xtick.color': '#999999',
    'ytick.color': '#999999',
    'grid.color': '#333355',
    'grid.alpha': 0.3,
    'font.size': 10,
    'font.family': 'sans-serif',
})

# reportlab colors
BLUE = HexColor('#1a5276'); GREEN = HexColor('#27ae60'); RED = HexColor('#e74c3c')
LBLUE = HexColor('#d6eaf8'); LGREEN = HexColor('#d5f5e3'); LRED = HexColor('#fadbd8')
ORANGE = HexColor('#e67e22')

# Pre-filter thresholds
PE_MAX = 40
PB_MAX = 15
ROE_MIN = 0.05
HIGH_GROWTH_THRESHOLD = 0.15   # revenue growth > 15% 可以豁免 PE 限制

# ============================================================
# FUTU CONNECTION (single context reused)
# ============================================================
_CTX = None
def get_ctx():
    global _CTX
    if _CTX is None:
        _CTX = OpenQuoteContext(host=HOST, port=PORT)
        print(f"✅ 已連接 Futu OpenD ({HOST}:{PORT})", file=sys.stderr)
    return _CTX

def close_ctx():
    global _CTX
    if _CTX is not None:
        try: _CTX.close()
        except: pass
        _CTX = None

# ============================================================
# DATA: Futu request_history_kline
# ============================================================
def check_quota():
    """Check 歷史K線額度。回傳 (used, remaining, used_codes_list)。"""
    ctx = get_ctx()
    try:
        ret, data = ctx.get_history_kl_quota(get_detail=True)
        if ret != RET_OK or not isinstance(data, tuple) or len(data) < 3:
            print(f"  ⚠️  quota 查詢失敗 ret={ret}", file=sys.stderr)
            return None, None, []
        used, remaining, used_list = data[0], data[1], data[2]
        print(f"  📊 歷史K線額度: 已用 {used} / 剩餘 {remaining}", file=sys.stderr)
        return used, remaining, used_list
    except Exception as e:
        print(f"  ⚠️  quota 查詢例外: {e}", file=sys.stderr)
        return None, None, []

def fetch_kline(code, years=5):
    """Fetch N 年日K (前復權 QFQ)，用年度分頁避開單次回傳上限。"""
    ctx = get_ctx()
    try:
        now = datetime.now(HKT)
        all_chunks = []
        for y in range(years):
            end_dt = now - timedelta(days=365 * y)
            start_dt = end_dt - timedelta(days=400)
            ret, data, _ = ctx.request_history_kline(
                code, start=start_dt.strftime('%Y-%m-%d'),
                end=end_dt.strftime('%Y-%m-%d'),
                ktype=KLType.K_DAY, autype=AuType.QFQ,
                fields=[KL_FIELD.ALL])
            if ret == RET_OK and data is not None and len(data) > 0:
                all_chunks.append(data)
            else:
                print(f"    chunk {start_dt.date()}~{end_dt.date()} 無數據 (ret={ret})", file=sys.stderr)
        if not all_chunks:
            print(f"  ⚠️  {code}: 完全無數據", file=sys.stderr)
            return None
        df = pd.concat(all_chunks).drop_duplicates(subset=['time_key']) \
                                   .sort_values('time_key').reset_index(drop=True)
        # 統一欄名 (全部小寫字串)
        df.columns = [str(c) for c in df.columns]
        print(f"  ✅ {code}: {len(df)} bars ({df['time_key'].iloc[0][:10]} ~ {df['time_key'].iloc[-1][:10]})", file=sys.stderr)
        return df[['time_key', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"  ❌ {code} fetch 例外: {e}", file=sys.stderr)
        return None

# ============================================================
# FEATURE ENGINEERING (42 technical features)
# ============================================================
def compute_features(df):
    if df is None or len(df) < 50:
        return None
    df_c = df.copy()
    close = df_c['close'].values.astype(float)
    high = df_c['high'].values.astype(float)
    low = df_c['low'].values.astype(float)
    open_ = df_c['open'].values.astype(float)
    volume = df_c['volume'].values.astype(float)
    dates = df_c['time_key'].values
    n = len(close)

    def sma(arr, p):
        out = np.full(n, np.nan)
        for i in range(p - 1, n):
            out[i] = np.mean(arr[i - p + 1:i + 1])
        return out

    def ema(arr, p):
        out = np.full(n, np.nan); m = 2 / (p + 1); out[0] = arr[0]
        for i in range(1, n):
            out[i] = (arr[i] - out[i - 1]) * m + out[i - 1]
        return out

    def rolling(arr, w, fn):
        out = np.full(n, np.nan)
        for i in range(w - 1, n):
            out[i] = fn(arr[i - w + 1:i + 1])
        return out

    feat = {}
    # Returns
    feat['ret_1d'] = np.full(n, np.nan); feat['ret_1d'][1:] = close[1:] / close[:-1] - 1
    for p, label in [(4, '5d'), (9, '10d'), (19, '20d')]:
        f = np.full(n, np.nan)
        for i in range(p, n):
            f[i] = close[i] / close[i - p] - 1
        feat[f'ret_{label}'] = f

    # MAs
    ma5 = sma(close, 5); ma10 = sma(close, 10); ma20 = sma(close, 20); ma50 = sma(close, 50)
    feat['ma5'] = ma5; feat['ma10'] = ma10; feat['ma20'] = ma20; feat['ma50'] = ma50
    feat['close_ma5'] = close / ma5 - 1; feat['close_ma10'] = close / ma10 - 1
    feat['close_ma20'] = close / ma20 - 1; feat['close_ma50'] = close / ma50 - 1
    feat['ma5_ma20'] = ma5 / ma20 - 1; feat['ma10_ma50'] = ma10 / ma50 - 1

    # EMA / MACD
    e12 = ema(close, 12); e26 = ema(close, 26)
    feat['ema12'] = e12; feat['ema26'] = e26; feat['ema_diff'] = e12 / e26 - 1
    ml = e12 - e26; sl = ema(ml, 9)
    feat['macd'] = ml; feat['macd_signal'] = sl; feat['macd_hist'] = ml - sl

    # RSI
    for period, label in [(14, '14'), (7, '7')]:
        rsi = np.full(n, np.nan)
        for i in range(period, n):
            g = l = 0
            for j in range(i - period + 1, i + 1):
                ch = close[j] - close[j - 1]
                if ch > 0: g += ch
                else: l += abs(ch)
            if l != 0:
                rsi[i] = 100 - 100 / (1 + (g / period) / (l / period))
            else:
                rsi[i] = 100 if g > 0 else 50
        feat[f'rsi_{label}'] = rsi

    # Bollinger
    bm = sma(close, 20); bs = np.full(n, np.nan)
    for i in range(19, n):
        bs[i] = np.std(close[i - 19:i + 1])
    bu = bm + 2 * bs; bl = bm - 2 * bs
    feat['bb_upper'] = bu; feat['bb_lower'] = bl
    feat['bb_width'] = (bu - bl) / bm; feat['bb_position'] = (close - bl) / (bu - bl)

    # ATR
    atr = np.full(n, np.nan)
    for i in range(14, n):
        trs = [max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
               for j in range(i - 13, i + 1)]
        atr[i] = np.mean(trs)
    feat['atr_14'] = atr; feat['atr_pct'] = atr / close

    # Stoch
    sk = np.full(n, np.nan)
    for i in range(13, n):
        hh = np.max(high[i - 13:i + 1]); ll = np.min(low[i - 13:i + 1])
        sk[i] = (close[i] - ll) / (hh - ll) * 100 if hh - ll > 0 else 50
    feat['stoch_k'] = sk

    # Volume
    vm5 = sma(volume, 5); vm20 = sma(volume, 20)
    feat['vol_ma5'] = vm5; feat['vol_ratio_5'] = volume / vm5; feat['vol_ratio_20'] = volume / vm20

    # OBV
    obv = np.full(n, np.nan); obv[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i - 1]: obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]: obv[i] = obv[i - 1] - volume[i]
        else: obv[i] = obv[i - 1]
    feat['obv'] = obv; feat['obv_ma5'] = sma(obv, 5); feat['obv_ratio'] = obv / sma(obv, 20)

    # Volatility
    ret = feat['ret_1d']
    feat['volatility_5d'] = rolling(abs(ret), 5, np.mean)
    feat['volatility_20d'] = rolling(abs(ret), 20, np.mean)

    # Candle
    feat['range_pct'] = (high - low) / close
    feat['body_pct'] = abs(close - open_) / (high - low)
    feat['upper_shadow'] = (high - np.maximum(open_, close)) / (high - low)
    feat['lower_shadow'] = (np.minimum(open_, close) - low) / (high - low)

    # Target
    target = np.full(n, np.nan)
    target[:-PREDICT_HORIZON] = (close[PREDICT_HORIZON:] > close[:-PREDICT_HORIZON]).astype(int)

    fd = pd.DataFrame(feat)
    fd['target'] = target
    fd['date'] = dates
    fd['_orig_idx'] = np.arange(n)   # 對應原 df 行號
    return fd

# ============================================================
# FUNDAMENTAL PRE-FILTER
# ============================================================
def fundamental_score(pe, pb, roe, profit_margin):
    """計算 0-100 嘅基本面評分。越高越好。"""
    score = 50.0  # baseline
    # PE：合理區間 5-25 加分，過高/負數扣分
    if pe is None or (isinstance(pe, float) and math.isnan(pe)):
        pass
    elif pe <= 0:
        score -= 15   # 虧損
    elif pe <= 10:
        score += 15
    elif pe <= 25:
        score += 10
    elif pe <= 40:
        score += 0
    else:
        score -= 10
    # PB：合理 0.5-3 加分
    if pb is not None and not (isinstance(pb, float) and math.isnan(pb)):
        if 0.5 <= pb <= 3: score += 8
        elif pb > 6: score -= 5
    # ROE：>10% 加分
    if roe is not None and not (isinstance(roe, float) and math.isnan(roe)):
        if roe > 0.15: score += 12
        elif roe > 0.10: score += 8
        elif roe > 0.05: score += 4
        elif roe < 0: score -= 10
    # Profit margin：>10% 加分
    if profit_margin is not None and not (isinstance(profit_margin, float) and math.isnan(profit_margin)):
        if profit_margin > 0.15: score += 10
        elif profit_margin > 0.05: score += 5
        elif profit_margin < 0: score -= 8
    return max(0.0, min(100.0, round(score, 1)))

def fundamental_prefilter(code, fundu_rec):
    """Pre-filter：回傳 (passed, reason, score, details)。"""
    details = {}
    for f in ['pe_ttm', 'pb', 'roe', 'profit_margin', 'revenue_growth_yoy', 'market_cap', 'log_market_cap']:
        details[f] = fundu_rec.get(f)

    pe = fundu_rec.get('pe_ttm')
    pb = fundu_rec.get('pb')
    roe = fundu_rec.get('roe')
    pm = fundu_rec.get('profit_margin')
    rev_growth = fundu_rec.get('revenue_growth_yoy')
    score = fundamental_score(pe, pb, roe, pm)

    # 高增長股豁免 PE 限制
    high_growth = (rev_growth is not None
                   and not (isinstance(rev_growth, float) and math.isnan(rev_growth))
                   and rev_growth > HIGH_GROWTH_THRESHOLD)

    def _bad(v):
        return v is None or (isinstance(v, float) and math.isnan(v))

    # PE > 40 或負數 → 除非高增長
    if not _bad(pe):
        if (pe > PE_MAX or pe <= 0) and not high_growth:
            return False, f"PE={pe:.1f} 過高/虧損 (非高增長)", score, details
    # PB 過高
    if not _bad(pb) and pb > PB_MAX:
        return False, f"PB={pb:.1f} 過高", score, details
    # ROE 太低 (有數據時)
    if not _bad(roe) and roe < 0:
        return False, f"ROE={roe:.2%} 負數", score, details

    return True, "通過基本面篩選", score, details

# ============================================================
# ML BACKTEST  (SVM rbf 為主 + RF 輔助)
# ============================================================
SVM_GRID = list(ParameterGrid({
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf'],
    'probability': [True],
    'max_iter': [5000],
}))

def backtest_stock(code, name, stype, fundu_rec=None, use_filter=True):
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"📊 {name} ({code}) [{stype}]", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    # ── Fundamental pre-filter ──
    passed, filter_reason, fundu_score, fundu_details = True, "未啟用/無數據", None, {}
    if use_filter and fundu_rec:
        passed, filter_reason, fundu_score, fundu_details = fundamental_prefilter(code, fundu_rec)
    if not passed:
        print(f"  🚫 基本面篩選未過: {filter_reason} (score={fundu_score})", file=sys.stderr)
        return {'ticker': code, 'name': name, 'type': stype,
                'filtered': True, 'filter_reason': filter_reason,
                'fundu_score': fundu_score, 'fundu_details': fundu_details,
                'results': []}

    # ── Fetch data ──
    df = fetch_kline(code, LOOKBACK_YEARS)
    if df is None or len(df) < 120:
        print(f"  ⚠️  {name}: 數據不足 ({0 if df is None else len(df)} bars)", file=sys.stderr)
        return None

    feat_df = compute_features(df)
    if feat_df is None:
        return None

    ml_data = feat_df[FEATURE_LIST + ['target', 'date', '_orig_idx']].dropna()
    if len(ml_data) < 120:
        print(f"  ⚠️  {name}: 清洗後數據不足 ({len(ml_data)} rows)", file=sys.stderr)
        return None

    X = np.nan_to_num(ml_data[FEATURE_LIST].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    y = ml_data['target'].values.astype(int)
    dates_arr = ml_data['date'].values
    orig_idx = ml_data['_orig_idx'].values.astype(int)
    closes_full = df['close'].values.astype(float)

    data_start = str(dates_arr[0])[:10]
    data_end = str(dates_arr[-1])[:10]
    total_days = len(ml_data)

    tscv = TimeSeriesSplit(n_splits=WALK_FORWARD_SPLITS)
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)

        best = {'score': -1, 'model': None, 'params': None, 'pred': None}
        for params in SVM_GRID:
            try:
                m = SVC(**params, random_state=42)
                m.fit(Xtr, y_train)
                acc = accuracy_score(y_test, m.predict(Xte))
                if acc > best['score']:
                    best = {'score': acc, 'model': m, 'params': params,
                            'pred': m.predict(Xte)}
            except Exception:
                continue
        if best['model'] is None:
            continue

        pred = best['pred']
        # 每日 trade return
        trade_rets = []
        test_dates = []
        test_close_entry = []
        test_close_exit = []
        for j in range(len(pred)):
            if pred[j] == 1:
                oi = orig_idx[test_idx[j]]
                if oi + PREDICT_HORIZON < len(closes_full):
                    r = closes_full[oi + PREDICT_HORIZON] / closes_full[oi] - 1
                    trade_rets.append(r)
                    test_dates.append(str(dates_arr[test_idx[j]]))
                    test_close_entry.append(closes_full[oi])
                    test_close_exit.append(closes_full[oi + PREDICT_HORIZON])

        wins = sum(1 for r in trade_rets if r > 0)
        trades = len(trade_rets)
        wr = wins / trades if trades > 0 else 0
        baseline = max(y_test.mean(), 1 - y_test.mean())

        fold_results.append({
            'fold': fold_i + 1,
            'acc': best['score'],
            'baseline': baseline,
            'wr': wr,
            'trades': trades,
            'trade_rets': trade_rets,
            'test_dates': test_dates,
            'test_close_entry': test_close_entry,
            'test_close_exit': test_close_exit,
            'params': best['params'],
            'model': best['model'],
            'pred': pred,
        })

    if not fold_results:
        print(f"  ❌ {name}: 無有效 fold", file=sys.stderr)
        return None

    # ── Aggregate ──
    fold_accs = [r['acc'] for r in fold_results]
    mean_acc = float(np.mean(fold_accs))
    mean_baseline = float(np.mean([r['baseline'] for r in fold_results]))

    all_trade_rets = []
    all_trade_dates = []
    all_entry_prices = []
    all_exit_prices = []
    for r in fold_results:
        all_trade_rets.extend(r['trade_rets'])
        all_trade_dates.extend(r['test_dates'])
        all_entry_prices.extend(r['test_close_entry'])
        all_exit_prices.extend(r['test_close_exit'])

    n_trades = len(all_trade_rets)
    win_rets = [r for r in all_trade_rets if r > 0]
    loss_rets = [r for r in all_trade_rets if r < 0]
    n_wins = len(win_rets)
    wr = n_wins / n_trades if n_trades > 0 else 0

    # Compound equity (aligned to trade dates)
    eq_values = [START_CAPITAL]
    eq_dates = []
    buy_dates = []
    buy_equity = []
    sell_dates = []
    sell_equity = []
    running = START_CAPITAL
    for i, r in enumerate(all_trade_rets):
        # BUY on entry date, SELL on exit (=next day, 用同一 trade 收結)
        entry_d = all_trade_dates[i]
        exit_d = _next_day_str(all_trade_dates[i], df)
        running *= (1 + POSITION_SIZE * r)
        # record equity point at exit date
        eq_values.append(running)
        eq_dates.append(exit_d)
        buy_dates.append(entry_d)
        buy_equity.append(running * (1 - POSITION_SIZE * r))  # 入場時嘅 equity (近似)
        sell_dates.append(exit_d)
        sell_equity.append(running)

    total_ret = (eq_values[-1] / START_CAPITAL) - 1 if eq_values else 0
    eq_arr = np.array(eq_values)
    peak = np.maximum.accumulate(eq_arr)
    max_dd = float(np.max((peak - eq_arr) / peak)) if len(eq_arr) > 1 else 0

    avg_win = float(np.mean(win_rets)) if win_rets else 0
    avg_loss = float(np.mean(loss_rets)) if loss_rets else 0
    pf = abs(sum(win_rets) / sum(loss_rets)) if loss_rets and sum(loss_rets) != 0 else float('inf')

    trade_arr = np.array(all_trade_rets) if all_trade_rets else np.array([0.0])
    sharpe = float(np.sqrt(252) * np.mean(trade_arr) / np.std(trade_arr)) if len(trade_arr) > 1 and np.std(trade_arr) > 0 else 0
    neg = trade_arr[trade_arr < 0]
    sortino = float(np.sqrt(252) * np.mean(trade_arr) / np.std(neg)) if len(neg) > 1 and np.std(neg) > 0 else 0
    calmar = float(total_ret / max_dd) if max_dd > 0 else 0

    # Feature importance (fit RF on full data)
    feat_imp = None
    try:
        rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        imp = rf.feature_importances_
        top_idx = np.argsort(imp)[-10:][::-1]
        feat_imp = [(FEATURE_LIST[i], float(imp[i])) for i in top_idx]
    except Exception:
        pass

    result = {
        'model': 'SVM-rbf',
        'params': {k: str(v) for k, v in fold_results[0]['params'].items()},
        'acc': round(mean_acc, 4),
        'acc_std': round(float(np.std(fold_accs)), 4),
        'baseline': round(mean_baseline, 4),
        'improvement': round(mean_acc - mean_baseline, 4),
        'total_return': round(float(total_ret), 4),
        'max_dd': round(max_dd, 4),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'calmar': round(calmar, 2),
        'win_rate': round(float(wr), 4),
        'profit_factor': round(float(pf), 2) if pf != float('inf') else 'inf',
        'avg_win': round(avg_win, 4),
        'avg_loss': round(avg_loss, 4),
        'trades': n_trades,
        'fold_accs': [round(float(a), 3) for a in fold_accs],
        'features': feat_imp,
        'fundu_score': fundu_score,
        'fundu_details': {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                          for k, v in fundu_details.items()},
        'filtered': False,
        'valid': True,
    }

    wr_str = f"{wr:.1%}" if n_trades > 0 else "N/A"
    print(f"  ✅ Acc={mean_acc:.2%}±{np.std(fold_accs):.1%} | WR={wr_str} | Ret={total_ret:+.2%} | "
          f"DD={max_dd:.2%} | Sharpe={sharpe:.2f} | Trades={n_trades}", file=sys.stderr)

    # ── Chart ──
    chart_path = os.path.join(CHART_DIR, f"{name}_equity.png")
    chart_data = {
        'eq_dates': [START_DATE_STR] + eq_dates,  # 加入起始日
        'eq_values': list(eq_values),
        'buy_dates': buy_dates,
        'buy_equity': buy_equity,
        'sell_dates': sell_dates,
        'sell_equity': sell_equity,
    }
    plot_equity_curve(name, stype, chart_data, result, chart_path)

    return {
        'ticker': code, 'name': name, 'type': stype,
        'total_days': total_days,
        'feature_count': len(FEATURE_LIST),
        'data_period': f"{data_start} ~ {data_end}",
        'results': [result],
        'chart_path': chart_path,
        'filtered': False,
        'fundu_score': fundu_score,
    }

START_DATE_STR = ''  # filled per-run

def _next_day_str(date_str, df):
    """用 df 嘅 time_key 搵下一個交易日；搵唔到就用 date_str+1 day 估算。"""
    try:
        tks = df['time_key'].astype(str).values
        idx = np.where(tks == date_str)[0]
        if len(idx) > 0 and idx[0] + 1 < len(tks):
            return tks[idx[0] + 1]
    except Exception:
        pass
    try:
        d = datetime.strptime(date_str[:10], '%Y-%m-%d') + timedelta(days=1)
        return d.strftime('%Y-%m-%d')
    except Exception:
        return date_str

# ============================================================
# EQUITY CURVE CHART (dark theme + buy/sell markers)
# ============================================================
def plot_equity_curve(name, stype, chart_data, result, out_path):
    eq_dates = chart_data['eq_dates']
    eq_values = chart_data['eq_values']
    buy_dates = chart_data['buy_dates']
    buy_equity = chart_data['buy_equity']
    sell_dates = chart_data['sell_dates']
    sell_equity = chart_data['sell_equity']

    if not eq_dates:
        print(f"  ⚠️  {name}: 無交易，跳過 chart", file=sys.stderr)
        return

    dt = pd.to_datetime([d[:10] for d in eq_dates])
    eq = np.array(eq_values, dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.suptitle(f'📈 {name} ({stype}) — SVM-rbf Equity Curve',
                 fontsize=14, fontweight='bold', y=0.98,
                 fontproperties=_CJK_FONT)

    # Equity line + fill
    ax1.fill_between(dt, eq, START_CAPITAL,
                     where=(eq >= START_CAPITAL), color='#26a69a', alpha=0.15, interpolate=True)
    ax1.fill_between(dt, eq, START_CAPITAL,
                     where=(eq < START_CAPITAL), color='#ef5350', alpha=0.15, interpolate=True)
    ax1.plot(dt, eq, color='#4fc3f7', linewidth=2, zorder=3)
    ax1.axhline(y=START_CAPITAL, color='#666666', linestyle='--', linewidth=1, alpha=0.6,
                label=f'起始 ${START_CAPITAL:,.0f}')

    # Buy markers (green UP triangle)
    if buy_dates:
        bdt = pd.to_datetime([d[:10] for d in buy_dates])
        ax1.scatter(bdt, buy_equity, marker='^', color='#26a69a', s=90,
                    zorder=5, edgecolors='#ffffff', linewidths=0.5, label='買入 BUY')
    # Sell markers (red DOWN triangle)
    if sell_dates:
        sdt = pd.to_datetime([d[:10] for d in sell_dates])
        ax1.scatter(sdt, sell_equity, marker='v', color='#ef5350', s=90,
                    zorder=5, edgecolors='#ffffff', linewidths=0.5, label='賣出 SELL')

    # Annotate last value
    last_eq = eq[-1]
    last_d = dt[-1]
    color = '#26a69a' if last_eq >= START_CAPITAL else '#ef5350'
    ax1.annotate(f'${last_eq:,.0f}', xy=(last_d, last_eq),
                 xytext=(10, 8), textcoords='offset points',
                 fontsize=11, fontweight='bold', color=color,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.8, edgecolor=color))

    ax1.set_ylabel('Portfolio Value ($)', fontproperties=_CJK_FONT)
    leg = ax1.legend(loc='upper left', fontsize=8, facecolor='#16213e',
                     edgecolor='#333', labelcolor='white')
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # Metrics box
    pf_str = f"{result['profit_factor']:.2f}" if result['profit_factor'] != 'inf' else '∞'
    metrics_text = (f"回報: {result['total_return']:+.2%}   |   MaxDD: {result['max_dd']:.2%}   |   "
                    f"Sharpe: {result['sharpe']:.2f}   |   WinRate: {result['win_rate']:.1%}   |   "
                    f"交易: {result['trades']}   |   PF: {pf_str}")
    ax1.text(0.02, 0.95, metrics_text, transform=ax1.transAxes, fontsize=8,
             color='white', va='top', fontproperties=_CJK_FONT,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', alpha=0.75))

    # Drawdown panel
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100
    ax2.fill_between(dt, dd, 0, color='#ef5350', alpha=0.3)
    ax2.plot(dt, dd, color='#ef5350', linewidth=1.5)
    ax2.axhline(y=0, color='#666666', linewidth=0.8)
    ax2.set_ylabel('回撤 (%)', fontproperties=_CJK_FONT)
    ax2.set_xlabel('日期', fontproperties=_CJK_FONT)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(max(float(np.max(dd)) * 1.2, 5), 0)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"  📊 Chart: {out_path}", file=sys.stderr)

# ============================================================
# PDF REPORT (reportlab + NotoSansSC + embedded charts)
# ============================================================
styles = getSampleStyleSheet()
sTitle = ParagraphStyle('Title', fontName='NotoSansSC', fontSize=16, leading=20, spaceAfter=6)
sH1 = ParagraphStyle('H1', fontName='NotoSansSC', fontSize=14, leading=18, spaceBefore=12, spaceAfter=6)
sH2 = ParagraphStyle('H2', fontName='NotoSansSC', fontSize=11, leading=14, spaceBefore=8, spaceAfter=4)
sN = ParagraphStyle('N', fontName='NotoSansSC', fontSize=8, leading=11, spaceAfter=2)
sCell = ParagraphStyle('Cell', fontName='NotoSansSC', fontSize=6.5, leading=9)

def _pf(v):
    return f"{v:.2f}" if v != 'inf' and v != float('inf') else '∞'

def generate_pdf(all_results, filtered_stocks, output_path):
    elements = []
    elements.append(Paragraph('📈 Futu ML 機器學習回測報告', sTitle))
    elements.append(HRFlowable(width='100%', thickness=2, color=BLUE))
    elements.append(Spacer(1, 0.5 * cm))

    elements.append(Paragraph(f'報告日期: {datetime.now(HKT).strftime("%Y年%m月%d日 %H:%M")} HKT', sN))
    elements.append(Paragraph(f'數據源: Futu OpenD 日K (前復權 QFQ, {LOOKBACK_YEARS}年)', sN))
    elements.append(Paragraph(f'模型: SVM-rbf 為主 + RandomForest 特徵重要性 | {len(FEATURE_LIST)} 技術因子', sN))
    elements.append(Paragraph(f'驗證: {WALK_FORWARD_SPLITS}折 Walk-Forward (TimeSeriesSplit) | 起始資金 ${START_CAPITAL:,.0f} | {POSITION_SIZE:.0%} 倉位', sN))
    elements.append(Paragraph(f'基本面: PE/PB/ROE/ProfitMargin pre-filter (PE>{PE_MAX} 或負數跳過, 除非高增長)', sN))
    n_passed = sum(1 for r in all_results.values() if not r.get('filtered', False))
    elements.append(Paragraph(f'標的: {len(STOCKS)} 隻美港股 | 通過篩選: {n_passed} | 被篩走: {len(filtered_stocks)}', sN))
    elements.append(Spacer(1, 0.5 * cm))

    # ===== FILTERED STOCKS =====
    if filtered_stocks:
        elements.append(Paragraph('🚫 基本面篩選未過之股票', sH1))
        elements.append(HRFlowable(width='100%', thickness=1, color=BLUE))
        elements.append(Spacer(1, 0.2 * cm))
        data = [['股票', '代碼', '原因', '基本面評分']]
        for r in filtered_stocks:
            data.append([r['name'], r['ticker'],
                         Paragraph(r.get('filter_reason', ''), sCell),
                         f"{r.get('fundu_score')}" if r.get('fundu_score') is not None else '—'])
        t = Table(data, colWidths=[3 * cm, 2.5 * cm, 8 * cm, 2 * cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, -1), 'NotoSansSC'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.3, grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LRED]),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.4 * cm))

    # ===== GLOBAL RANKING =====
    elements.append(Paragraph('🏆 綜合排名 — 通過篩選之股票', sH1))
    elements.append(HRFlowable(width='100%', thickness=1, color=BLUE))
    elements.append(Spacer(1, 0.2 * cm))

    valid = [r for r in all_results.values() if r.get('results') and not r.get('filtered', False)]
    valid_sorted = sorted(valid, key=lambda x: x['results'][0]['sharpe'], reverse=True)
    data = [['排名', '股票', '類型', '準確率', '總回報', 'MaxDD', 'Sharpe', 'WinRate', '交易', 'ProfitF', '基本面']]
    for i, res in enumerate(valid_sorted):
        r = res['results'][0]
        data.append([
            str(i + 1), res['name'], res['type'],
            f"{r['acc']:.2%}", f"{r['total_return']:+.2%}", f"{r['max_dd']:.2%}",
            f"{r['sharpe']:.2f}", f"{r['win_rate']:.1%}", str(r['trades']),
            _pf(r['profit_factor']),
            f"{r.get('fundu_score')}" if r.get('fundu_score') is not None else '—',
        ])
    t = Table(data, colWidths=[1.2*cm, 2.8*cm, 1.3*cm, 1.6*cm, 1.7*cm, 1.4*cm, 1.4*cm, 1.4*cm, 1.2*cm, 1.4*cm, 1.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, -1), 'NotoSansSC'),
        ('FONTSIZE', (0, 0), (-1, -1), 6.5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.3, grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LBLUE]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.4 * cm))

    # ===== PER-STOCK DETAILS + CHART =====
    elements.append(PageBreak())
    elements.append(Paragraph('📊 逐股詳細分析 + Equity Curve', sH1))
    elements.append(HRFlowable(width='100%', thickness=1, color=BLUE))
    elements.append(Spacer(1, 0.3 * cm))

    for res in valid_sorted:
        r = res['results'][0]
        elements.append(Paragraph(f"{res['name']} ({res['ticker']}) [{res['type']}]", sH2))
        elements.append(Paragraph(
            f"📅 {res['data_period']} | {res['total_days']} bars | {len(FEATURE_LIST)}因子 | 基本面評分: "
            f"{r.get('fundu_score') if r.get('fundu_score') is not None else '—'}", sN))

        # Metrics table
        mdata = [
            ['準確率', 'AccStd', 'Baseline', '總回報', 'MaxDD', 'Sharpe', 'Sortino', 'Calmar',
             'WinRate', 'ProfitF', 'AvgWin', 'AvgLoss', '交易數'],
            [f"{r['acc']:.2%}", f"{r['acc_std']:.2%}", f"{r['baseline']:.2%}",
             f"{r['total_return']:+.2%}", f"{r['max_dd']:.2%}", f"{r['sharpe']:.2f}",
             f"{r['sortino']:.2f}", f"{r['calmar']:.2f}", f"{r['win_rate']:.1%}",
             _pf(r['profit_factor']), f"{r['avg_win']:.2%}", f"{r['avg_loss']:.2%}", str(r['trades'])],
        ]
        t = Table(mdata, colWidths=[1.5*cm]*13)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, -1), 'NotoSansSC'),
            ('FONTSIZE', (0, 0), (-1, -1), 5.5),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.3, grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, 1), [LBLUE]),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.15 * cm))

        # Fold accuracies
        elements.append(Paragraph(
            f"Fold 準確率: {', '.join(f'{a:.1%}' for a in r['fold_accs'])} | "
            f"參數: {json.dumps(r['params'], ensure_ascii=False)}", sN))

        # Fundamental details
        fd = r.get('fundu_details') or {}
        if fd:
            fd_str = ' | '.join(f"{k}={v}" for k, v in fd.items() if v is not None)
            elements.append(Paragraph(f"基本面: {fd_str}", sN))

        # Feature importance
        if r.get('features'):
            feat_str = ' | '.join(f"{f}:{v:.3f}" for f, v in r['features'][:8])
            elements.append(Paragraph(f"🔑 特徵重要性 (RF): {feat_str}", sN))

        # Embed chart
        chart_path = res.get('chart_path')
        if chart_path and os.path.exists(chart_path):
            try:
                img = Image(chart_path, width=17 * cm, height=9.7 * cm)
                elements.append(Spacer(1, 0.15 * cm))
                elements.append(img)
            except Exception as e:
                elements.append(Paragraph(f"(chart 載入失敗: {e})", sN))
        else:
            elements.append(Paragraph("(無 chart)", sN))

        elements.append(Spacer(1, 0.5 * cm))

    # ===== FOOTER NOTE =====
    elements.append(Paragraph(
        '⚠️ 免責聲明: 過往回測表現不代表未來結果。模型存在過擬合風險。'
        f'{POSITION_SIZE:.0%} 倉位、未計交易成本（手續費/滑點）。'
        '基本面數據為最新 snapshot 並非歷史時序，pre-filter 為啟發式規則。', sN))

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=1.2 * cm, rightMargin=1.2 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.2 * cm)
    doc.build(elements)
    print(f"\n✅ PDF: {output_path}", file=sys.stderr)

# ============================================================
# MAIN
# ============================================================
def main():
    global START_DATE_STR
    args = sys.argv[1:]
    use_filter = True
    selected_codes = []
    for a in args:
        if a == '--no-filter':
            use_filter = False
        else:
            selected_codes.append(a)

    stock_list = STOCKS
    if selected_codes:
        stock_list = [s for s in STOCKS if s[0] in selected_codes]
        if not stock_list:
            print(f"⚠️  指定嘅股票 {selected_codes} 唔喺預設列表，自動加入")
            for c in selected_codes:
                stock_list.append((c, c, '美股' if c.startswith('US.') else '港股'))

    print(f"\n🚀 Futu ML Backtest 啟動", file=sys.stderr)
    print(f"   股票數: {len(stock_list)} | Lookback: {LOOKBACK_YEARS}年 | "
          f"WF splits: {WALK_FORWARD_SPLITS} | Pre-filter: {use_filter}", file=sys.stderr)

    # Check quota
    check_quota()

    # Fetch fundamentals (if available)
    fundu_data = {}
    if _HAS_FUNDU and use_filter:
        codes = [s[0] for s in stock_list]
        try:
            print(f"\n📡 拎基本面數據 ({len(codes)} 隻)...", file=sys.stderr)
            fundu_data = get_fundamentals(codes, use_futu=True, force_refresh=False,
                                         financials_codes=codes)
            print(f"   ✅ 基本面: {len(fundu_data)} 隻", file=sys.stderr)
        except Exception as e:
            print(f"   ⚠️  基本面拎取失敗: {e} — 會跳過 pre-filter", file=sys.stderr)
            use_filter = False

    START_DATE_STR = (datetime.now(HKT) - timedelta(days=365 * LOOKBACK_YEARS)).strftime('%Y-%m-%d')

    all_results = {}
    filtered_stocks = []

    for code, name, stype in stock_list:
        try:
            fundu_rec = fundu_data.get(code, {})
            res = backtest_stock(code, name, stype, fundu_rec, use_filter)
            if res is None:
                continue
            if res.get('filtered'):
                filtered_stocks.append(res)
            else:
                all_results[name] = res
        except Exception as e:
            print(f"❌ {name} 當機: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    close_ctx()

    # ── Summary ──
    print(f"\n\n{'='*100}", file=sys.stderr)
    print("📈  FINAL SUMMARY — Futu ML Backtest", file=sys.stderr)
    print('=' * 100, file=sys.stderr)

    if filtered_stocks:
        print(f"\n🚫 被基本面篩走 ({len(filtered_stocks)}):", file=sys.stderr)
        for r in filtered_stocks:
            print(f"   {r['name']:<14s} {r['ticker']:<10s} {r.get('filter_reason','')}", file=sys.stderr)

    for name, res in all_results.items():
        r = res['results'][0]
        print(f"\n{name} ({res['ticker']}):", file=sys.stderr)
        print(f"   🥇 SVM-rbf | Acc={r['acc']:.2%}±{r['acc_std']:.2%} | Sharpe={r['sharpe']:.2f} | "
              f"Ret={r['total_return']:+.2%} | DD={r['max_dd']:.2%} | WR={r['win_rate']:.1%} | "
              f"Trades={r['trades']} | FundScore={r.get('fundu_score')}", file=sys.stderr)

    # ── Generate PDF ──
    pdf_path = os.path.join(OUTPUT_DIR, 'futu_ml_report.pdf')
    generate_pdf(all_results, filtered_stocks, pdf_path)

    # ── Save JSON ──
    json_path = os.path.join(OUTPUT_DIR, 'futu_ml_results.json')
    with open(json_path, 'w') as f:
        out = {
            'generated': datetime.now(HKT).strftime('%Y-%m-%d %H:%M:%S HKT'),
            'config': {
                'lookback_years': LOOKBACK_YEARS,
                'walk_forward_splits': WALK_FORWARD_SPLITS,
                'predict_horizon': PREDICT_HORIZON,
                'start_capital': START_CAPITAL,
                'position_size': POSITION_SIZE,
                'model': 'SVM-rbf',
                'feature_count': len(FEATURE_LIST),
                'prefilter_enabled': use_filter,
            },
            'stocks': [{
                'name': v['name'], 'ticker': v['ticker'], 'type': v['type'],
                'data_period': v.get('data_period'), 'total_days': v.get('total_days'),
                'chart': v.get('chart_path'),
                'results': v.get('results', []),
            } for v in all_results.values()],
            'filtered': [{
                'name': r['name'], 'ticker': r['ticker'],
                'filter_reason': r.get('filter_reason'),
                'fundu_score': r.get('fundu_score'),
                'fundu_details': r.get('fundu_details'),
            } for r in filtered_stocks],
        }
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ JSON: {json_path}", file=sys.stderr)

    # stdout 俾 parent agent 用
    print(json.dumps({
        'pdf': pdf_path,
        'json': json_path,
        'charts': [v.get('chart_path') for v in all_results.values()],
        'n_passed': len(all_results),
        'n_filtered': len(filtered_stocks),
    }, ensure_ascii=False))

if __name__ == '__main__':
    main()
