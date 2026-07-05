#!/home/kenny/.hermes/hermes-agent/venv/bin/python3
"""Generate HTML Dashboard from futu_ml_results.json + charts/"""
import json, os, base64, sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
JSON_PATH = BASE_DIR / 'cache' / 'backtest' / 'futu_ml_results.json'
CHARTS_DIR = BASE_DIR / 'cache' / 'backtest' / 'charts'
OUTPUT = BASE_DIR / 'futu_ml_dashboard.html'

# Fallback — also check repo root
if not JSON_PATH.exists():
    JSON_PATH = BASE_DIR / 'futu_ml_results.json'
if not CHARTS_DIR.exists():
    CHARTS_DIR = BASE_DIR / 'charts'

def embed_png(path):
    """Read PNG and return base64 data URI."""
    p = Path(path)
    if not p.exists():
        # try charts/ dir
        p = CHARTS_DIR / p.name
    if p.exists():
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f"data:image/png;base64,{b64}"
    return ""

def fmt_pct(v):
    if v is None: return '—'
    return f"{v:+.2%}" if isinstance(v, float) else str(v)

def fmt_num(v, d=2):
    if v is None: return '—'
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)

def clr_val(v, green_positive=True):
    """Return CSS class for numeric value coloring."""
    if v is None: return ''
    if isinstance(v, (int, float)):
        if v > 0: return 'green' if green_positive else 'red'
        if v < 0: return 'red' if green_positive else 'green'
    return ''

def fund_score_label(score):
    if score is None: return '⚪ 無數據', 'neutral'
    if score >= 70: return f'🟢 {score:.0f}', 'green'
    if score >= 50: return f'🟡 {score:.0f}', 'yellow'
    return f'🔴 {score:.0f}', 'red'

def generate():
    with open(JSON_PATH) as f:
        data = json.load(f)

    stocks = data.get('stocks', [])
    filtered = data.get('filtered', [])
    config = data.get('config', {})
    gen_time = data.get('generated', datetime.now().strftime('%Y-%m-%d %H:%M'))

    # Sort by fundamental score desc, then Sharpe desc
    def sort_key(s):
        r = s.get('results', [{}])[0]
        fs = s.get('fundamental_score') or r.get('fundu_score') or 0
        sharpe = r.get('sharpe', 0) or 0
        return (-(fs or 0), -(sharpe or 0))
    stocks.sort(key=sort_key)

    # ─── BUILD HTML ───
    lines = []
    def w(s): lines.append(s)

    w('<!DOCTYPE html>')
    w('<html lang="zh-HK">')
    w('<head>')
    w('<meta charset="UTF-8">')
    w('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    w(f'<title>Futu ML Backtest Dashboard — {gen_time[:10]}</title>')
    w('<style>')
    w("""
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; background:#0b0e1a; color:#e0e0e0; }
.container { max-width:1400px; margin:0 auto; padding:20px; }
h1 { font-size:28px; color:#4fc3f7; margin-bottom:5px; }
h2 { font-size:20px; color:#81d4fa; margin:20px 0 10px 0; padding-bottom:6px; border-bottom:2px solid #1a237e; }
.header { background:linear-gradient(135deg,#0d1b2a,#1b2838); padding:20px; border-radius:12px; margin-bottom:20px; border:1px solid #1a237e; }
.header-meta { font-size:13px; color:#78909c; margin:8px 0; line-height:1.6; }
.nav { display:flex; flex-wrap:wrap; gap:6px; margin:12px 0; }
.nav-link { padding:5px 12px; background:#1a237e; color:#90caf9; text-decoration:none; border-radius:20px; font-size:11px; transition:0.2s; }
.nav-link:hover { background:#283593; color:#fff; transform:translateY(-1px); }
.summary-table { width:100%; border-collapse:collapse; margin:10px 0 20px; background:#111827; border-radius:10px; overflow:hidden; }
.summary-table th { background:#1a237e; color:#fff; padding:10px 12px; font-size:13px; text-align:left; }
.summary-table td { padding:8px 12px; font-size:13px; border-bottom:1px solid #1e293b; }
.summary-table tr:hover td { background:#1e293b; }
.stock-link { color:#4fc3f7; text-decoration:none; font-weight:600; }
.stock-link:hover { text-decoration:underline; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.green { color:#26a69a !important; font-weight:600; }
.red { color:#ef5350 !important; font-weight:600; }
.yellow { color:#fdd835 !important; }
.stock-card { background:linear-gradient(135deg,#111827,#1a1a2e); border:1px solid #1e293b; border-radius:12px; padding:20px; margin:16px 0; }
.stock-header { margin-bottom:16px; }
.stock-name { font-size:20px; font-weight:700; color:#e0e0e0; }
.stock-ticker { font-size:12px; color:#78909c; background:#1e293b; padding:2px 8px; border-radius:4px; margin-left:6px; }
.stock-type { font-size:11px; color:#78909c; background:#0d2137; padding:2px 8px; border-radius:4px; margin-left:4px; }
.stock-period { font-size:12px; color:#546e7a; margin-top:4px; }
.stock-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media (max-width:900px) { .stock-grid { grid-template-columns:1fr; } }
.panel { background:#0d1b2a; border:1px solid #1a237e; border-radius:10px; padding:14px; }
.feat-panel { margin-top:12px; }
.panel-title { font-size:14px; font-weight:600; color:#81d4fa; margin-bottom:10px; padding-bottom:6px; border-bottom:1px solid #1a237e; }
.metric-table { width:100%; border-collapse:collapse; }
.metric-table td { padding:5px 8px; font-size:12px; }
.ml-label { color:#78909c; width:30%; }
.ml-val { color:#e0e0e0; font-weight:600; text-align:right; }
.metrics-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; }
.metric-card { background:#111827; border:1px solid #1e293b; border-radius:8px; padding:10px; text-align:center; }
.metric-value { font-size:18px; font-weight:700; color:#4fc3f7; }
.metric-label { font-size:11px; color:#78909c; margin-top:4px; }
.filtered-list { background:#1a1111; border:1px solid #4a1a1a; border-radius:8px; padding:12px; margin:10px 0; }
.filtered-list h3 { color:#ef5350; font-size:15px; margin-bottom:6px; }
.filtered-list li { color:#b0bec5; font-size:13px; margin:4px 0; list-style:none; }
.equity-img { width:100%; border-radius:8px; margin-top:12px; border:1px solid #1a237e; }
.footer { text-align:center; color:#546e7a; font-size:11px; padding:20px 0; border-top:1px solid #1a237e; margin-top:30px; }
""")
    w('</style>')
    w('</head><body>')
    w('<div class="container">')

    # ─── HEADER ───
    w('<div class="header">')
    w('<h1>📈 Futu ML Backtest Dashboard</h1>')
    w(f'<div class="header-meta">⏰ {gen_time} | 🏢 {config.get("model","SVM-rbf")} | '
      f'📊 {config.get("walk_forward_splits",10)}-fold Walk-Forward | '
      f'🔬 {config.get("feature_count",41)} features | '
      f'📏 {config.get("lookback_years",5)}yr lookback | '
      f'💰 ${config.get("start_capital",100000):,.0f} start, {config.get("position_size",0.5):.0%} pos</div>')

    # Summary stats
    total = len(stocks)
    avg_acc = 0
    avg_sharpe = 0
    best_name = ''
    best_sharpe = 0
    if stocks:
        accs = [s['results'][0].get('sharpe', 0) or 0 for s in stocks if s.get('results')]
        avg_acc = sum(s['results'][0].get('acc', 0) or 0 for s in stocks if s.get('results')) / max(len(stocks), 1)
        avg_sharpe = sum(accs) / max(len(accs), 1)
        best = max(stocks, key=lambda s: (s['results'][0].get('sharpe', 0) or 0) if s.get('results') else -999)
        if best.get('results'):
            best_name = best['name']
            best_sharpe = best['results'][0].get('sharpe', 0) or 0

    w(f'<div class="header-meta">📈 股票總數: {total} | '
      f'平均 Acc: {avg_acc:.2%} | 平均 Sharpe: {avg_sharpe:.2f} | '
      f'🏆 最佳: {best_name} (Sharpe {best_sharpe:.2f}) | '
      f'🚫 篩走: {len(filtered)}</div>')

    # Navigation
    w('<div class="nav">')
    for s in stocks:
        ticker = s.get('ticker', '?')
        name = s.get('name', '?')
        w(f'<a class="nav-link" href="#{ticker}">{name}</a>')
    w('</div></div>')

    # ─── FILTERED ───
    if filtered:
        w('<div class="filtered-list"><h3>🚫 基本面篩選未通過</h3>')
        w('<ul>')
        for f in filtered:
            reason = f.get('filter_reason', '?')
            fs = f.get('fundu_score', '')
            fs_str = f' (Score={fs:.0f})' if fs else ''
            w(f'<li>❌ {f.get("name","?")} ({f.get("ticker","?")}) — {reason}{fs_str}</li>')
        w('</ul></div>')

    # ─── SUMMARY TABLE ───
    w('<h2>📊 總覽</h2>')
    w('<table class="summary-table">')
    w('<tr><th>#</th><th>股票</th><th>ML Acc</th><th>Ret</th><th>MaxDD</th><th>Sharpe</th><th>Sortino</th>'
      '<th>WR</th><th>PF</th><th>Fund Score</th></tr>')
    for i, s in enumerate(stocks, 1):
        r = s.get('results', [{}])[0]
        name = s.get('name', '?')
        ticker = s.get('ticker', '?')
        acc = r.get('acc', 0)
        ret = r.get('total_return', 0)
        mdd = r.get('max_dd', 0)
        sharpe = r.get('sharpe', 0)
        sortino = r.get('sortino', 0)
        wr = r.get('win_rate', 0)
        pf = r.get('profit_factor', 0)
        fs = s.get('fundamental_score') or r.get('fundu_score')
        fs_lbl, fs_cls = fund_score_label(fs)
        w(f'<tr><td>{i}</td>'
          f'<td><a class="stock-link" href="#{ticker}">{name} <span class="stock-ticker">{ticker}</span></a></td>'
          f'<td class="num {clr_val(acc-0.5)}">{acc:.2%}</td>'
          f'<td class="num {clr_val(ret)}">{ret:+.2%}</td>'
          f'<td class="num red">{mdd:.2%}</td>'
          f'<td class="num {clr_val(sharpe)}">{sharpe:.2f}</td>'
          f'<td class="num {clr_val(sortino)}">{sortino:.2f}</td>'
          f'<td class="num {clr_val(wr-0.5)}">{wr:.1%}</td>'
          f'<td class="num {clr_val(pf-1)}">{pf:.2f}</td>'
          f'<td class="num {fs_cls}">{fs_lbl}</td></tr>')
    w('</table>')

    # ─── STOCK CARDS ───
    for s in stocks:
        r = s.get('results', [{}])[0]
        name = s.get('name', '?')
        ticker = s.get('ticker', '?')
        stype = s.get('type', '?')
        period = s.get('data_period', '—')
        days = s.get('total_days', 0)

        acc = r.get('acc', 0)
        acc_std = r.get('acc_std', 0)
        baseline = r.get('baseline', 0)
        ret = r.get('total_return', 0)
        mdd = r.get('max_dd', 0)
        sharpe = r.get('sharpe', 0)
        sortino = r.get('sortino', 0)
        calmar = r.get('calmar', 0)
        wr = r.get('win_rate', 0)
        pf = r.get('profit_factor', 0)
        trades = r.get('trades', 0)
        avg_win = r.get('avg_win', 0)
        avg_loss = r.get('avg_loss', 0)
        fold_accs = r.get('fold_accs', [])
        features = r.get('features', {})
        fundu_details = r.get('fundu_details', {})
        fs = s.get('fundamental_score') or r.get('fundu_score')
        chart_path = s.get('chart', '')

        w(f'<div class="stock-card" id="{ticker}">')
        w('<div class="stock-header">')
        w(f'<span class="stock-name">{name}</span>'
          f'<span class="stock-ticker">{ticker}</span>'
          f'<span class="stock-type">{stype}</span>')
        w(f'<div class="stock-period">{period} | {days} bars</div>')
        w('</div>')

        # Metrics Grid
        w('<div class="metrics-grid">')
        w(f'<div class="metric-card {clr_val(sharpe)}"><div class="metric-value">{sharpe:.2f}</div><div class="metric-label">Sharpe</div></div>')
        w(f'<div class="metric-card {clr_val(ret)}"><div class="metric-value">{ret:+.2%}</div><div class="metric-label">Return</div></div>')
        w(f'<div class="metric-card red"><div class="metric-value">{mdd:.2%}</div><div class="metric-label">MaxDD</div></div>')
        w(f'<div class="metric-card {clr_val(acc-0.5)}"><div class="metric-value">{acc:.2%}</div><div class="metric-label">Accuracy</div></div>')
        w(f'<div class="metric-card {clr_val(sortino)}"><div class="metric-value">{sortino:.2f}</div><div class="metric-label">Sortino</div></div>')
        w(f'<div class="metric-card {clr_val(wr-0.5)}"><div class="metric-value">{wr:.1%}</div><div class="metric-label">Win Rate</div></div>')
        w('</div>')

        # Two-column detail panels
        w('<div class="stock-grid">')

        # ML Performance panel
        w('<div class="panel">')
        w('<div class="panel-title">🧠 ML Model</div>')
        w('<table class="metric-table">')
        pairs = [
            ('模型', f'SVM-rbf (C={r.get("params",{}).get("C","?")}, gamma={r.get("params",{}).get("gamma","?")})'),
            ('準確率', f'{acc:.2%} ± {acc_std:.2%}'),
            ('Baseline', f'{baseline:.2%}'),
            ('Improvement', f'{acc-baseline:+.2%}'),
            ('Fold 準確率', ', '.join(f'{a:.1%}' for a in fold_accs[:10])),
        ]
        for label, val in pairs:
            w(f'<tr><td class="ml-label">{label}</td><td class="ml-val">{val}</td></tr>')
        w('</table></div>')

        # Trade panel
        w('<div class="panel">')
        w('<div class="panel-title">💰 交易績效</div>')
        w('<table class="metric-table">')
        pf_str = f'{pf:.2f}' if isinstance(pf, float) else str(pf)
        calmar_str = f'{calmar:.2f}' if calmar else '—'
        pairs2 = [
            ('總回報', f'{ret:+.2%}'),
            ('Max Drawdown', f'{mdd:.2%}'),
            ('Sharpe', f'{sharpe:.2f}'),
            ('Sortino', f'{sortino:.2f}'),
            ('Calmar', calmar_str),
            ('Win Rate', f'{wr:.1%}'),
            ('Profit Factor', pf_str),
            ('Avg Win', f'{avg_win:.2%}'),
            ('Avg Loss', f'{avg_loss:.2%}'),
            ('交易次數', str(trades)),
        ]
        for label, val in pairs2:
            w(f'<tr><td class="ml-label">{label}</td><td class="ml-val">{val}</td></tr>')
        w('</table></div>')

        # Fundamental panel
        fd_str_lines = []
        if fundu_details:
            w('<div class="panel">')
            w(f'<div class="panel-title">🏢 基本面 <span class="{fund_score_label(fs)[1]}">(Score: {fund_score_label(fs)[0]})</span></div>')
            w('<table class="metric-table">')
            fd_map = {
                'pe_ttm': ('PE (TTM)', fmt_num),
                'pb': ('PB', fmt_num),
                'roe': ('ROE', lambda v: fmt_pct(v/100) if v and v > 1 else fmt_pct(v)),
                'profit_margin': ('Profit Margin', fmt_pct),
                'revenue_growth_yoy': ('Rev Growth', fmt_pct),
                'market_cap': ('Market Cap', lambda v: f'${v/1e9:.1f}B' if v else '—'),
                'debt_to_equity': ('D/E', fmt_num),
            }
            for field, (label, formatter) in fd_map.items():
                val = fundu_details.get(field)
                if val is not None and val != 'None' and str(val) != 'nan':
                    w(f'<tr><td class="ml-label">{label}</td><td class="ml-val">{formatter(val)}</td></tr>')
            w('</table></div>')
        else:
            w('<div class="panel">')
            w('<div class="panel-title">🏢 基本面</div>')
            w(f'<div style="color:#78909c;font-size:12px;padding:8px;">評分: {fund_score_label(fs)[0]} | 無詳細數據</div>')
            w('</div>')

        w('</div>')  # end stock-grid

        # Feature importance
        if features:
            if isinstance(features, dict):
                feat_items = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            else:
                feat_items = list(features)[:10]
            w('<div class="panel feat-panel">')
            w('<div class="panel-title">🔑 Top 10 Features</div>')
            w('<div style="display:flex;flex-wrap:wrap;gap:4px;">')
            for fname, fval in feat_items:
                bar_w = max(3, min(100, abs(fval) * 100))
                bar_clr = '#26a69a' if fval > 0 else '#ef5350'
                w(f'<div style="flex:1 1 45%;font-size:11px;margin:2px 0;">'
                  f'<span style="color:#78909c;">{fname}:</span> '
                  f'<span style="color:{bar_clr};font-weight:600;">{fval:.3f}</span>'
                  f'<div style="height:4px;background:#1e293b;border-radius:2px;margin-top:2px;overflow:hidden;">'
                  f'<div style="height:100%;width:{bar_w}%;background:{bar_clr};border-radius:2px;"></div></div>'
                  f'</div>')
            w('</div></div>')

        # Equity curve image
        if chart_path:
            b64 = embed_png(chart_path)
            if b64:
                w(f'<img class="equity-img" src="{b64}" alt="{name} Equity Curve">')

        w('</div>')  # end stock-card

    # ─── FOOTER ───
    w('<div class="footer">')
    w(f'⚠️ 免責: 過往回測不代表未來結果。模型存在過擬合風險。{config.get("position_size",0.5):.0%}倉位，未計手續費/滑點。基本面為最新snapshot。')
    w('</div></div></body></html>')

    html = '\n'.join(lines)
    OUTPUT.write_text(html, encoding='utf-8')
    print(f'✅ Dashboard: {OUTPUT} ({os.path.getsize(OUTPUT)/1e6:.1f}MB)', file=sys.stderr)

if __name__ == '__main__':
    generate()