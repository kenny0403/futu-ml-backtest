# Futu ML Backtest System 📈

基於富途數據嘅 Hybrid ML Backtest 系統 — SVM model + Fundamental Pre-filter

## 系統架構

```
富途OpenAPI (FutuOpenD)
    ├── request_history_kline → 歷史K線 (QFQ前復權)
    ├── get_stock_screen      → 基本面篩選 (PE/ROE/市值)
    └── get_financials_statements → 財務數據
                        │
                        ▼
              Hybrid 3-Layer Filter
    ┌─────────────────────────────────┐
    │ Layer 1: Fundamental Pre-Filter │ ← 硬篩選 (PE<30, ROE>5%, 市值>50億)
    │ Layer 2: SVM ML Model (41 feat)│ ← 41個技術因子預測
    │ Layer 3: Fundamental Score Adj  │ ← 基本面分數調整Confidence (±30%)
    └─────────────────────────────────┘
                        │
                        ▼
                Per-Stock Performance
    ├── Accuracy / Win Rate / Sharpe Ratio
    ├── Max Drawdown / Calmar Ratio
    ├── Equity Curve (buy/sell markers)
    └── Cantonese PDF Report
```

## 使用方式

```bash
python futu_ml_backtest.py
```

需要 FutuOpenD gateway 運行中（預設 localhost:11111）。

## 參數設定

| 參數 | HK 股票 | US 股票 |
|------|---------|---------|
| min_market_cap | 50億 HKD | 100億 USD |
| max_pe | 30 | 40 |
| min_roe | 5% | 0% |
| 訓練窗口 | 200 bars | 200 bars |
| CV folds | 10 | 10 |

## 技術因子 (41個)

Price-based: ret_1d/5d/10d/20d, close_ma5/10/20/50, ma5_ma20, ma10_ma50, ema12, ema26
Momentum: macd_line, macd_hist, rsi_14/7, bb_width, bb_position, atr_pct
Volume: obv, vol_ratio, volatility_10
Time: dayofweek, dayofmonth
Candle: doji, hammer, engulfing

## 輸出

- `charts/*.png` — 9 張 equity curve 圖表（dark theme，買賣點標記）
- `futu_ml_report.pdf` — Cantonese PDF 報告
- `futu_ml_results.json` — 完整結果數據