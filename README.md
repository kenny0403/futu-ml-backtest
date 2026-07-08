# Futu ML Backtest & Trading System 📈

基於富途 OpenAPI + SVM(rbf) Machine Learning 嘅港股美股自動交易系統。
包含 **歷史回測** (`futu_ml_backtest.py`) 同 **即市掃描交易** (`ml_auto_trader.py`) 兩個組件。

---

## 系統總覽

```
富途 OpenD (TCP:11111)
    │
    ├─ request_history_kline → 歷史日K (QFQ前復權)
    ├─ get_stock_screen      → 基本面篩選 (PE/PB/ROE/市值)
    ├─ get_capital_flow      → 盤路資金流 (大/中/小單)
    ├─ get_short_interest    → 沽空比率 / US Short %
    ├─ get_shareholders_institutional → 機構持倉變化
    └─ get_financials_statements → 財務報表
                │
                ▼
    ┌──────────────────────────────────────┐
    │   Layer 1: 基本面 Soft Scoring       │  ← 0-100 分（非硬閘）
    │   Layer 2: SVM(rbf) 技術因子模型     │  ← 42/22 features
    │   Layer 3: 盤路特徵 (資金流/沽空)    │  ← 11 features (2026-07-08+)
    │   Layer 4: Confidence 綜合判定       │  ← Buy% × AvgAcc × FundScore
    └──────────────────────────────────────┘
                │
                ▼
    模擬倉執行 (SIMULATE only!)
    HK: 9423131 | US: 12144954
```

---

## 核心 ML 模型

### 主模型：SVM(rbf) — Support Vector Machine with RBF Kernel

| 參數 | 預設值 | 搜索範圍 | 說明 |
|------|--------|----------|------|
| **C** (正則化) | 10 | [0.1, 1, 10] | 控制模型複雜度。大 C = 貼 data，細 C = 穩定 |
| **gamma** (核寬) | 'scale' | ['scale', 'auto'] | 每點 training data 嘅影響半徑 |
| **kernel** | 'rbf' | — | 非線性邊界 |
| **probability** | True | — | 輸出 Buy% 概率 |

**Why SVM(rbf)?** Backtest 比咗 9 個 models — SVM(rbf) 贏 9/13 隻股票，平衡最好：
- Transformer：Conf 高但信號唔穩定（CIEN 錯 SELL signal）
- LSTM：全部輸俾 SVM，細 data 完全唔 work
- 其他（RF/GB/MLP）：冇乜 edge

### 輔助模型：RandomForest（特徵重要性分析）

用嚟計算邊個技術因子對 prediction 貢獻最大，唔用嚟 trade。

---

## 兩套 Feature Set

### 1️⃣ 回測用 — 42 技術因子 + 11 盤路因子（53 features）

```python
# 技術因子 (42個)
ret_1d/5d/10d/20d          # 不同週期回報
ma5/10/20/50               # 移動平均
close_ma5/10/20/50         # 價格相對 MA 距離
ma5_ma20, ma10_ma50        # MA 交叉
ema12, ema26, ema_diff     # EMA
macd, macd_signal, macd_hist  # MACD
rsi_14, rsi_7              # RSI
bb_upper/lower/width/position  # Bollinger Band
atr_14, atr_pct            # ATR 波動率
stoch_k                    # Stochastic
vol_ma5, vol_ratio_5/20    # 成交量
obv, obv_ma5, obv_ratio    # OBV
volatility_5d/20d          # 波動率
range_pct, body_pct, upper/lower_shadow  # 燭台特徵

# 盤路因子 (11個) — 資金流/沽空/機構持倉
short_sell_ratio           # 港股沽空比率 (ArcticDB)
capital_in_flow            # 資金淨流入
capital_super_in           # 超大手淨流入
capital_big_in             # 大單淨流入
capital_mid_in             # 中單淨流入
capital_sml_in             # 小單淨流入
capital_main_in            # 主力淨流入 (超大手+大單)
short_interest_pct         # 美股沽空比例
us_short_pct               # US short % of float
institutional_hold_pct     # 機構持倉 %
institutional_change       # 機構持倉 QoQ 變化
```

### 2️⃣ 即市掃描用 — 22 核心因子（快 + effective）

只保留 22 個最高 Discriminative Power 嘅 features，trade-off 少少準確度換 scanning speed（~3 mins scan 734 HK + ~8s 1507 US stocks）。

```python
ret_1d/5d/10d/20d          # 回報
close_ma5/10/20/50         # MA 距離
ma5_ma20, ma10_ma50        # MA 交叉
rsi_14, rsi_7              # 超買超賣
macd_hist                  # MACD 動能
bb_width, bb_position      # 波幅位置
atr_pct                    # 波動率
stoch_k, stoch_d           # Stochastic
vol_ratio                  # 成交量比
volatility_10              # 10日波動率
dayofweek, dayofmonth      # 時間效應
```

---

## 驗證方法論

### Walk-Forward (TimeSeriesSplit) — 10-fold

```python
# 核心邏輯
tscv = TimeSeriesSplit(n_splits=10)

for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    # Train on fold 1..N-1, Test on fold N
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Grid search per fold: C=[0.1,1,10] × gamma=['scale','auto'] = 6 combos
    for params in SVM_GRID:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
        model = SVC(**params).fit(Xtr, y_train)
        acc = accuracy_score(y_test, model.predict(Xte))
        # Pick best params for this fold

# Final AvgAcc = mean of all 10 fold accuracies
# Final model = refit on ALL data with average best params
```

**為什麼不用簡單 80/20 split？**
- 10-fold 每支 bar 至少做過一次 test data
- Accuracy std 低好多（±3-6% vs ±12-19%）
- 尤其係 200 bars 呢個規模，random split 好易俾某一 set test data 扭曲結果

### 即市用：StratifiedKFold（10-fold）

`ml_auto_trader.py` 用 StratifiedKFold（確保每個 fold 保持正負例比例），因為即時 scan 用全部 data 做 final model，fold 只係用嚟 estimate 模型可靠度。

---

## Confidence 計算

```
Confidence = Buy% × AvgAcc × 100
```

| 閾值 | 數字 | 來源 |
|------|------|------|
| MIN_BUY_PROB | 55% | 優化後（原來 60% 係死區，Sharpe -0.10） |
| MIN_CONFIDENCE | 27 | 5×200 walk-forward optimization |
| SELL_THRESHOLD | 35% | Buy% 低過 35% 賣出 |

**60% 死區現象：** Backtest 發現 Buy% 喺 55-60% 之間嘅 trade 平均回報最高。60% 以上反而 signal 過度自信，經常係 overfit 嘅結果。

Fundamental Score 會 adjust Confidence ±5-10%（Score>80 +5%, Score<50 -10%）。

---

## 基本面 Soft Scoring（非硬閘）

2026-07-05 由 hard binary filter 改做 **soft scoring system**。

### 公式 (0-100)

| Factor | 條件 | +/‑ | Max |
|--------|------|:---:|:---:|
| PE | 5-25 | +20 | ±20 |
| PE | 25-40 | +10 | |
| PE | Negative | **‑20** | |
| ROE | >20% | +20 | ±20 |
| ROE | >10% | +10 | |
| ROE | <0 | **‑15** | |
| Profit Margin | >20% | +15 | ±15 |
| Revenue Growth | >20% | +10 | ±10 |
| 初始分 | — | 50 | 50 |
| **Total** | | | **0-115 → clamped 0-100** |

### 預篩選（只 block 真正「有毒」嘅）

只有同時滿足以下 **全部** 條件先會被 block：
1. PE < 0（蝕錢）
2. ROE < -30%（深度虧損）
3. 營收增長 < 2%（冇增長）
4. 市值 < 100億 USD（唔係 mega-cap）

NVDA/AMD/AVGO/COHR 呢啲高 PE 但大增長嘅股全部 pass。

---

## 盤路特徵（2026-07-08 加入）

| 數據源 | API | 頻率 | HK | US |
|--------|-----|:----:|:--:|:--:|
| 沽空比率 | ArcticDB `short_selling_nb` | 每日 | ✅ | ❌ |
| 資金流 | `get_capital_flow(PeriodType.DAY)` | 每日 | ✅ | ✅ |
| 沽空比例 (US) | `get_short_interest()` | 雙週 | ❌ | ✅ |
| 機構持倉 | `get_shareholders_institutional()` | 季度 | ✅ | ✅ |

Missing data forward-fill 處理 — 唔會因為盤路 data 唔齊而 skip 隻股票。

---

## 兩大組件對比

| Feature | `futu_ml_backtest.py` | `ml_auto_trader.py` |
|---------|-----------------------|---------------------|
| 用途 | 歷史回測 + 報告 | 即市掃描 + 模擬倉交易 |
| Features | 42 技術 + 11 盤路 | 22 技術 |
| 驗證 | TimeSeriesSplit 10-fold | StratifiedKFold 10-fold |
| 數據長度 | 5年 | 200 bars (~10個月) |
| 掃描範圍 | 硬編碼 27 隻 / --top-hk N | >50億市值動態（~734 HK）+ >100億（~1500 US） |
| 執行 | 唔落單 | 落模擬倉單 |
| 輸出 | `.json` / `.pdf` / `.html` / charts | 即時 Telegram report |

---

## 自動化 Cron 工作

| 工作 | 時間 | 做乜 |
|------|------|------|
| pre-market scan | 08:45 HKT | ML scan + 短賣率/DI enrichment → Telegram |
| 港股 scan | 09:00 HKT | 重點掃描 + trade |
| 美股 scan | 21:00 HKT | US 主力掃描 + trade |
| 港股收市報告 | 16:15 HKT | P&L summary |
| 美股收市報告 | 05:15 HKT | US P&L summary |
| Daily Recap | 05:30 HKT |  equity curve + trade review + 策略建議 |
| Dashboard 更新 | 06:00 HKT | 回測 → HTML dashboard → Git push |
| 數據 pipeline | 16:16 HKT | 沽空 + 基本面 → ArcticDB |
| Gateway watchdog | 每 3 分鐘 | 自動重啟死咗嘅 gateway |

---

## Dashboard

https://kenny0403.github.io/futu-ml-backtest/futu_ml_dashboard.html

每隻股票顯示：
- Sharpe/Sortino/Calmar Ratio、MaxDD、Profit Factor
- 10-fold 每折 accuracy bar chart
- Feature Importance top-10
- 基本面分數 + PE/PB/ROE
- Equity curve（買賣標記，$100K 起始）

---

## 歷史 Backtest 結果 (2026-07-05, 22 stocks)

| Stock | Acc | Sharpe | MaxDD | FundScore |
|-------|:---:|:------:|:-----:|:---------:|
| SNDK | 61.5% | 5.09 | 4.7% | 47 |
| AMD | 53.4% | 4.03 | 1.7% | 47 |
| 1888建滔 | 53.5% | 3.94 | 4.1% | 52 |
| 0700騰訊 | 54.2% | 3.15 | 6.4% | 82 |
| 7709湛江 | 60.0% | 3.08 | 7.7% | 50 |
| 0189東岳 | 54.6% | 2.69 | 5.6% | 90 |
| NVDA | 54.1% | 1.72 | 12.3% | 57 |
| MSFT | 54.9% | 0.86 | 14.7% | 67 |

---

## 重要安全規則

> **永遠用模擬倉！永遠用模擬倉！永遠用模擬倉！**
> HK: acc_id=9423131 (SIMULATE CASH)
> US: acc_id=12144954 (SIMULATE STOCK_AND_OPTION)

---

## Dependencies

```
富途 OpenD (gateway running on 172.17.96.1:11111)
Python 3.11+ with:
  - futu (OpenAPI SDK)
  - pandas, numpy
  - scikit-learn (SVM, RF, StandardScaler)
  - matplotlib (equity curve charts)
  - reportlab (PDF reports)
  - PyTorch 2.12.1 (optional — for DL comparison only)
  - ArcticDB (HKEX data store)
```

---

## 點用

```bash
# 完整回測
cd ~/.hermes/profiles/trading/scripts
python futu_ml_backtest.py

# 指定 top N
python futu_ml_backtest.py --top-hk 50 --top-us 100

# 指定股票
python futu_ml_backtest.py US.NVDA HK.00700

# 即市 scan (唔落單)
cd ~/.hermes/profiles/trading/scripts
python ml_auto_trader.py scan

# 即市 scan + trade
python ml_auto_trader.py trade

# 生成 Dashboard
python gen_dashboard.py
```