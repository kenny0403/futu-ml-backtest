#!/bin/bash
# Daily ML Backtest Pipeline — run backtest → gen dashboard → git push
# Runs from futu-ml-backtest repo directory
set -e

cd /home/kenny/futu-ml-backtest

PIPELINE_LOG="/home/kenny/.hermes/profiles/trading/cache/pipeline_last_run.log"
echo "🚀 Daily ML Backtest Pipeline — $(date '+%Y-%m-%d %H:%M HKT')" | tee "$PIPELINE_LOG"
echo "══════════════════════════════════════════════" | tee -a "$PIPELINE_LOG"

# Step 1: Run backtest (smaller universe → faster, top 30 HK + top 20 US)
echo "" | tee -a "$PIPELINE_LOG"
echo "📊 Step 1/3: Running ML Backtest (top 30 HK + top 20 US)..." | tee -a "$PIPELINE_LOG"
# Capture stderr too — don't hide errors
timeout 5400 python3 futu_ml_backtest.py --top-hk 30 --top-us 20 >> "$PIPELINE_LOG" 2>&1
BACKTEST_EXIT=$?

if [ $BACKTEST_EXIT -ne 0 ]; then
    echo "❌ Backtest FAILED (exit=$BACKTEST_EXIT). Check log:" | tee -a "$PIPELINE_LOG"
    echo "$PIPELINE_LOG" | tee -a "$PIPELINE_LOG"
    exit $BACKTEST_EXIT
fi

echo "✅ Backtest complete" | tee -a "$PIPELINE_LOG"

# Step 2: Generate HTML dashboard from JSON
echo "" | tee -a "$PIPELINE_LOG"
echo "📈 Step 2/3: Generating HTML Dashboard..." | tee -a "$PIPELINE_LOG"
python3 gen_dashboard.py >> "$PIPELINE_LOG" 2>&1
DASH_EXIT=$?

if [ $DASH_EXIT -ne 0 ]; then
    echo "❌ Dashboard generation FAILED (exit=$DASH_EXIT)" | tee -a "$PIPELINE_LOG"
    exit $DASH_EXIT
fi
echo "✅ Dashboard generated" | tee -a "$PIPELINE_LOG"

# Step 3: Git commit + push
echo "" | tee -a "$PIPELINE_LOG"
echo "🔄 Step 3/3: Pushing to GitHub..." | tee -a "$PIPELINE_LOG"
git add -A
git diff --cached --quiet || {
    git commit -m "auto: daily backtest update $(date '+%Y-%m-%d')"
    git push origin main && echo "✅ Pushed to GitHub" | tee -a "$PIPELINE_LOG" || echo "⚠️ Push failed" | tee -a "$PIPELINE_LOG"
}
echo "" | tee -a "$PIPELINE_LOG"
echo "✅ Pipeline complete — $(date '+%Y-%m-%d %H:%M HKT')" | tee -a "$PIPELINE_LOG"