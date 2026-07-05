#!/bin/bash
# Daily ML Backtest Pipeline — run backtest → gen dashboard → git push
# Runs from futu-ml-backtest repo directory
set -e

cd /home/kenny/futu-ml-backtest

echo "🚀 Daily ML Backtest Pipeline — $(date '+%Y-%m-%d %H:%M HKT')"
echo "══════════════════════════════════════════════"

# Step 1: Run backtest (generates JSON + PDF + charts)
echo ""
echo "📊 Step 1/3: Running ML Backtest..."
# Use timeout to prevent hanging — backtest for ~30 stocks can take 60-90 min
timeout 5400 python3 futu_ml_backtest.py 2>/dev/null || {
    echo "⚠️ Backtest returned non-zero, check stderr for details"
}

# Step 2: Generate HTML dashboard from JSON
echo ""
echo "📈 Step 2/3: Generating HTML Dashboard..."
python3 gen_dashboard.py 2>/dev/null || {
    echo "⚠️ Dashboard generation failed"
}

# Step 3: Git commit + push
echo ""
echo "🔄 Step 3/3: Pushing to GitHub..."
git add -A
git diff --cached --quiet || {
    git commit -m "auto: daily backtest update $(date '+%Y-%m-%d')"
    git push origin main && echo "✅ Pushed to GitHub" || echo "⚠️ Push failed"
}
echo ""
echo "✅ Pipeline complete — $(date '+%Y-%m-%d %H:%M HKT')"