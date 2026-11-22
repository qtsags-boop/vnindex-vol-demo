#!/usr/bin/env bash
# Dựng site từ web/ + state/site_data rồi đẩy lên nhánh gh-pages.
set -euo pipefail
python3 pwa/build_site.py "${GITHUB_SHA:-dev}" --state state --out _site
bash .github/scripts/push_dir.sh _site gh-pages "Web ${GITHUB_SHA::7} · $(TZ=Asia/Ho_Chi_Minh date '+%Y-%m-%d %H:%M')"
