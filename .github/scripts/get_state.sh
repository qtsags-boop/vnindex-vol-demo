#!/usr/bin/env bash
# Lấy trạng thái (dữ liệu, mô hình, kết quả) từ nhánh state vào ./state; lần đầu chưa có thì tạo rỗng.
set -euo pipefail
rm -rf state
if git ls-remote --exit-code --heads "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" state >/dev/null 2>&1; then
  git clone -q --depth 1 --branch state "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" state
  rm -rf state/.git
  echo "Đã lấy nhánh state."
else
  mkdir -p state
  echo "Chưa có nhánh state — bắt đầu từ trạng thái rỗng."
fi
