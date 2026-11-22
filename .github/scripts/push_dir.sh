#!/usr/bin/env bash
# Đẩy nội dung một thư mục lên một nhánh dưới dạng 1 commit duy nhất (ghi đè), như nhánh apk của app A330.
#   push_dir.sh <thư mục> <nhánh> <thông điệp>
set -euo pipefail
DIR="$1"; BRANCH="$2"; MSG="$3"
TMP="$(mktemp -d)"
cp -a "$DIR"/. "$TMP"/
rm -rf "$TMP/.git"
cd "$TMP"
git init -q -b "$BRANCH"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
git commit -q -m "$MSG"
git push -q -f "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" "$BRANCH"
echo "Đã đẩy $DIR → nhánh $BRANCH"
