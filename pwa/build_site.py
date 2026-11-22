"""Dựng trang GitHub Pages (web cài được như ứng dụng / PWA) từ thư mục web/ và dữ liệu state/site_data.

    python pwa/build_site.py [sha] [--state state] [--out _site]
"""
import argparse
import json
import os
import shutil
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("sha", nargs="?", default="dev")
ap.add_argument("--state", default=os.path.join(ROOT, "state"))
ap.add_argument("--out", default=os.path.join(ROOT, "_site"))
a = ap.parse_args()

repo = os.environ.get("GITHUB_REPOSITORY", "")
owner, _, name = repo.partition("/")
remote = f"https://{owner.lower()}.github.io/{name}/" if repo else ""
now = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d-%H%M")
version = f"{a.sha[:7]}·{now}"

out = a.out
shutil.rmtree(out, ignore_errors=True)
shutil.copytree(os.path.join(ROOT, "web"), out)
os.makedirs(os.path.join(out, "data"), exist_ok=True)
src = os.path.join(a.state, "site_data")
n = 0
if os.path.isdir(src):
    for f in os.listdir(src):
        if f.endswith(".json"):
            shutil.copy(os.path.join(src, f), os.path.join(out, "data", f))
            n += 1

html = open(os.path.join(out, "index.html"), encoding="utf-8").read()
head = """<link rel="manifest" href="manifest.webmanifest">
<meta name="theme-color" content="#0E1B2C">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="VN-Index Vol">
<link rel="apple-touch-icon" href="icons/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="192x192" href="icons/icon-192.png">"""
cfg = f"<script>window.APP_VERSION={json.dumps(version)};window.APP_REMOTE={json.dumps(remote)};</script>"
html = html.replace("<!--PWA_HEAD-->", head).replace("<!--APP_CONFIG-->", cfg)
open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(html)

sw = open(os.path.join(ROOT, "pwa", "sw.js"), encoding="utf-8").read().replace("__VERSION__", version)
open(os.path.join(out, "sw.js"), "w", encoding="utf-8").write(sw)
shutil.copy(os.path.join(ROOT, "pwa", "manifest.webmanifest"), out)
shutil.copytree(os.path.join(ROOT, "pwa", "icons"), os.path.join(out, "icons"))
open(os.path.join(out, ".nojekyll"), "w").close()
print(f"Đã dựng site → {out} ({n} file dữ liệu, phiên bản {version}, remote={remote or '-'})")
