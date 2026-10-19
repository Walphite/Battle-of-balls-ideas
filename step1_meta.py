#!/usr/bin/env python3
"""第 1 步：只拉元数据（不下载视频），输出 meta.csv 并按年份统计。"""
import csv, json, subprocess, sys
from collections import Counter
from datetime import datetime

UID = "363185881"
cmd = ["yt-dlp", "--flat-playlist", "--sleep-requests", "2", "-J",
       f"https://space.bilibili.com/{UID}/video"]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    sys.exit("yt-dlp 失败:\n" + r.stderr[-800:])
data = json.loads(r.stdout)

rows = []
for e in data.get("entries") or []:
    ts, ud = e.get("timestamp"), e.get("upload_date")
    if ts:
        date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    elif ud:
        date = f"{ud[:4]}-{ud[4:6]}-{ud[6:]}"
    else:
        date = ""
    rows.append({
        "bv": e.get("id") or "",
        "title": e.get("title") or "",
        "date": date,
        "duration": e.get("duration") or "",
        "views": e.get("view_count") if e.get("view_count") is not None else "",
        "danmaku": e.get("comment_count") if e.get("comment_count") is not None else "",
    })
rows.sort(key=lambda x: x["date"] or "9999")

with open("meta.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["bv"])
    w.writeheader(); w.writerows(rows)

print(f"共 {len(rows)} 个视频，已写入 meta.csv")
for y, n in sorted(Counter((x["date"][:4] or "未知") for x in rows).items()):
    print(f"  {y}: {n}")
for c in ("date", "duration", "views", "danmaku"):
    miss = sum(1 for x in rows if x[c] == "")
    if miss: print(f"  [注意] {c} 列有 {miss} 行为空")
