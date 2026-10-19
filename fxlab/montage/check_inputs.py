#!/usr/bin/env python3
"""检查 input/ 下的素材：文件名、年份、分辨率、帧率、时长、有无音轨，并标出命名不规范的文件。
只读，不修改任何文件。需要本机已有 python3 和 ffprobe（缺少时只提示，不安装）。"""
import json, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXTS = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm", ".m4v", ".mp3", ".wav", ".m4a", ".aac", ".flac"}
NAME_RE = re.compile(r"^(\d{4})_(\d{2,})_(.+)\.mp4$", re.I)


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                        "-show_format", "-show_streams", str(path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)


def fps(s):
    try:
        a, b = s.split("/")
        return f"{float(a) / float(b):.2f}" if float(b) else "?"
    except Exception:
        return "?"


def report(folder, check_name):
    files = sorted(p for p in (ROOT / "input" / folder).iterdir()
                   if p.is_file() and p.suffix.lower() in EXTS)
    print(f"\n== input/{folder}/  共 {len(files)} 个文件 ==")
    bad = []
    for p in files:
        m = NAME_RE.match(p.name)
        year = m.group(1) if m else "-"
        info = probe(p)
        if info is None:
            print(f"{p.name} | 年份 {year} | 无法读取（文件可能损坏）")
            bad.append(p.name)
            continue
        v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
        has_a = any(s["codec_type"] == "audio" for s in info["streams"])
        dur = float(info["format"].get("duration", 0))
        res = f"{v['width']}x{v['height']}" if v else "-"
        rate = fps(v.get("avg_frame_rate", "")) if v else "-"
        flag = ""
        if check_name and not m:
            flag = "  ⚠ 命名不规范"
            bad.append(p.name)
        print(f"{p.name} | 年份 {year} | {res} | {rate}fps | {dur:.1f}秒 | "
              f"{'有音轨' if has_a else '无音轨'}{flag}")
    return bad


def main():
    missing = [t for t in ("ffprobe",) if not shutil.which(t)]
    if missing:
        sys.exit("本机缺少：" + "、".join(missing) + "（未自行安装，请先装好再说「检查素材」）")
    bad = report("clips", True)
    report("music", False)
    print(f"\n命名不规范/无法读取：{len(bad)} 个" + ("：" + "、".join(bad) if bad else ""))


if __name__ == "__main__":
    main()
