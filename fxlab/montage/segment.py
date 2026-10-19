#!/usr/bin/env python3
"""第 1 步：基于 540p 代理分析长视频，切段并输出 segments.csv 与联系图。

用法：python3 segment.py [--thr 0.35]
输入  work/proxy_540p.mp4
输出  work/segments.csv、work/scene_scores.npy、../docs/montage_sheets/sheet_N.jpg
"""
import argparse, csv, re, subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
PROXY = ROOT / "work" / "proxy_540p.mp4"
FPS = 30
SW, SH = 64, 36  # 分析用的小图尺寸


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)


def small_frames():
    """整片解码为 64x36 RGB，返回 (N,36,64,3) uint8。"""
    r = run(["ffmpeg", "-v", "error", "-i", str(PROXY), "-vf", f"fps={FPS},scale={SW}:{SH}",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    a = np.frombuffer(r.stdout, np.uint8)
    n = a.size // (SW * SH * 3)
    return a[: n * SW * SH * 3].reshape(n, SH, SW, 3)


def scene_scores():
    """ffmpeg 场景分数（逐帧），返回 {帧号: 分数}。"""
    r = run(["ffmpeg", "-v", "info", "-i", str(PROXY), "-vf",
             "select='gte(scene,0)',metadata=print:file=-", "-an", "-f", "null", "-"], text=True)
    out, scores, t = r.stdout, {}, None
    for line in out.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            t = float(m.group(1))
        m = re.search(r"lavfi.scene_score=([\d.]+)", line)
        if m and t is not None:
            scores[int(round(t * FPS))] = float(m.group(1))
    return scores


def audio_rms(win=0.1):
    r = run(["ffmpeg", "-v", "error", "-i", str(PROXY), "-vn", "-ac", "1", "-ar", "8000",
             "-f", "s16le", "-"])
    x = np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768
    n = int(8000 * win)
    k = x.size // n
    return np.sqrt((x[: k * n].reshape(k, n) ** 2).mean(1)), win


def color_name(rgb):
    import colorsys
    h, s, v = colorsys.rgb_to_hsv(*(np.asarray(rgb) / 255))
    if v < 0.2:
        return "黑"
    if s < 0.15:
        return "白" if v > 0.8 else "灰"
    names = [(15, "红"), (45, "橙"), (70, "黄"), (160, "绿"), (200, "青"), (260, "蓝"), (310, "紫"), (345, "粉"), (361, "红")]
    d = h * 360
    return next(n for lim, n in names if d < lim)


def cuts_from(scores, diff, thr, n):
    """场景分数过阈值 + 帧差尖峰，合并（0.3 秒内去重），返回切点帧号（含 0）。"""
    s = np.zeros(n)
    for f, v in scores.items():
        if f < n:
            s[f] = v
    cand = set(np.nonzero(s > thr)[0].tolist())
    # 相邻帧差尖峰：显著高于邻域，且绝对值不小
    for i in range(3, n - 3):
        nb = np.r_[diff[i - 3:i], diff[i + 1:i + 4]].mean() + 1e-6
        if diff[i] > 5 * nb and diff[i] > 12:
            cand.add(i)
    cuts, last = [0], 0
    for f in sorted(cand):
        if f - last >= int(0.3 * FPS):
            cuts.append(f)
            last = f
    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.35)
    args = ap.parse_args()

    fr = small_frames()
    n = len(fr)
    g = fr.mean(3)
    diff = np.r_[0, np.abs(np.diff(g, axis=0)).mean((1, 2))]
    sc = scene_scores()
    print(f"帧数 {n}，时长 {n / FPS:.1f}s")
    for t in (0.25, 0.35, 0.5):
        c = cuts_from(sc, diff, t, n)
        print(f"阈值 {t}: {len(c)} 段")
    cuts = cuts_from(sc, diff, args.thr, n) + [n]

    rms, win = audio_rms()
    rows = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        mo = diff[a + 2:b].max() if b - a > 3 else diff[a:b].max()
        r0, r1 = int(a / FPS / win), max(int(b / FPS / win), int(a / FPS / win) + 1)
        seg = fr[a:b]
        rows.append({
            "seg": i + 1, "start": round(a / FPS, 3), "end": round(b / FPS, 3),
            "dur": round((b - a) / FPS, 3), "motion_peak": round(float(mo), 2),
            "audio_peak": round(float(rms[r0:r1].max()) if r1 > r0 else 0, 4),
            "brightness": round(float(g[a:b].mean()), 1),
            "color": color_name(seg.reshape(-1, 3).mean(0)),
        })
    with open(ROOT / "work" / "segments.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    d = np.array([r["dur"] for r in rows])
    print(f"采用阈值 {args.thr}: {len(rows)} 段，平均 {d.mean():.2f}s，中位 {np.median(d):.2f}s，"
          f"最短 {d.min():.2f}s，最长 {d.max():.2f}s")
    odd = [(r["seg"], r["start"], r["dur"]) for r in rows if r["dur"] < 1.0 or r["dur"] > 6.0]
    print("可疑区间（<1s 或 >6s）：", odd)
    np.save(ROOT / "work" / "diff.npy", diff)
    sheets(rows)


def sheets(rows):
    """每段取起点后 0.15s 的帧，5x5 一张，每格标段号。"""
    out = ROOT.parent / "docs" / "montage_sheets"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("sheet_*.jpg"):
        old.unlink()
    cw, ch = 288, 200
    for s in range(0, len(rows), 25):
        chunk = rows[s:s + 25]
        sheet = Image.new("RGB", (cw * 5, ch * 5), (20, 20, 20))
        for k, r in enumerate(chunk):
            t = min(r["start"] + 0.15, r["end"] - 0.05)
            p = run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(PROXY), "-frames:v", "1",
                     "-f", "image2pipe", "-vcodec", "mjpeg", "-"])
            import io
            im = Image.open(io.BytesIO(p.stdout)).convert("RGB").resize((cw, ch))
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, 44, 20), fill=(0, 0, 0))
            d.text((4, 4), f"#{r['seg']}", fill=(255, 255, 0))
            sheet.paste(im, ((k % 5) * cw, (k // 5) * ch))
        sheet.save(out / f"sheet_{s // 25 + 1}.jpg", quality=80)
        print("saved", out / f"sheet_{s // 25 + 1}.jpg")


if __name__ == "__main__":
    main()
