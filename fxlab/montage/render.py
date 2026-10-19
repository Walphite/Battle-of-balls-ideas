#!/usr/bin/env python3
"""渲染模块（ffmpeg 滤镜实现，不逐帧用 Python 处理）。

    python3 render.py sample A     # 高潮 10 秒小样（A 克制 / B 炫技），540p 30fps
断点续跑：每个镜头先渲染成 work/cache/<tag>/shot_XXX.mp4，已存在则跳过。
"""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROXY = ROOT / "work" / "proxy_540p.mp4"
MUSIC = ROOT / "input" / "music" / "music.flac"
FPS = 30
CW, CH = 960, 540
LEAD = 0.03  # 切点比拍点提前 30 ms

STYLES = {
    "A": dict(pulse=0.03, rgb=False, shake=0.0, flash=0.20, glitch=False, grain=5, base_zoom=1.0),
    "B": dict(pulse=0.06, rgb=True, shake=0.015, flash=0.40, glitch=True, grain=14, base_zoom=1.03),
}


def speed_expr(mode, dur, src):
    """返回 setpts 表达式，使 src 秒素材恰好铺满 dur 秒。"""
    if mode == "ramp":  # 0.5k → 1k 的速度坡，k 由总长决定
        k = src / (0.75 * dur)
        return f"(2*{dur}*(sqrt(0.25+PTS*TB/({k}*{dur}))-0.5))/TB"
    return f"PTS*{dur / src}"


def shot_filter(sl, st, nframes, period):
    dur = nframes / FPS
    src = sl["src_out"] - sl["src_in"]
    mode = "ramp" if sl["trans"] == "速度坡" else "const"
    P = period
    lvl = sl["lvl"]
    amp = min(st["pulse"], st["pulse"] * lvl / 0.82) if lvl > 0.25 else 0.0
    ph = f"mod(t+{P - LEAD if sl['i'] else 0},{P})"
    z = f"({st['base_zoom']}+{amp}*exp(-9*{ph})"
    if sl["trans"] == "缩放推近":
        z += "+0.12*max(0,1-t/0.2)"
    z += ")"
    sx, sy = st["shake"] * CW, st["shake"] * CH
    shake = lvl > 0.6 and st["shake"] > 0
    cx = f"(iw-{CW})/2" + (f"+{sx * lvl:.2f}*sin(t*61)" if shake else "")
    cy = f"(ih-{CH})/2" + (f"+{sy * lvl:.2f}*cos(t*73)" if shake else "")
    f = [f"[0:v]setpts=PTS-STARTPTS,setpts={speed_expr(mode, dur, src)},fps={FPS},"
         f"trim=end_frame={nframes},setpts=PTS-STARTPTS,split=2[a][b]",
         f"[b]scale={CW}:{CH}:force_original_aspect_ratio=increase,crop={CW}:{CH},boxblur=18:2,"
         f"eq=brightness=-0.22:saturation=0.9[bg]",
         f"[a]scale=-2:{CH}[fg]",
         f"[bg][fg]overlay=x=(W-w)/2:y=0,eq=contrast=1.08:saturation=1.08[c]"]
    chain = [f"scale=w='trunc({CW}*{z}/2)*2':h='trunc({CH}*{z}/2)*2':eval=frame",
             f"crop={CW}:{CH}:x='{cx}':y='{cy}'"]
    if st["rgb"] and lvl > 0.5 and sl["sec"] in ("推进", "高潮"):
        chain.append(f"rgbashift=rh=-5:bh=5:enable='lt({ph},0.083)'")
    if sl.get("glitch") and st["glitch"]:
        chain.append("rgbashift=rh=28:bh=-28:gv=10:enable='lt(t,0.1)'")
        chain.append("noise=alls=60:allf=t:enable='lt(t,0.1)'")
    chain.append("vignette=angle=PI/5:eval=init")
    if lvl > 0.4:
        chain.append(f"noise=alls={st['grain']}:allf=t")
    if "闪白" in sl["trans"]:
        chain.append(f"eq=brightness='{st['flash']}*exp(-16*t)':eval=frame")
    chain.append("format=yuv420p")
    f.append("[c]" + ",".join(chain) + "[v]")
    return ";".join(f)


def render_shot(sl, st, nframes, period, out):
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{sl['src_in']:.3f}", "-t", f"{sl['src_out'] - sl['src_in'] + 0.1:.3f}",
           "-i", str(PROXY), "-filter_complex", shot_filter(sl, st, nframes, period), "-map", "[v]",
           "-frames:v", str(nframes), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
           "-pix_fmt", "yuv420p", "-r", str(FPS), str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"shot {sl['i']}: {r.stderr[-600:]}")


def build_timeline(slots, t_start, total):
    """从 t_start（成片时间）起取 total 秒的镜头，返回 [(slot, 帧数)]，按整帧边界避免漂移。"""
    sel = [s for s in slots if s["t"] + s["beats"] * PERIOD > t_start + 1e-6 and s["t"] < t_start + total - 1e-6]
    bounds = [0.0 if k == 0 else s["t"] - t_start - LEAD for k, s in enumerate(sel)] + [total]
    bounds = [round(b * FPS) for b in bounds]
    return [(s, bounds[k + 1] - bounds[k]) for k, s in enumerate(sel)]


def mux(clips, audio_args, out, list_path):
    list_path.write_text("".join(f"file '{c}'\n" for c in clips))
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), *audio_args,
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "24", "-maxrate", "6M", "-bufsize", "12M",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-600:])


plan = json.load(open(ROOT / "plan.json"))
MP = plan["meta"]["mp"]
PERIOD = MP["period"]


def sample(style, total=10.0):
    st = STYLES[style]
    slots = plan["slots"]
    start_bar = next(s["bar"] for s in slots if s["sec"] == "高潮")
    t_start = next(s["t"] for s in slots if s["bar"] == start_bar)
    tl = build_timeline(slots, t_start, total)
    tag = f"sample_{style}"
    clips = []
    for sl, n in tl:
        o = ROOT / "work" / "cache" / tag / f"shot_{sl['i']:03d}.mp4"
        render_shot(sl, st, n, PERIOD, o)
        clips.append(o)
    a0 = MP["w0"] + t_start
    audio = ["-ss", f"{a0:.3f}", "-t", f"{total}", "-i", str(MUSIC), "-af",
             "loudnorm=I=-10:TP=-2.2:LRA=7,afade=t=out:st=9.4:d=0.6"]
    out = ROOT / "output" / f"{tag}.mp4"
    mux(clips, audio, out, ROOT / "work" / "cache" / tag / "list.txt")
    return out, tl


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        out, tl = sample(sys.argv[2])
        print(out, sum(n for _, n in tl), "frames,", len(tl), "shots")
