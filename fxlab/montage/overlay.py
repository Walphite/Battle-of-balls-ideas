"""叠加层接口：在指定时间点和时长，叠加带透明通道的 PNG 序列或 WebM（VP9+alpha），
支持缩放、不透明度和混合模式（normal 正常 / screen 滤色）。

    ov = Overlay("work/fx/spark_%04d.png", start=12.0, dur=1.5, scale=0.8, blend="screen")
    fc, out = filter_graph("[0:v]", [ov], first_input=1, size=(960, 540), fps=30)
    cmd = ["ffmpeg", "-i", base, *input_args(ov), "-filter_complex", fc, "-map", out, ...]

粒子/字幕层以后只需产出 PNG 序列或 WebM，再用本接口合成；此处不含任何粒子实现。
"""
import subprocess
from dataclasses import dataclass


@dataclass
class Overlay:
    path: str                 # WebM 文件，或 PNG 序列模板（含 %04d）
    start: float              # 在成片时间轴上的起点（秒）
    dur: float                # 持续时间（秒）
    scale: float = 1.0        # 相对原始尺寸的缩放
    blend: str = "normal"     # normal | screen
    opacity: float = 1.0
    fps: int = 30             # 仅 PNG 序列用


def input_args(ov):
    if "%" in ov.path:
        return ["-framerate", str(ov.fps), "-i", ov.path]
    if ov.path.lower().endswith(".webm"):
        return ["-c:v", "libvpx-vp9", "-i", ov.path]  # 只有 libvpx-vp9 解码器会读出 alpha
    return ["-i", ov.path]


def filter_graph(base, overlays, first_input, size, fps=30):
    """返回 (filter_complex 字符串, 输出标签)。base 形如 "[0:v]"；叠加层输入从 first_input 开始依次编号。"""
    W, H = size
    parts, cur = [], base
    for k, ov in enumerate(overlays):
        i, a, b = first_input + k, ov.start, ov.start + ov.dur
        lab, out = f"[ov{k}]", f"[o{k}]"
        parts.append(f"[{i}:v]format=rgba,trim=duration={ov.dur},setpts=PTS-STARTPTS+{a}/TB,"
                     f"scale=iw*{ov.scale}:ih*{ov.scale},colorchannelmixer=aa={ov.opacity}{lab}")
        if ov.blend == "normal":
            parts.append(f"{cur}{lab}overlay=x=(W-w)/2:y=(H-h)/2:eof_action=pass:"
                         f"enable='between(t,{a},{b})'{out}")
        elif ov.blend == "screen":
            # 先把带 alpha 的叠加层铺在黑底上（预乘），再与底图做 screen 混合
            parts.append(f"color=c=black:s={W}x{H}:r={fps},format=rgba[blk{k}]")
            parts.append(f"[blk{k}]{lab}overlay=x=(W-w)/2:y=(H-h)/2:eof_action=pass:format=auto,format=gbrp[pm{k}]")
            parts.append(f"{cur}format=gbrp[bs{k}]")
            parts.append(f"[bs{k}][pm{k}]blend=all_mode=screen:shortest=1:enable='between(t,{a},{b})'{out}")
        else:
            raise ValueError("blend 只支持 normal / screen")
        cur = out
    return ";".join(parts), cur


def render(base, overlays, out, size, fps=30, duration=None, extra=()):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", base]
    for ov in overlays:
        cmd += input_args(ov)
    fc, lab = filter_graph("[0:v]", overlays, 1, size, fps)
    cmd += ["-filter_complex", fc, "-map", lab, "-an"]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "26", *extra, out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-800:])


def make_solid_test(path, size=(480, 270), color="red", alpha=0.6, seconds=5, fps=30):
    """生成 5 秒带透明通道的纯色 WebM（测试用）。"""
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"color=c={color}@{alpha}:s={size[0]}x{size[1]}:r={fps}:d={seconds},format=yuva420p",
                        "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "200k", "-auto-alt-ref", "0", path],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-500:])


if __name__ == "__main__":
    import os
    from pathlib import Path
    here = Path(__file__).resolve().parent
    w = here / "work"
    sol = str(w / "test_solid.webm")
    make_solid_test(sol)
    ovs = [Overlay(sol, start=0.5, dur=2.0, scale=0.5, blend="normal"),
           Overlay(sol, start=2.5, dur=2.0, scale=1.0, blend="screen", opacity=0.8)]
    render(str(w / "proxy_540p.mp4"), ovs, str(w / "test_overlay.mp4"), (784, 540), duration=5)
    print("ok", os.path.getsize(w / "test_overlay.mp4"), "bytes")
