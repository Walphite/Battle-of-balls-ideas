#!/usr/bin/env python3
"""第 2 步：只出方案（不渲染）。输出 plan.json、fxlab/docs/PLAN_MONTAGE.md、fxlab/docs/montage_sheets/selected.jpg。"""
import csv, io, json, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat

ROOT = Path(__file__).resolve().parent
PROXY = ROOT / "work" / "proxy_540p.mp4"
MUSIC = ROOT / "input" / "music" / "music.flac"
DOCS = ROOT.parent / "docs"
FPS = 30
W, H = 128, 72
BPM_GRID = 104          # 稳定网格（见方案：104 与 156 都稳定，二者为 2:3 关系）
N_BARS, DROP_BAR = 40, 27
MERGE_PAIRS = [(5, 6), (22, 23), (57, 58), (63, 64), (67, 68), (69, 70), (71, 72)]
EXCLUDE = {17, 37, 49, 59, 73}
SUSPICIOUS = {6, 9, 13, 15, 17, 22, 24, 26, 30, 35, 49, 59, 67, 69, 71, 76, 82, 85}
TRIM = 5                # 0.15s ≈ 5 帧
# (起始小节, 结束小节(含), 每小节镜头拍数, 段落)
PATTERNS = [(0, 9, [4], "铺垫"), (10, 19, [2, 2], "推进"), (20, 26, [2, 1, 1], "推进"),
            (27, 31, [1, 1, 1, 1], "高潮"), (32, 33, [1, 1, .5, .5, .5, .5], "高潮"),
            (34, 37, [1, 1, 1, 1], "高潮"), (38, 39, [4], "收尾")]


# ---------- 画面 ----------
def load_frames():
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(PROXY), "-vf", f"fps={FPS},scale={W}:{H}",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, np.uint8)
    return a[: a.size // (W * H * 3) * W * H * 3].reshape(-1, H, W, 3)


def ssim(a, b, blk=8):
    a = a.astype(float); b = b.astype(float)
    h, w = a.shape[0] // blk * blk, a.shape[1] // blk * blk
    def bl(x):
        return x[:h, :w].reshape(h // blk, blk, w // blk, blk).transpose(0, 2, 1, 3).reshape(-1, blk * blk)
    A, B = bl(a), bl(b)
    ma, mb = A.mean(1), B.mean(1)
    va, vb = A.var(1), B.var(1)
    cov = ((A - ma[:, None]) * (B - mb[:, None])).mean(1)
    c1, c2 = 6.5025, 58.5225
    return float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def hist(f):
    q = (f // 64).reshape(-1, 3)
    h = np.bincount(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], minlength=64).astype(float)
    return h / h.sum()


def hcorr(a, b):
    return float(np.corrcoef(hist(a), hist(b))[0, 1])


def saturation(f):
    mx, mn = f.max(-1).astype(float), f.min(-1).astype(float)
    return (mx - mn) / (mx + 1e-6)


def pct_rank(v):
    o = np.argsort(np.argsort(v))
    return o / max(len(v) - 1, 1)


def global_shift(g0, g1):
    """中心区域相位相关，返回 (dx, dy) 像素位移（粗略，仅作方向参考）。"""
    c0, c1 = g0[14:58, 26:102], g1[14:58, 26:102]
    f = np.fft.fft2(c0 - c0.mean()) * np.conj(np.fft.fft2(c1 - c1.mean()))
    r = np.fft.ifft2(f / (np.abs(f) + 1e-9)).real
    y, x = np.unravel_index(np.argmax(r), r.shape)
    if y > r.shape[0] // 2: y -= r.shape[0]
    if x > r.shape[1] // 2: x -= r.shape[1]
    return x, y


# ---------- 段 ----------
def build_segments(F, G, D):
    rows = list(csv.DictReader(open(ROOT / "work" / "segments.csv", encoding="utf-8-sig")))
    seg = {int(r["seg"]): [round(float(r["start"]) * FPS), round(float(r["end"]) * FPS)] for r in rows}
    merge_report, merged_into, joins = [], {}, {}
    for a, b in MERGE_PAIRS:
        fa, fb = F[seg[a][1] - 1], F[seg[b][0]]
        s, hc = ssim(G[seg[a][1] - 1], G[seg[b][0]]), hcorr(fa, fb)
        ok = s > 0.9 or hc > 0.98
        merge_report.append((a, b, round(s, 3), round(hc, 3), ok))
        if ok:
            merged_into[b] = a
            joins[a] = seg[a][1]
    out = []
    for k in sorted(seg):
        if k in merged_into:
            continue
        a0, b0 = seg[k]
        name = str(k)
        j = []
        nxt = k
        while nxt in joins and nxt + 1 in merged_into:
            j.append(seg[nxt][1])
            nxt += 1
            b0 = seg[nxt][1]
            name += f"+{nxt}"
        ids = list(range(k, nxt + 1))
        out.append(dict(id=name, members=ids, a=a0, b=b0, joins=j))
    return out, merge_report


def features(segs, F, G, D):
    for s in segs:
        a, b = s["a"], s["b"]
        s["dur"] = (b - a) / FPS
        u0, u1 = a + TRIM, b - TRIM
        s["u0"], s["u1"] = u0, u1
        s["usable"] = max(u1 - u0, 0) / FPS
        mask = np.ones(len(D), bool)
        mask[a:a + 3] = False
        for j in s["joins"]:
            mask[j:j + 3] = False
        idx = [i for i in range(u0, u1) if mask[i]]
        d = D[idx] if idx else np.array([0.0])
        s["mot"] = float(d.mean())
        s["fluct"] = float(d.std())
        s["sat"] = float(np.mean([saturation(F[i]).mean() for i in range(u0, u1, 3)])) if u1 > u0 else 0
        mid = (u0 + u1) // 2
        s["hist"] = hist(F[mid])
        s["comp"] = G[mid].reshape(4, 18, 8, 16).mean((1, 3)).ravel()
        s["comp"] = (s["comp"] - s["comp"].mean()) / (s["comp"].std() + 1e-6)
        s["color_h"] = s["hist"]
        s["excluded"] = any(m in EXCLUDE for m in s["members"])
        s["suspicious_short"] = (s["dur"] < 1.0) and any(m in SUSPICIOUS for m in s["members"])
    el = [s for s in segs if not s["excluded"] and s["dur"] >= 1.0]
    r_m = pct_rank([s["mot"] for s in el]); r_s = pct_rank([s["sat"] for s in el]); r_f = pct_rank([s["fluct"] for s in el])
    for s, m, sa, f in zip(el, r_m, r_s, r_f):
        s["score"] = float(0.5 * m + 0.25 * sa + 0.25 * f)
    return el


def similar(a, b):
    return 0.5 * float(np.corrcoef(a["hist"], b["hist"])[0, 1]) + 0.5 * float(np.corrcoef(a["comp"], b["comp"])[0, 1])


def best_window(s, m, F, G, D, avoid=None):
    """在可用区间内选长度 m 帧、运动量之和最大的窗口；avoid=(a,b) 为需避开的已用窗口。"""
    u0, u1 = s["u0"], s["u1"]
    m = int(min(m, u1 - u0))
    best, bs = None, -1
    sm = np.convolve(D, np.ones(3) / 3, "same")
    for st in range(u0, u1 - m + 1):
        if avoid and not (st + m <= avoid[0] or st >= avoid[1]):
            continue
        if any(st < j < st + m for j in s["joins"]):
            continue  # 窗口不跨越合并接缝
        v = sm[st:st + m].mean()
        if v > bs:
            best, bs = st, v
    if best is None:
        best = u0
    return best, best + m


def direction(F, G, a, b):
    xs = [global_shift(G[i], G[i + 3]) for i in range(a, max(a + 1, b - 3), 3)]
    v = np.mean(xs, 0) if xs else np.zeros(2)
    n = np.linalg.norm(v)
    return (v / n if n > 0.3 else np.zeros(2)), float(n)


# ---------- 配乐 ----------
def music_plan():
    x = beat.load_mono(MUSIC)
    mag = beat.stft_mag(x)
    fps = beat.SR / beat.HOP
    env = beat.onset_env(mag)
    alt = beat.best_grid(env, fps)
    b = beat.beat_track(env, fps, BPM_GRID, tight=400)
    t0, p = beat.fit_grid(b, BPM_GRID)
    resid = float(np.abs(b - (t0 + p * np.round((b - t0) / p))).mean() * 1000)
    drops, _, _ = beat.find_drop(x)
    dt = drops[0][0]
    lo, hi = int((dt - 1.0) * fps), int((dt + 1.0) * fps)
    peak = (lo + int(np.argmax(env[lo:hi]))) / fps  # 宽带谱通量峰 = 重新进入的第一击
    k = round((peak - t0) / p)
    drop_beat = t0 + k * p
    w0 = drop_beat - DROP_BAR * 4 * p
    e, wn = beat.energy_curve(x)
    return dict(bpm=60 / p, period=p, t0=t0, resid_ms=resid, drops=[(round(float(t), 2), round(float(v), 2)) for t, v in drops],
                drop_detect=float(dt), drop_peak=float(peak), drop_beat=float(drop_beat),
                align_err_ms=float((peak - drop_beat) * 1000), w0=float(w0),
                w1=float(w0 + N_BARS * 4 * p), energy=e.tolist(), alt=alt["tried"],
                alt_bpm=60 / alt["period"])


# ---------- 镜头表 ----------
def make_slots(mp):
    p = mp["period"]
    slots, t = [], 0.0
    for b0, b1, pat, sec in PATTERNS:
        for bar in range(b0, b1 + 1):
            for k, L in enumerate(pat):
                slots.append(dict(bar=bar, beats=L, t=t, sec=sec, k_in_bar=k))
                t += L * p
    for i, s in enumerate(slots):
        s["i"] = i
        s["pos"] = s["t"] / (N_BARS * 4 * p)
    return slots


def assign_transitions(slots):
    """转场：硬切+闪白 / 缩放推近 / 速度坡。闪白任意 1 秒内 ≤2 次；高潮主要靠缩放脉冲和色差。"""
    for s in slots:
        i, sec, L = s["i"], s["sec"], s["beats"]
        if i == 0:
            s["trans"] = "开场淡入（非转场）"
        elif sec == "铺垫":
            s["trans"] = "缩放推近" if i % 2 else "速度坡"
        elif sec == "推进" and L == 2:
            s["trans"] = "硬切+闪白" if i % 4 == 1 else ("缩放推近" if i % 2 else "速度坡")
        elif sec == "推进":
            s["trans"] = "硬切+闪白" if i % 3 == 1 else "缩放推近"
        elif sec == "高潮":
            k = s["k_in_bar"]
            s["trans"] = "硬切+闪白" if (L >= 1 and k % 2 == 0 and s["beats"] == 1) or k == 0 else "缩放推近" if s["beats"] == 1 else "硬切"
        else:
            s["trans"] = "缩放推近"
    ft = []
    for s in slots:
        if "闪白" in s["trans"]:
            ft = [x for x in ft if s["t"] - x < 1.0]
            if len(ft) >= 2:
                s["trans"] = "硬切"
            else:
                ft.append(s["t"])
    return slots


def effects(slots, mp):
    e = np.array(mp["energy"])
    en = (e - e.min()) / (e.max() - e.min() + 1e-9)
    bounds = {}
    for s in slots:
        lo, hi = bounds.get(s["sec"], (1e9, -1e9))
        bounds[s["sec"]] = (min(lo, s["pos"]), max(hi, s["pos"]))
    hi_bars = [s for s in slots if s["sec"] == "高潮"]
    glitch_idx = [hi_bars[0]["i"], [s["i"] for s in hi_bars if s["bar"] == 33][-1]]
    for s in slots:
        mt = mp["w0"] + s["t"]
        ev = float(en[min(int(mt / 0.5), len(en) - 1)])
        lo, hi = bounds[s["sec"]]
        u = (s["pos"] - lo) / max(hi - lo, 1e-9)
        arc = {"铺垫": 0.15 + 0.15 * u, "推进": 0.3 + 0.5 * u, "高潮": 0.9, "收尾": 0.2}[s["sec"]]
        lvl = float(np.clip(0.6 * arc + 0.4 * ev, 0, 1))
        s["lvl"] = round(lvl, 2)
        fx = ["暗角"]
        if lvl > 0.25:
            fx.append(f"节拍缩放脉冲{6 * lvl:.1f}%")
        if lvl > 0.4:
            fx.append("胶片颗粒")
        if lvl > 0.5 and s["sec"] in ("推进", "高潮"):
            fx.append("击中帧RGB色差2–3帧")
        if lvl > 0.6 and s["sec"] in ("推进", "高潮"):
            fx.append(f"轻抖{1.5 * lvl:.1f}%")
        if s["i"] in glitch_idx:
            fx.append("故障")
        s["fx"] = fx
        s["glitch"] = s["i"] in glitch_idx
    return slots


def assign_segments(slots, el, F, G, D, mp):
    """先用尽每段的第 1 个窗口（受限最多的段先挑槽位），再补第 2 个不重叠窗口（与第 1 次间隔 ≥8 个镜头）。"""
    p = mp["period"]
    fmax = max(e["fluct"] for e in el) + 1e-9
    for s in slots:
        factor = 0.75 if s["trans"] == "速度坡" else (0.5 if s["sec"] == "收尾" else 1.0)
        s["need"] = s["beats"] * p * factor
        s["speed"] = "0.5×→1×" if s["trans"] == "速度坡" else ("0.5×慢放" if s["sec"] == "收尾" else "1×")
        lo = min(x["pos"] for x in slots if x["sec"] == s["sec"]); hi = max(x["pos"] for x in slots if x["sec"] == s["sec"])
        u = (s["pos"] - lo) / max(hi - lo, 1e-9)
        s["q"] = {"铺垫": 0.12, "推进": 0.35 + 0.35 * u, "高潮": 0.9, "收尾": 0.3}[s["sec"]]
    thr = np.percentile([similar(a, b) for i, a in enumerate(el) for b in el[i + 1:]], 80)
    placed, taken = {}, {}   # 槽位 -> seg；seg -> [槽位]
    wins = {}                # seg -> [(a,b)]

    def cost(sg, sl, second):
        c = abs(sg["score"] - sl["q"])
        if sl["sec"] == "铺垫":
            c += 0.5 * sg["fluct"] / fmax
        if sl["sec"] == "收尾":
            c += 0.3 * (1 - sg["sat"])
        for off in (-1, 1):
            nb = placed.get(sl["i"] + off)
            if nb is None:
                continue
            if sg["id"] == nb["id"]:
                c += 50
            if similar(sg, nb) > thr:
                c += 0.15
                nb2 = placed.get(sl["i"] + 2 * off)
                if nb2 is not None and similar(nb, nb2) > thr:
                    c += 5
        return c

    def put(sg, sl):
        m = int(round(sl["need"] * FPS))
        st, en = best_window(sg, m, F, G, D, avoid=None if sg["id"] not in wins else wins[sg["id"]][0])
        wins.setdefault(sg["id"], []).append((st, en))
        taken.setdefault(sg["id"], []).append(sl["i"])
        placed[sl["i"]] = sg
        sl.update(seg=sg["id"], src_in=st / FPS, src_out=en / FPS, reuse=len(taken[sg["id"]]) == 2,
                  score=round(sg["score"], 2))

    free = {sl["i"]: sl for sl in slots}
    # 第 1 轮：每段一个窗口。先放能填 4 拍槽的长段，再按可用时长从短到长。
    long_need = max(sl["need"] for sl in slots)
    four = [sl for sl in slots if sl["beats"] == 4]
    order = sorted(el, key=lambda g: g["usable"])
    for sg in order:
        cand = [sl for sl in free.values() if sg["usable"] >= sl["need"] - 1e-6]
        if not cand:
            raise SystemExit(f"段 {sg['id']} 可用 {sg['usable']:.2f}s，放不进任何槽位")
        sl = min(cand, key=lambda x: cost(sg, x, False) + (0.6 if x["beats"] != 4 and sg["usable"] > 2.5 and
                 any(f["beats"] == 4 for f in free.values()) else 0))
        put(sg, sl)
        del free[sl["i"]]
    # 第 2 轮：剩余槽位补第 2 个不重叠窗口
    for sl in sorted(free.values(), key=lambda x: (-x["need"], x["i"])):
        best, bc = None, 1e9
        for sg in el:
            if len(taken[sg["id"]]) >= 2:
                continue
            a, b = wins[sg["id"]][0]
            room = max(a - sg["u0"], sg["u1"] - b) / FPS
            if room < sl["need"] - 1e-6:
                continue
            if any(abs(sl["i"] - j) < 8 for j in taken[sg["id"]]):
                continue
            c = cost(sg, sl, True)
            if c < bc:
                best, bc = sg, c
        if best is None:
            raise SystemExit(f"槽位 {sl['i']}（{sl['beats']} 拍）补不上第 2 个窗口")
        put(best, sl)
    return slots, {k: len(v) for k, v in taken.items()}, thr


def contact_sheet(items, out, cols=9, cw=224, ch=154):
    """items: [(标签, 取帧时间秒)]；保存为 ≤1MB 的 JPEG。"""
    rows = max((len(items) + cols - 1) // cols, 1)
    sheet = Image.new("RGB", (cw * cols, ch * rows), (20, 20, 20))
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    for k, (label, t) in enumerate(items):
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(PROXY), "-frames:v", "1",
                            "-f", "image2pipe", "-vcodec", "mjpeg", "-"], capture_output=True)
        im = Image.open(io.BytesIO(p.stdout)).convert("RGB").resize((cw, ch))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 14 + 9 * len(label), 22), fill=(0, 0, 0))
        d.text((4, 3), f"#{label}", fill=(255, 255, 0), font=font)
        sheet.paste(im, ((k % cols) * cw, (k // cols) * ch))
    q = 80
    sheet.save(out, quality=q)
    while out.stat().st_size > 1_000_000 and q > 30:
        q -= 10
        sheet.save(out, quality=q)


def main():
    F = load_frames()
    G = F.mean(3)
    D = np.r_[0, np.abs(np.diff(G, axis=0)).mean((1, 2))]
    segs, merge_report = build_segments(F, G, D)
    el = features(segs, F, G, D)
    mp = music_plan()
    slots = effects(assign_transitions(make_slots(mp)), mp)
    slots, uses, thr = assign_segments(slots, el, F, G, D, mp)
    byid = {s["id"]: s for s in segs}
    for sl in slots:
        v, n = direction(F, G, round(sl["src_in"] * FPS), round(sl["src_out"] * FPS))
        sl["dir"] = "·" if n < 0.3 else ("→" if v[0] > 0 else "←") if abs(v[0]) >= abs(v[1]) else ("↓" if v[1] > 0 else "↑")
    seen = []
    first = {}
    for s in slots:
        if s["seg"] not in seen:
            seen.append(s["seg"])
            first[s["seg"]] = s
    contact_sheet([(i, (first[i]["src_in"] + first[i]["src_out"]) / 2) for i in seen],
                  DOCS / "montage_sheets" / "selected.jpg", cols=10)
    unused = [s for s in segs if s["id"] not in first]
    reasons = {s["id"]: ("排除名单" if s["excluded"] else "时长<1秒") for s in unused}
    contact_sheet([(s["id"], (s["a"] + s["b"]) / 2 / FPS) for s in unused],
                  DOCS / "montage_sheets" / "unused.jpg", cols=8)
    ft = [sl["t"] for sl in slots if "闪白" in sl["trans"]]
    maxflash = max(sum(1 for y in ft if x <= y < x + 1.0) for x in ft)
    run, maxrun = 1, 1
    for a_, b_ in zip(slots, slots[1:]):
        run = run + 1 if similar(byid[a_["seg"]], byid[b_["seg"]]) > thr else 1
        maxrun = max(maxrun, run)
    used_src = sum(sl["src_out"] - sl["src_in"] for sl in slots)
    total = len(F) / FPS
    gap_ok = all(abs(a_ - b_) >= 8 for v in [[sl["i"] for sl in slots if sl["seg"] == sid] for sid in seen]
                 if len(v) == 2 for a_, b_ in [(v[0], v[1])])
    meta = dict(mp={k: v for k, v in mp.items() if k not in ("energy",)}, merge_report=merge_report,
                n_slots=len(slots), distinct=len(uses), reused=sum(1 for v in uses.values() if v == 2),
                util=used_src / total, used_src=used_src, total_src=total, maxflash=maxflash, maxrun=maxrun,
                gap_ok=gap_ok, sim_thr=float(thr), elig=len(el),
                merged=[s["id"] for s in segs if "+" in s["id"]],
                unused=[(s["id"], reasons[s["id"]], round(s["dur"], 2)) for s in unused],
                dmax=float(mp["drop_beat"] / 0.65))
    json.dump(dict(meta=meta, slots=slots), open(ROOT / "plan.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    write_doc(meta, slots, seen, byid)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("mp", "merge_report")}, ensure_ascii=False, default=float))
    print({k: v for k, v in meta["mp"].items() if k not in ("energy",)})


def write_doc(meta, slots, seen, byid):
    mp = meta["mp"]
    p = mp["period"]
    Dd = N_BARS * 4 * p
    L = []
    L.append("# 《球球大作战》卡点花式集锦 · 方案（已按第 2 次决定更新，未渲染）\n")
    L.append(f"成片 **{Dd:.1f} 秒**（{N_BARS} 小节 × 4 拍），镜头 **{meta['n_slots']}** 个，使用不同段 **{meta['distinct']}** 个，"
             f"第 2 个窗口 **{meta['reused']}** 次，素材利用率 **{meta['util'] * 100:.0f}%**"
             f"（成片取用素材 {meta['used_src']:.1f}s ÷ 原片 {meta['total_src']:.1f}s）。\n")
    L.append("## 1. 配乐窗口\n")
    L.append(f"- 网格 {mp['bpm']:.2f} BPM（拍长 {p * 1000:.0f} ms，残差 {mp['resid_ms']:.1f} ms）；drop 第一击 {mp['drop_beat']:.3f}s，对齐误差 {mp['align_err_ms']:+.0f} ms。\n"
             f"- 可用窗口长度：配乐共 161.5s，drop 落在 65–70% 时，起点不能早于 0，所以**最长约 {meta['dmax']:.1f}s**（drop 在 65%），"
             f"85–95s 全部放得下。\n"
             f"- 采用窗口：起点 **{mp['w0']:.3f}s**，终点 {mp['w1']:.3f}s，drop 在第 {DROP_BAR} 小节开头，"
             f"即成片 {DROP_BAR * 4 * p:.2f}s / {Dd:.1f}s = **{DROP_BAR / N_BARS * 100:.1f}%**；最后 2 小节渐弱。切点比拍点提前 30 ms。\n")
    L.append("## 2. 选段规则（已更新）\n")
    L.append(f"- 可用段 = 时长 ≥1 秒且不在排除名单（#17、#37、#49、#59、#73），共 **{meta['elig']}** 个，全部入选；"
             f"每段最多 2 个不重叠窗口，同一段两次出现间隔 ≥8 个镜头（检查：{'通过' if meta['gap_ok'] else '未通过'}）。先用尽每段的第 1 个窗口，再补第 2 个。\n"
             f"- 7 对合并保持原决定：{'、'.join(meta['merged'])} 已合并，22/23 分开；窗口不跨合并接缝。\n"
             f"- 闪白任意 1 秒内最多 **{meta['maxflash']}** 次（要求 ≤2），峰值 ≤60%；高潮靠缩放脉冲和色差。"
             f"最长相似连续 **{meta['maxrun']}**（要求 ≤2）。\n"
             f"- 联系图：`fxlab/docs/montage_sheets/selected.jpg`（{len(seen)} 个）、`unused.jpg`（{len(meta['unused'])} 个）。\n"
             f"- **未使用的段号（{len(meta['unused'])}）**：{', '.join(f'{a}（{r}，{d}s）' for a, r, d in meta['unused'])}\n")
    L.append(f"- 入选段号（{len(seen)} 个，按首次出现顺序）：{', '.join(seen)}\n")
    L.append("## 3. 镜头表\n")
    L.append("「二次」= 该段第 2 个窗口；「方向」为中心区域整体运动的粗略方向（·=不明显）；强度 lvl 随配乐能量与弧线连续变化。\n")
    L.append("| # | 小节 | 段落 | 成片时间 | 段号 | 素材入点→出点 (s) | 拍数 | 速度 | 方向 | 强度 | 转场 | 特效 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in slots:
        t0 = s["t"] - 0.03 if s["i"] else 0
        L.append(f"| {s['i'] + 1} | {s['bar'] + 1} | {s['sec']} | {max(t0, 0):.2f} | {s['seg']}{'（二次）' if s['reuse'] else ''} | "
                 f"{s['src_in']:.2f}→{s['src_out']:.2f} | {s['beats']} | {s['speed']} | {s['dir']} | {s['lvl']} | {s['trans']} | {'、'.join(s['fx'])} |")
    L.append("\n## 4. 统一设计\n")
    L.append("- 调色：全片同一套曲线（略提对比、饱和 +8%）。画幅：16:9 完整画面不裁切，左右用同素材放大模糊压暗版填充；9:16 上下同理。预览 30 帧，终片 60 帧（做不到 ≤90MB 则回退 30 帧并告知）。\n"
             "- 声音：只放配乐（-10~-11 LUFS，真峰值 ≤ -1 dBTP）；原片静音。HUD：无文字无进度条。叠加层接口见 `overlay.py`。\n")
    (DOCS / "PLAN_MONTAGE.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
