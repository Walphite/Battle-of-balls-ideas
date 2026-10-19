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
N_BARS, DROP_BAR = 28, 19
MERGE_PAIRS = [(5, 6), (22, 23), (57, 58), (63, 64), (67, 68), (69, 70), (71, 72)]
EXCLUDE = {17, 37, 49, 59, 73}
SUSPICIOUS = {6, 9, 13, 15, 17, 22, 24, 26, 30, 35, 49, 59, 67, 69, 71, 76, 82, 85}
TRIM = 5                # 0.15s ≈ 5 帧
MAX_DISTINCT = 45
# (起始小节, 结束小节(含), 每小节镜头拍数, 段落)
PATTERNS = [(0, 5, [4], "铺垫"), (6, 13, [2, 2], "推进"), (14, 18, [2, 1, 1], "推进"),
            (19, 21, [1, 1, 1, 1], "高潮"), (22, 23, [1, 1, .5, .5, .5, .5], "高潮"),
            (24, 25, [1, 1, 1, 1], "高潮"), (26, 27, [4], "收尾")]


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
    el = [s for s in segs if not s["excluded"] and not s["suspicious_short"] and s["usable"] >= 0.6]
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
            for L in pat:
                slots.append(dict(bar=bar, beats=L, t=t, sec=sec))
                t += L * p
    for i, s in enumerate(slots):
        s["i"] = i
        s["pos"] = s["t"] / (N_BARS * 4 * p)
    return slots


def assign_transitions(slots):
    last_flash = []
    for s in slots:
        i, sec, L = s["i"], s["sec"], s["beats"]
        if i == 0:
            s["trans"] = "开场淡入（非转场）"
        elif sec == "铺垫":
            s["trans"] = "缩放推近" if i % 2 else "速度坡"
        elif sec == "推进" and L == 2:
            s["trans"] = "硬切+闪白" if i % 2 else "缩放推近"
        elif sec == "推进":
            s["trans"] = "硬切+闪白" if L == 1 and s["i"] % 3 == 1 else "缩放推近"
        elif sec == "高潮":
            if L == .5:
                s["trans"] = "硬切+闪白" if s["i"] % 2 == 0 else "硬切"
            else:
                s["trans"] = "硬切+闪白"
        else:
            s["trans"] = "缩放推近"
    # 光敏安全：任意 1 秒内闪白 ≤3 次（超过则降级为纯硬切）
    ft = []
    for s in slots:
        if "闪白" in s["trans"]:
            ft = [x for x in ft if s["t"] - x < 1.0]
            if len(ft) >= 3:
                s["trans"] = "硬切"
            else:
                ft.append(s["t"])
    return slots


def effects(slots, mp):
    e = np.array(mp["energy"])
    en = (e - e.min()) / (e.max() - e.min() + 1e-9)
    glitch_idx = [next(s["i"] for s in slots if s["sec"] == "高潮"),
                  [s["i"] for s in slots if s["sec"] == "高潮" and s["bar"] == 23][-1]]
    for s in slots:
        mt = mp["w0"] + s["t"]
        ev = float(en[min(int(mt / 0.5), len(en) - 1)])
        arc = {"铺垫": 0.15 + 0.15 * s["pos"] / 0.2, "推进": 0.3 + 0.5 * (s["pos"] - 0.2) / 0.48,
               "高潮": 0.9, "收尾": 0.2}[s["sec"]]
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
    p = mp["period"]
    uses, window_log, order = {}, {}, {}
    for s in slots:
        factor = 0.75 if s["trans"] == "速度坡" else (0.5 if s["sec"] == "收尾" else 1.0)
        s["need"] = s["beats"] * p * factor
        s["speed"] = "0.5×→1×" if s["trans"] == "速度坡" else ("0.5×慢放" if s["sec"] == "收尾" else "1×")
        s["q"] = {"铺垫": 0.12, "推进": 0.35 + 0.35 * (s["pos"] - 0.2) / 0.48, "高潮": 0.9, "收尾": 0.3}[s["sec"]]
    ranks = {s["id"]: s["score"] for s in el}
    thr = np.percentile([similar(a, b) for i, a in enumerate(el) for b in el[i + 1:]], 80)
    placed = {}
    prio = {"高潮": 0, "收尾": 1, "铺垫": 2, "推进": 3}
    for sl in sorted(slots, key=lambda x: (prio[x["sec"]], x["i"])):
        best, bc = None, 1e9
        for sg in el:
            n = uses.get(sg["id"], 0)
            if n >= 2 or sg["usable"] < sl["need"] - 1e-6:
                continue
            if n == 0 and len(uses) >= MAX_DISTINCT:
                continue
            if n == 1 and any(abs(sl["i"] - j) < 6 for j in order[sg["id"]]):
                continue
            c = abs(sg["score"] - sl["q"])
            if sl["sec"] == "铺垫":
                c += 0.5 * sg["fluct"] / (max(e["fluct"] for e in el) + 1e-9)  # 画面干净
            if sl["sec"] == "收尾":
                c += 0.3 * (1 - sg["sat"])  # 收束感：偏好安静、画面简洁
            if n == 1:
                c += 0.25
            for off in (-1, 1):
                nb = placed.get(sl["i"] + off)
                if nb is None:
                    continue
                sim = similar(sg, nb["seg"])
                if sim > thr:
                    c += 0.15
                    nb2 = placed.get(sl["i"] + 2 * off)
                    if nb2 is not None and similar(nb["seg"], nb2["seg"]) > thr:
                        c += 5   # 连续 3 个相似
                if sg["id"] == nb["seg"]["id"]:
                    c += 5
            if c < bc:
                best, bc = sg, c
        if best is None:
            raise SystemExit(f"槽位 {sl['i']} 找不到可用段")
        avoid = window_log.get(best["id"])
        m = int(round(sl["need"] * FPS))
        st, en = best_window(best, m, F, G, D, avoid)
        window_log[best["id"]] = (st, en) if avoid is None else window_log[best["id"]]
        uses[best["id"]] = uses.get(best["id"], 0) + 1
        order.setdefault(best["id"], []).append(sl["i"])
        sl.update(seg=best["id"], src_in=st / FPS, src_out=en / FPS, reuse=uses[best["id"]] == 2, score=round(best["score"], 2))
        placed[sl["i"]] = dict(seg=best)
    return slots, uses, thr


def selected_sheet(slots, el_by_id, F_path=PROXY):
    seen = []
    for s in slots:
        if s["seg"] not in seen:
            seen.append(s["seg"])
    cw, ch, cols = 224, 154, 9
    rows = (len(seen) + cols - 1) // cols
    sheet = Image.new("RGB", (cw * cols, ch * rows), (20, 20, 20))
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    first = {}
    for s in slots:
        first.setdefault(s["seg"], s)
    for k, sid in enumerate(seen):
        s = first[sid]
        t = (s["src_in"] + s["src_out"]) / 2
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(PROXY), "-frames:v", "1",
                            "-f", "image2pipe", "-vcodec", "mjpeg", "-"], capture_output=True)
        im = Image.open(io.BytesIO(p.stdout)).convert("RGB").resize((cw, ch))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 52, 22), fill=(0, 0, 0))
        d.text((4, 3), f"#{sid}", fill=(255, 255, 0), font=font)
        sheet.paste(im, ((k % cols) * cw, (k // cols) * ch))
    out = DOCS / "montage_sheets" / "selected.jpg"
    q = 80
    sheet.save(out, quality=q)
    while out.stat().st_size > 1_000_000 and q > 40:
        q -= 10
        sheet.save(out, quality=q)
    return seen


def main():
    F = load_frames()
    G = F.mean(3)
    D = np.r_[0, np.abs(np.diff(G, axis=0)).mean((1, 2))]
    segs, merge_report = build_segments(F, G, D)
    el = features(segs, F, G, D)
    mp = music_plan()
    slots = effects(assign_transitions(make_slots(mp)), mp)
    slots, uses, thr = assign_segments(slots, el, F, G, D, mp)
    # 方向参考（粗略）
    byid = {s["id"]: s for s in segs}
    for sl in slots:
        v, n = direction(F, G, round(sl["src_in"] * FPS), round(sl["src_out"] * FPS))
        sl["dir"] = ["·", "→", "←", "↓", "↑"][0 if n < 0.3 else (1 if abs(v[0]) >= abs(v[1]) and v[0] > 0 else 2 if abs(v[0]) >= abs(v[1]) else 3 if v[1] > 0 else 4)]
    seen = selected_sheet(slots, byid)
    # 闪白密度检查
    ft = [sl["t"] for sl in slots if "闪白" in sl["trans"]]
    maxflash = max(sum(1 for y in ft if x <= y < x + 1.0) for x in ft)
    # 相似连续检查
    run, maxrun = 1, 1
    for a, b in zip(slots, slots[1:]):
        if similar(byid[a["seg"]], byid[b["seg"]]) > thr:
            run += 1
        else:
            run = 1
        maxrun = max(maxrun, run)
    meta = dict(mp={k: v for k, v in mp.items() if k not in ("energy",)}, merge_report=merge_report,
                n_slots=len(slots), distinct=len(uses), reused=sum(1 for v in uses.values() if v == 2),
                maxflash=maxflash, maxrun=maxrun, sim_thr=float(thr), elig=len(el), merged=[s["id"] for s in segs if "+" in s["id"]])
    json.dump(dict(meta=meta, slots=slots), open(ROOT / "plan.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    write_doc(meta, slots, seen, byid)
    print(json.dumps({k: v for k, v in meta.items() if k != "mp"}, ensure_ascii=False, default=float))
    print({k: v for k, v in meta["mp"].items() if k not in ("energy",)})


def write_doc(meta, slots, seen, byid):
    mp = meta["mp"]
    p = mp["period"]
    D = N_BARS * 4 * p
    L = []
    L.append("# 《球球大作战》卡点花式集锦 · 方案（第 2 步，未渲染）\n")
    L.append(f"成片 **{D:.1f} 秒**（{N_BARS} 小节 × 4 拍），镜头 **{meta['n_slots']}** 个，入选段 **{meta['distinct']}** 个"
             f"（其中 {meta['reused']} 个段在高潮/推进里用了第二个不同窗口）。\n")
    L.append("## 1. 配乐分析\n")
    L.append(f"- 文件：`input/music/music.flac`，时长 161.5 秒。\n"
             f"- **采用网格 {mp['bpm']:.2f} BPM**（拍长 {p * 1000:.0f} ms，t0={mp['t0']:.3f}s，拍点网格残差 {mp['resid_ms']:.1f} ms）。\n"
             f"- ⚠ 速度有歧义：候选网格残差（BPM, ms）= {mp['alt']}；104 与 156 的网格都稳定（比例 2:3），"
             f"其余候选都不稳。我选 104：每镜头更长，镜头总数可控；若你觉得原曲更像 156 的律动，回复即可改，所有拍数按拍长等比缩短、镜头数约增 1.5 倍。\n"
             f"- drop 候选（时间 s, 跃升分）：{mp['drops']}。采用 **{mp['drop_detect']:.2f}s**，"
             f"该处是 4 秒窗口里能量跃升的中心（约 73.0s 起有一段渐强铺垫）；精修到其后第一个宽带强起音 {mp['drop_peak']:.2f}s（即 drop 的第一击），吸附到网格拍 {mp['drop_beat']:.3f}s，对齐误差 **{mp['align_err_ms']:+.0f} ms**。\n"
             f"- 配乐窗口：起点 **{mp['w0']:.3f}s**，终点 {mp['w1']:.3f}s（最后 2 小节渐弱）；drop 落在第 {DROP_BAR} 小节开头，"
             f"即成片 {DROP_BAR * 4 * p:.2f}s / {D:.1f}s = **{DROP_BAR / N_BARS * 100:.1f}%**（目标 65–70%）。\n"
             f"- 切点提前量：每个切点比拍点提前 **30 ms**（在 0–50 ms 内）；音频不动，视频切点整体前移。\n")
    L.append("## 2. 可疑切点处理\n")
    L.append("| 对 | SSIM | 直方图相关 | 结果 |\n|---|---|---|---|")
    for a, b, s, h, ok in mp.get("_", []) or meta["merge_report"]:
        L.append(f"| {a}/{b} | {s} | {h} | {'合并' if ok else '保持分开'} |")
    L.append("\n合并规则：SSIM > 0.9 或颜色直方图相关 > 0.98。⚠ 所有对的 SSIM 都只有 0.37–0.59（没有一对满足 SSIM 条件），"
             "合并完全是靠直方图相关（画面底色相同，直方图本来就容易高），所以这 6 个合并段很可能是「同场景的相邻两个动作」而不是无缝连续；"
             "我已让所有窗口不跨越合并接缝，这样即使合并错了也不会出现跳变。22/23 的相关 0.975 略低于阈值，保持分开。看过 selected.jpg 后可告诉我改回拆开。")
    L.append("排除：#17、#37、#49、#59、#73（半截画面）。其余 <1 秒且在可疑名单内的段一律不入选"
             "（没有任何一个满足「≥0.6 秒且不在可疑名单」的短段，所以半拍镜头改为从正常段里截取 0.29 秒的窗口）。\n")
    L.append("## 3. 选段与排序\n")
    L.append(f"- 可入选段 {meta['elig']} 个；强度分 = 0.5×运动量分位 + 0.25×饱和度分位 + 0.25×帧差波动分位（原片静音，声音项为 0）。\n"
             f"- 每段两端各内缩 0.15 秒；每个镜头取该段运动量最大的窗口。\n"
             f"- 相似判定：色块直方图相关 + 构图相关，阈值取全体两两相似度的 80 分位（{meta['sim_thr']:.2f}）；"
             f"成稿中最长相似连续 **{meta['maxrun']}**（要求 ≤2）。\n"
             f"- 闪白：任意 1 秒内最多 **{meta['maxflash']}** 次（要求 ≤3），闪白峰值设为 60%（≤70%）。\n"
             f"- 入选段联系图：`fxlab/docs/montage_sheets/selected.jpg`（按首次出现顺序）。\n"
             f"- **入选段号（{len(seen)} 个）**：{', '.join(seen)}\n")
    L.append("## 4. 镜头表\n")
    L.append("段号带 `+` 表示合并段；「方向」为中心区域整体运动的粗略方向（·=不明显）；强度 lvl 随配乐能量与弧线连续变化。\n")
    L.append("| # | 小节 | 段落 | 成片时间 | 段号 | 素材入点→出点 (s) | 拍数 | 速度 | 方向 | 强度 | 转场 | 特效 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in slots:
        t0 = s["t"] - 0.03 if s["i"] else 0
        L.append(f"| {s['i'] + 1} | {s['bar'] + 1} | {s['sec']} | {max(t0, 0):.2f} | {s['seg']}{'（二次）' if s['reuse'] else ''} | "
                 f"{s['src_in']:.2f}→{s['src_out']:.2f} | {s['beats']} | {s['speed']} | {s['dir']} | {s['lvl']} | {s['trans']} | {'、'.join(s['fx'])} |")
    L.append("\n## 5. 统一设计\n")
    L.append("- 调色：全片同一套曲线（略提对比、饱和 +8%、暗部轻压），不逐镜头调。\n"
             "- 画幅：16:9 完整画面不裁切，左右用同素材放大模糊压暗版填充；9:16 上下同理。预览 30 帧，终片 60 帧（做不到 ≤90MB 或超时则回退 30 帧并告知）。\n"
             "- 声音：只放配乐（-10~-11 LUFS，真峰值 ≤ -1 dBTP，结尾 2 小节渐弱）；原片静音，故无击中保留音。\n"
             "- HUD：不放任何文字和进度条。\n"
             "- 叠加层接口：`overlay.py`（PNG 序列 / WebM，指定起点、时长、缩放、混合模式：正常/滤色），已用 5 秒纯色测试验证。\n")
    L.append("## 6. 需要你确认的问题\n")
    L.append("1. **速度网格**：104 BPM（本方案）还是 156 BPM？\n"
             f"2. **镜头数 {meta['n_slots']} > 入选段上限 {MAX_DISTINCT}**：1 拍/半拍的高潮在 {D:.0f} 秒里必然超过 45 个镜头，"
             f"所以 {meta['reused']} 个段被用了第二次（取不同窗口、间隔 ≥6 个镜头）。接受，还是改成缩短高潮的 1 拍区？\n"
             "3. 成片时长 64.6 秒，在 60–90 秒的偏短端（再长会让镜头数更多、复用更多），接受？\n"
             "4. 有无必须保留/排除的段号？看过 selected.jpg 后告诉我。\n")
    (DOCS / "PLAN_MONTAGE.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
