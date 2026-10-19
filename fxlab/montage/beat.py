"""配乐分析模块（纯 ffmpeg + numpy）：BPM、拍点、能量曲线、drop 检测。"""
import subprocess
import numpy as np

SR = 22050
HOP = 512
NFFT = 2048


def load_mono(path, sr=SR):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                        "-f", "f32le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.float32)


def stft_mag(x):
    win = np.hanning(NFFT)
    n = 1 + (len(x) - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(x[idx] * win, axis=1))


def onset_env(mag):
    """对数谱的正向谱通量。"""
    lm = np.log1p(10 * mag)
    flux = np.maximum(np.diff(lm, axis=0), 0).mean(1)
    flux = np.r_[0, flux]
    return (flux - flux.mean()) / (flux.std() + 1e-9)


def tempo(env, fps, lo=80, hi=180):
    """梳状滤波估计 BPM：对每个候选速度、每个相位取节拍位置的平均起音强度，取最大。"""
    best, bs = None, -1e9
    n = len(env)
    for bpm in np.arange(lo, hi, 0.5):
        period = 60 / bpm * fps
        m = int(n / period) - 1
        k = np.arange(m)
        top = -1e9
        for ph in np.arange(0, period, 1.0):
            idx = np.round(ph + k * period).astype(int)
            top = max(top, env[idx].mean())
        if top > bs:
            best, bs = bpm, top
    return best


def beat_track(env, fps, bpm, tight=100.0):
    """Ellis 动态规划拍点追踪。返回拍点时间（秒）。"""
    period = 60 / bpm * fps
    n = len(env)
    score = env.copy()
    back = -np.ones(n, int)
    win = np.arange(-int(2 * period), -int(period / 2))
    pen = -tight * (np.log(-win / period)) ** 2
    cum = score.copy()
    for i in range(n):
        j = i + win
        ok = j >= 0
        if ok.any():
            c = cum[j[ok]] + pen[ok]
            k = np.argmax(c)
            cum[i] = score[i] + c[k]
            back[i] = j[ok][k]
    i = int(np.argmax(cum[-int(period):]) + n - int(period))
    beats = [i]
    while back[i] >= 0:
        i = back[i]
        beats.append(i)
    return np.array(beats[::-1]) / fps


def fit_grid(beats, bpm):
    """用线性拟合把拍点规整成等间隔网格：t = t0 + k*period。返回 (t0, period)。"""
    k = np.round((beats - beats[0]) / (60 / bpm))
    p, t0 = np.polyfit(k, beats, 1)
    return float(t0), float(p)


def energy_curve(x, sr=SR, win=0.5):
    n = int(sr * win)
    k = len(x) // n
    e = np.sqrt((x[: k * n].reshape(k, n) ** 2).mean(1))
    return e, win


def lowband_energy(mag, sr=SR, fmax=200):
    bins = int(fmax / (sr / NFFT))
    return mag[:, :bins].mean(1)


def find_drop(x, sr=SR, span=4.0):
    """drop = 低频+整体能量在 span 秒内最大的跃升。返回候选 [(时间, 跃升)]（降序）。"""
    mag = stft_mag(x)
    fps = sr / HOP
    e = np.log1p(10 * mag).mean(1) + np.log1p(10 * lowband_energy(mag))
    w = int(span * fps)
    sm = np.convolve(e, np.ones(w) / w, "same")
    c = np.zeros(len(e))
    for i in range(w, len(e) - w):
        c[i] = sm[i + w // 2] - sm[i - w // 2]
    out, taken = [], []
    for i in np.argsort(c)[::-1]:
        t = i / fps
        if all(abs(t - u) > 8 for u in taken):
            out.append((t, float(c[i])))
            taken.append(t)
        if len(out) >= 5:
            break
    return out, c, fps


def best_grid(env, fps, lo=80, hi=180):
    """候选速度（梳状峰值及其 ×2/×½/×1.5/×⅔）逐个做拍点追踪，取网格残差最小者。
    返回 dict(bpm, t0, period, beats, resid_ms, tried)。"""
    seeds = set()
    n = len(env)
    scores = []
    for bpm in np.arange(lo, hi, 1.0):
        period = 60 / bpm * fps
        k = np.arange(int(n / period) - 1)
        top = max(env[np.round(ph + k * period).astype(int)].mean() for ph in np.arange(0, period, 2.0))
        scores.append((top, bpm))
    for _, b in sorted(scores, reverse=True)[:6]:
        for r in (1, 2, 0.5, 1.5, 2 / 3):
            if lo <= b * r <= hi:
                seeds.add(round(b * r))
    tried, best = [], None
    for b in sorted(seeds):
        beats = beat_track(env, fps, b, tight=400)
        t0, p = fit_grid(beats, b)
        res = float(np.abs(beats - (t0 + p * np.round((beats - t0) / p))).mean() * 1000)
        tried.append((b, round(res, 1)))
        if best is None or res < best["resid_ms"]:
            best = dict(bpm=60 / p, t0=t0, period=p, beats=beats, resid_ms=res)
    best["tried"] = tried
    return best
