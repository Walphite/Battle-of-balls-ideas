"""由 work/plan_v4_stats.json + work/edl.csv 生成 ../docs/PLAN_MONTAGE_V3.md（V3 文档，第 4 版规则：片段不可分割）"""
import json,csv,numpy as np
S=json.load(open('work/plan_v4_stats.json')); E_=list(csv.DictReader(open('work/edl.csv',encoding='utf-8')))
segdesc={'intro':'scan 0.1–0.2 + 暗角 0.3；取景窗 75→100%；BGA 渐显','regular':'scan、暗角；30% 后每 2 拍缩放脉冲 2–5%；60% 后镜头起点 rgb 2f','peak':'scan 0.3、每拍缩放脉冲 3–6%、起点 rgb 3f、抖动 ≤1.5%、noise 0.15、最强起音 rgb 2f（≈1.5/s）、闪白 ≤60%（间隔 ≥4 s）','irregular':'每个 corp 采样点：stutter 3f + rgb 2f；其余最强起音（≈2/s）按能量 rgb 2f / block 3f / tear 4f；每 2 拍缩放脉冲；scan、noise','sparse':'暗角 0.6、noise 0.2、轻去饱和','frag':'28 个碎片起音各触发 block/tear 4f（强者加 stutter 2f）；前 10 个镜头尾 ASCII 0.5 s','after':'tone:cold、暗角 0.45、每 2 拍缩放脉冲 2–4%','outro':'scan 0.4→0.7、noise 0.25→0.55、暗角 0.6、tone:cold、取景窗 100→75%、淡出'}
rows=[]
for a,b,n in [(0,16,'intro'),(16,79,'regular'),(79,100,'peak'),(100,188,'irregular'),(188,202,'sparse'),(202,219.5,'frag'),(219.5,230,'peak'),(230,266,'after'),(266,278.8,'outro')]:
    r=[x for x in E_ if a-0.05<=float(x['t_start'])<b-0.05]
    if not r: continue
    nb=[int(x['beats']) for x in r]; v=[float(x['speed'].split(':')[1]) if x['speed'].startswith('ramp') else float(x['speed']) for x in r]
    rows.append(f"| {a}–{b} | {n} | {len(r)} | {np.mean(nb):.2f} | {min(nb)}–{max(nb)} | {min(v):.3f}–{max(v):.3f} | {segdesc[n]} |")
bga=[]
for i in S['bga_beat_info']:
    hit=[h for h in S['bga_hit_list'] if h[0]==i['event']]
    bga.append(f"| {i['event']} | {i['beat']} | {i['beat_time']} | {i['event_minus_beat_ms']:+d} | {'命中（镜头 #%d，切点−事件 %+d ms）'%(hit[0][1],hit[0][2]) if hit else '未命中 → 事件时刻镜头内重音'} |")
disc='；'.join(f"{k}：{v}" for k,v in S['discard_reasons'].items())
md=f"""# 《球球大作战》× Frums《Credits》(EX) 卡点集锦 · 方案 V3（第 4 版规则：片段不可分割；只出方案，未渲染）

生成：`fxlab/montage/plan_v4.py` → `work/edl.csv` + `work/plan_v4_stats.json`；覆盖项 `fxlab/montage/overrides_v4.json`（keep / drop）；本文由 `plan_v4_doc.py` 生成。沿用第 1 步分析（配乐结构、BGA 偏移 0.160 s、顶部裁切、画幅）。

## 1. 核心规则（已实现）
- 每个片段最多用 1 次、整段播放；首尾各修剪 ≤0.1 s（实际修剪 = (原长 − 所需长)/2，两端对称）；不拆分、不重复、不做取景变体。
- 播放时长 = 整数拍，拍数 ∈ {{4,6,7,8,9,10,12}}；匀速 0.85–1.15×，或击中点速度坡（击中点降到 0.5×、σ=0.15 s，仍是同一段连续画面）。**拍数 4 对所有片段不可行**（最短片段 2.1 s 在 1.15× 下也 ≥5.5 拍）。
- 拍网格：178.96 BPM，全局网格 + 局部相位漂移（±6 s 窗中位数；不规则区 100–188 用两端线性插值，不跟随其不稳定相位）。拍 0 = 0.271 s，拍 830 = 278.487 s；切点提前 10 ms（BGA 命中边界不提前）。
- 顺序：按音频强度弧线做名次匹配（片段强度分 ↔ 每拍强度名次）得到目标时间，收束区 4 段用「收束感」（段内运动前高后低）最强者；带宽 4 的 DP 在此顺序附近决定弃用 / 局部重排 / 拍数，目标 = 变速偏差 + 强度错配 + 相似惩罚（连续相似 >2 禁止）+ BGA 边界奖励 + 段内拍数偏好（峰值 ≤7、铺垫/收束 ≥9，软约束）。

## 2. 结果
| 配乐秒 | 段落 | 片段数 | 平均拍数 | 拍数范围 | 变速范围 | 画面内特效 |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

- 使用 **{S['clips_used']}** 段 / 116；弃用 **{S['clips_discarded']}**（{disc}）。
- 平均拍数 **{S['avg_beats']}**（分布 {S['beats_dist']}）；镜头 {S['shot_len_s']['min']}–{S['shot_len_s']['max']} s，均 {S['shot_len_s']['mean']} s。
- 变速分布 {S['speed_dist']}，均值 {S['speed_mean']}；速度坡 {S['ramps']} 段（匀速已足够凑拍，DP 未选用）。
- 素材利用率 **{S['utilization']*100:.1f}%**（{S['src_used_s']} s ÷ 304.5 s）。最长相似连续 {S['max_similar_run']}；闪白 {S['n_flash']} 次（峰值区，间隔 ≥4 s，≤60%）；ASCII 转场合计 {S['ascii_total']:.1f} s。
- 片段 1 从拍 0（{S['first_clip_start']} s）开始，最后一段在拍 830（{S['last_clip_end']} s）结束；首 0.26 s 与末 0.33 s 只有背景（淡入/淡出中）。

## 3. BGA 切换点对位（±40 ms）
| 事件(s) | 最近拍 | 拍时刻 | 事件−拍 (ms) | 结果 |
|---|---|---|---|---|
{chr(10).join(bga)}

命中 **{S['bga_hits']}/7**。未命中的 5 个各在事件时刻放 ≤4 帧重音（rgb/block 交替）+ BGA 亮度闪 + 边线闪；144.31、187.21 所在镜头尾另加 0.6 s ASCII 转场。

## 4. 需要你决定的问题（规则 7）
1. **BGA 对位**：5.11（+145 ms）、144.31（−103）、187.21（−109）、233.14（−96）离拍点太远，44.57 差 48 ms（阈值 40）。按规则 2（边界必须在拍点）只能命中 2/7。若允许这 5 个边界偏离拍点（相邻两段播放时长各改变 ≤0.11 s，变速仍在范围内）可 7/7 命中；或维持现状（事件时刻镜头内重音）。
2. **峰值区节奏**：最短片段只能 6 拍（2.0 s），峰值区 14 段均 6.93 拍；「卡点感」靠每拍缩放脉冲 + 起音 RGB + 闪白，不靠切碎。
3. **收束区慢放**：受 0.85× 下限限制，收束 4 段实际 0.99–1.03×；慢放感改由扫描线/噪点加重 + 取景窗收缩表达。若想真正慢放，需允许收束区单独放宽到 0.5×。
4. 97、109 默认弃用（整段播放无法满足 ≤0.4 s）；61（与相邻相似）、73（强度最低）被 DP 弃用。可在 overrides_v4.json 指定 keep/drop 后重跑。

## 5. 沿用项
- 整首 0–278.806 s，30 fps；顶部裁切 6% + 取景窗顶部 2.2% 深色边条（第 1/2 步结论：7% 会切到 clip 15/33 的文字）；取景窗按段落 75–100%，B 版上限 88%（BGA 自身两侧 8% 为黑边，已裁掉再铺满）；BGA 色阶拉伸（99.5 百分位→90% 白）+ 冷暗基色，窗外亮度 ≤0.85×窗内；BGA 亮度随起音脉冲 +40%。
- 配色：阴影→#293b4f、高光→#e1decd；强调色 = BGA 字符前景色提亮；无文字、无片尾字幕；只用这首歌，loudnorm −10.5 LUFS / TP ≤ −1 dBTP。
- 光敏：镜头 ≥2 s；闪白 ≤2/s（实际 ≤1/4 s）、≤60%；block/tear/rgb 不改变平均亮度。
- EDL 新字段：`beats`、`beat_start`、`speed`（数值或 `ramp:v0:hit=源秒:dip=0.5:sigma=0.15`）、`bga_event`、`fx` 里 `beatpulse:A%:every`（每 every 拍脉冲）与 `type:Nf@偏移秒` 事件；渲染器 render_v3.py 在第 3 步按此更新。
"""
open('../docs/PLAN_MONTAGE_V3.md','w',encoding='utf-8').write(md); print('doc ok',len(md))
