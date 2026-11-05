"""由 work/plan_v3_stats.json + work/edl.csv 生成 ../docs/PLAN_MONTAGE_V3.md"""
import json,csv,numpy as np
S=json.load(open('work/plan_v3_stats.json')); VO=json.load(open('work/vocal_onsets.json')); AC=json.load(open('work/bga_accent.json'))
E_=list(csv.DictReader(open('work/edl.csv',encoding='utf-8')))
seg_rows=[]
for a,b,n,desc in [(0,16,'intro 铺垫','长镜头 12–16 八分拍、0.75× 慢放、取景窗 75→100%、scan 0.1–0.2 + 暗角、BGA 渐显'),(16,79,'regular 规则区','8→4→2 八分拍（按时间+能量）；≥1.2s 镜头 0.75×；scan→zoompulse(≤6%)→rgb 2f 逐级'),(79,100,'peak 峰值 1','2/1 八分拍（30fps 下不用半八分拍）；满屏；全套：zoompulse、rgb 3f、shake≤1.5%、noise；小节首拍闪白（≤60%）'),(100,188,'irregular Corporation','采样点 + 最强 80 个起音切；簇内 5–12 帧；簇外镜头在最强起音处加 zoompulse/rgb；低密度段 ≥0.5s 慢放；>2s 空档按 8 八分拍补切(0.5×)；每 ≤11s 一个呼吸镜头'),(188,202,'sparse 回落','8 八分拍、0.75×、暗角 0.6、noise 0.2、轻去饱和'),(202,219.5,'frag 碎片采样','28 个碎片起音各一次硬切 + block/tear 4f；空档按 4 八分拍补切；ASCII 转场 10×0.5s'),(219.5,230,'peak 峰值 2','同峰值 1'),(230,266,'after 余波','4–8 八分拍，整段 0.5×；每 4 小节 ≤1 次 ≤1s 速度坡 0.5→1→0.5；tone:cold'),(266,278.8,'outro 收束','3 个长镜头 0.5×、取景窗 100→75%、scan 0.4→0.7、noise 0.25→0.55、淡出')]:
    rows=[r for r in E_ if a<=float(r['t_start'])<b]; d=[float(r['dur_ms'])/1000 for r in rows]
    seg_rows.append(f"| {a}–{b} | {n} | {len(rows)} | {np.mean(d):.2f} | {min(d):.2f}–{max(d):.2f} | {desc} |")
corp=VO['corp_100_188']
md=f"""# 《球球大作战》× Frums《Credits》(EX) 卡点集锦 · 方案 V3（已按确认项更新，未渲染成片）

生成：`fxlab/montage/plan_v3.py` → `work/edl.csv`（可手改）+ `work/plan_v3_stats.json`；本文由 `plan_v3_doc.py` 生成。

## 1. 窗口与对齐
- 整首配乐 0–278.806 s，不裁；成片时长 = 配乐时长；**成片与预览均 30 fps**（大量慢放）。开头 1.5 s 淡入（画面+音频），结尾最后 2 小节（2.68 s）淡出。
- 网格：178.96 BPM，拍长 335.3 ms，八分拍 q = 167.6 ms，t0 = 0.221 s；各段相位偏移：铺垫 +44 ms、规则 +39、峰值 +31、不规则 +22、回落/碎片 +3、余波 −4、收束 −13。网格切点提前 25 ms；帧按中心时间归属镜头，量化误差 ≤17 ms。
- BGA 从其 0.160 s 处开始播放，音轨静音；BGA 比配乐短 0.83 s，末帧补齐。
- BGA 对位事件（配乐秒 → 镜头号 → 重音）：{', '.join(f"{t}→#{n}→{'rgb' if i%2==0 else 'block'} 4f" for i,(t,n) in enumerate(S['bga_events']))}；每处一次硬切 + ≤4 帧重音 + 边线/BGA 亮度闪一下；144.3 与 187.2 另各带 0.6 s ASCII 转场。

## 2. 段落与镜头
| 配乐秒 | 段落 | 镜头数 | 平均长(s) | 长度范围 | 规则 / 特效 |
|---|---|---|---|---|---|
{chr(10).join(seg_rows)}

全片 **{S['shots']}** 个镜头，最短 {S['min_shot_frames']} 帧，最长 {S['max_shot']} s；不规则区连续碎切（<0.3 s）最长 {S['max_irregular_frag_run_s']} s（要求 <15 s）；余波速度坡 {S['n_ramps']} 次。
取景窗：铺垫 75–100%（随 1 s 平滑能量）、规则区 ≥80%、峰值/不规则/碎片 100%、回落 92%、余波 96%、收束 100→75%；相邻镜头取景窗按镜头中点线性插值。

## 3. 采样起音点来源
- 100–188 s：重检（300–3400 Hz 起音 z 分 + 0.7×人声频段占比 z 分）取前 40（`work/vocal_onsets.json`），{S['corp_cuts']} 个进入切点；100–113 s 为 Corp-ra-corp 簇，125–154、166–187 为第二、三组。前 10 个：{', '.join(str(t) for t,_ in corp[:10])}。
- 202–219.5 s：原 7 个 + 补检合并 28 个（最小间隔 0.3 s），全部硬切（{S['frag_cuts']} 个独立切点）。
- 0–80 s：第 1 步候选仅作参考，不改切点。不规则区其余切点：最强 80 个起音。

## 4. 裁切与画幅
- 顶部裁 **6%**（检查过 7%：clip 15「绅士快车」、clip 33「裂开了吃刺球」等文字落在 6–7% 带内，故回退）；剩余的盾点条一截由取景窗顶部 2.2% 高的深色边条遮住（盾点条实际延伸到原画面约 7.5–8%）（B 版边条下沿是强调色细线）。对比图：`fxlab/docs/crop_compare_v2.jpg`、`crop_topstrips_v2.jpg`、`crop_band_4to11pct_v2.jpg`（红线 6%、青线 7%）。
- 裁后 3132×2030（1.543:1），不再裁切；取景窗按高度缩放居中。版本 A：窗外同素材模糊放大 1.15× 并压暗至 42%，BGA 只在 0–16 s 与 266 s 后渐显。版本 B：窗外全程 BGA。**实测 BGA 自身左右各 ~8% 列是纯黑**（全片两侧亮像素 0%），满屏取景窗下两侧 6.6% 的条里什么都没有；因此 B 版：BGA 先裁掉自身黑边（取 8–92% 列）再铺满画布，**B 版取景窗上限 88%**（两侧各 ≈12% 宽字符条），色阶拉伸（99.5 百分位→90% 白）+ 冷暗基色 #293b4f，窗外平均亮度 ≈ min(30%, 0.85×窗内亮度)，不比窗内亮（小样实测窗外 19%、窗内 23%）；BGA 亮度随起音脉冲 +40%（120 ms 衰减）；窗边 1 px 强调色细线。
- 9:16 版：同一 EDL，上下填充，另行渲染。

## 5. 配色
- BGA 主色 #293b4f、#47420b、#6e6f5c、#4c060c、#61151d；游戏画面色调映射：阴影→#293b4f、高光→#e1decd，强度 0.15（余波/收束 0.28）。
- 字符前景色（每帧最亮 3% 均值）中位 #473d38，很暗；特效强调色取 10 s 代表色按同色相提亮到 L≈0.75。
- ASCII 转场 {S['n_ascii']} 次，累计 {S['ascii_total']} s（≤10 s），480p 字符栅格 + 强调色。

## 6. 素材预算（已按 B 方案）
- 需求 {S['need_total']} s 源素材 / 供给 {S['supply_raw']} s（116 段各内缩 0.15 s）。
- 规则：窗口间隔 0.1 s；每段 ≤5 窗（第 3+ 窗只在碎切段）；同段两次出现间隔 ≥3 s；余波整段 0.5×；不规则区簇外镜头 0.5–1.5 s。
- 结果：镜头 {S['shots']}；素材利用率 **{S['utilization']*100:.1f}%**（{S['src_used']} s ÷ 304.5 s）；复用分布（出现次数：段数）{S['reuse_dist']}；**取景变体填充 {len(S['variant_filled'])} 个**（#{', #'.join(map(str,S['variant_filled']))}，放大 120–180%、裁切位在运动中心附近）；**空镜头 {len(S['unassigned'])} 个**；未使用片段：无。
- 检查：最长相似连续 {S['max_similar_run']}；短镜头亮度差违规 {len(S['short_lum_viol'])}；闪白 {S['n_flash']} 次、任意 1 s 内最多 {S['max_flash_per_s']} 次；97/109 各 1 次 ≤0.4 s，前后 0.5 s 无闪白。

## 7. 交付
- 540p 整首预览 ≤90 MB（视频 ≤2.3 Mbps + 音频 160 kbps）。1080p：REST API 建 Release `deliver-v3` 上传；失败则 3 段 ≤90 MB 提交到 `deliver-media` 分支。
- 音频：只用这首歌；loudnorm I=−10.5 LUFS、TP ≤ −1 dBTP。无文字、无片尾字幕（无 names.txt）。
"""
open('../docs/PLAN_MONTAGE_V3.md','w',encoding='utf-8').write(md); print('doc ok',len(md))
