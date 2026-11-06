# 技术栈说明（实测统计，2026-11）

## 1 代码规模（fxlab/montage/，不含 work/ 中间文件）
| 语言 | 文件数 | 代码行数 |
|--|--|--|
| Python 3.13 | 77（含 effects/glyph_dissolve.py） | 7964 行 |
| Shell | 0（仅 fxlab/deliver/ 下 3 个 join_1080p*.sh，各 5 行） | — |
| JavaScript | 0 | — |
| 数据 | CSV 1（bw_keyframes.csv）、JSON 2 | — |
核心：render_c.py 456 行（合成器）、render_v4.py 278、effects/glyph_dissolve.py 178、beat.py 137、onsets_fine.py 54。

## 2 库与外部程序
- 实际用到：Python 3.13.18、**numpy 2.5.3**、**Pillow 12.3.0**（滤波/缩放/高斯模糊）、标准库（subprocess、multiprocessing、csv、json）。
- **未使用**：librosa、OpenCV、scipy、numba、soundfile。节拍/起音分析是自写的 numpy 实现（rfft 短时傅里叶 + 频谱通量，onsets_fine.py / beat.py）。
- 外部程序：**ffmpeg / ffprobe 6.1.1**（libx264、AAC）。

## 3 渲染流程分工
- **ffmpeg**：解码素材片段（rawvideo 管道）与 BGA（`fps=30,scale`）；x264 编码（540p 单遍 CRF+码率上限，1080p 两遍 7.2 Mbps）；`concat -c copy` 拼接分块；音频 `adelay=25`（画面领先 25 ms）、AAC 256k（`-aac_pns 0 -aac_is 0`）、静态增益 + alimiter 混音（mix.wav）；封装与 remux；质检用 ebur128/blackdetect。
- **Python（numpy 逐帧）**：节拍/起音/锚点分析、EDL 规划（匈牙利指派）、每一帧的合成（窗口、特效、BGA 文字层、光晕、黑白）、帧送入 ffmpeg 编码、质检统计。
- 每帧在 float32 数组里合成，整片 8364 帧；分块（540p 10 s / 1080p 2 s）可断点续跑，多进程并行。

## 4 各特效实现（均为 numpy/Pillow 逐帧，没有使用 ffmpeg 滤镜做特效）
| 特效 | 实现 |
|--|--|
| 卡点切换 | Python 规划：切点吸附八分拍网格/强起音锚点，EDL 写入 CSV；ffmpeg 只负责解码（帧按帧中心时刻取） |
| 速度坡 | Python：时间映射表 `np.interp`（均匀/ramp/hit 局部速度包络），再按映射取源帧，不用 ffmpeg setpts |
| 字符溶解（溶出/凝出/扫描式/交叉溶解/局部块/收束） | numpy：8×13 像素格子按亮度映射字符集 " .:-=+*#"，字形图集预渲染（Pillow），覆盖率按事件进度逐帧计算，12 Hz 以内阶梯刷新 |
| 黑白渐变 | numpy：`out = luma + S·(color − luma)`，Rec.709 亮度，S 由 bw_keyframes.csv 经 smoothstep 逐帧算出，只作用于游戏层 |
| 边缘光晕 / 渐变 | numpy+Pillow：窗口向黑渐变蒙版，边缘 ~40px 平均色经高斯模糊生成外溢光晕，screen 叠加 |
| BGA 文字层 | numpy：BGA 帧亮度键 → 文字行分段 → 逐段强度、1.5 px 描边、字形暗晕；PBS 区按局部背景亮度在 HSL 空间调字色 |
| 其他（扫描线、噪点、色差、暗角、泛光、闪白） | numpy 逐帧 |

## 5 视频简介用语（准确版）
“本片由自写 Python（numpy + Pillow）逐帧合成，画面解码与编码、音频混音与封装使用 ffmpeg；节拍与起音分析为自实现的频谱通量算法，字符溶解、黑白渐变、边缘光晕和文字层均为程序生成，未使用 AI 生成画面。”
