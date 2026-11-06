#!/usr/bin/env python3
"""只重新封装音频、复用视频流：改变「画面相对音乐的提前量」。
用法：python3 remux_lead.py <提前量毫秒> [输入.mp4] [输出.mp4] [mix.wav]
  提前量 = 音频相对画面延迟的毫秒数（0 = 画面与音乐同步；25 = 本片默认；45 = 更强的提前感）。
  - 视频流直接复制（不重新编码）；音频重新编码为 AAC 256k（关闭噪声替代与强度立体声）。
  - 若给出 mix.wav（整首、静态增益+限幅的连续混音），音频由它生成（最干净）；否则从输入文件里解码音频，
    先去掉原来的 25ms 延迟再按新提前量延迟（会多一次 AAC 编码，音质略降）。
例：python3 remux_lead.py 45 final_1080p_v3_full.mp4 final_1080p_lead45.mp4"""
import sys,subprocess,os,tempfile
def run(a): subprocess.run(['ffmpeg','-v','error','-y']+a,check=True)
ms=int(sys.argv[1]); src=sys.argv[2] if len(sys.argv)>2 else 'final_1080p_v3_full.mp4'; dst=sys.argv[3] if len(sys.argv)>3 else f'remux_lead{ms}_'+os.path.basename(src); mix=sys.argv[4] if len(sys.argv)>4 else None
BASE=25   # 成片音频已含的延迟
dur=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',src]).decode().strip())
tmp=tempfile.mkdtemp(); wav=os.path.join(tmp,'a.wav')
if mix and os.path.exists(mix): run(['-i',mix,'-af',f'adelay={ms}:all=1','-t',f'{dur:.3f}','-ar','48000','-ac','2',wav])
else: run(['-i',src,'-vn','-af',f'atrim=start={BASE/1000:.3f},asetpts=PTS-STARTPTS,adelay={ms}:all=1','-t',f'{dur:.3f}','-ar','48000','-ac','2',wav])
aac=os.path.join(tmp,'a.m4a'); run(['-i',wav,'-c:a','aac','-b:a','256k','-aac_pns','0','-aac_is','0',aac])
run(['-i',src,'-i',aac,'-map','0:v','-map','1:a','-c','copy','-t',f'{dur:.3f}','-movflags','+faststart',dst]); print('完成',dst,'（提前量',ms,'ms）')
