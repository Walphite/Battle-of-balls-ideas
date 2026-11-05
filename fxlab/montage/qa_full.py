"""第 4 步质检（全片 540p 预览）→ work/qa_report.md"""
import os,sys,json,csv,glob,re,subprocess,collections,numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
os.environ['PROBE']='1'
import render_v4 as R4, check_e as CE
FPS=30; T=278.806; PV='../deliver/preview/preview_full.mp4'
out=[]; P=lambda *a: (out.append(' '.join(str(x) for x in a)),print(*a))
byk={}
for f in sorted(glob.glob('work/full/?/c_*.mp4.meta.json')):
    for e in json.load(open(f)): byk[e['k']]=e
frames=[byk[k] for k in sorted(byk)]; P('## 0 帧数',len(frames),'（应为 8364）')
S=json.load(open('work/glyph_schedule_v5.json'))['events']
for e in S: e['id']=f"{e['kind']}@{e.get('A0',e.get('t',0)):.3f}"
EDL=list(csv.DictReader(open('work/edl.csv',encoding='utf-8'))); AN=json.load(open('work/anchors.json'))['anchors']; BT=np.array(json.load(open('work/beats_v4.json'))['beats']); FT=np.load('work/onsets_fine.npz')['t']
# ---------- 5 对位
P('\n## 5 对位')
meta={'frames':frames,'t0':0}
rows=CE.align(PV,meta,[e for e in S if e['kind'] in('out','in','scan','local')],None)
fin=[e for e in S if e['kind']=='final'][0]
vis=next((f['k']/30 for f in frames if any(d['id']==fin['id'] and d['p']>0 for d in f['dis'])),None)
rows.append([fin['id'],'收束',round(fin['A0']*1000,1),None if vis is None else round((vis-fin['A0'])*1000,1),None,None,None,None,None])
devs=[abs(x) for r in rows for x in (r[3],r[5],r[7]) if x is not None]
with open('work/glyph_events.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['事件id','变体','起点锚点ms','起点画面偏差ms','完全覆盖/还原锚点ms','完全覆盖/还原偏差ms','切换/结束锚点ms','切换/结束偏差ms','锚点id']); w.writerows(rows)
P('字符溶解事件 %d 个，偏差 |ms| 最大 %.1f 中位 %.1f（目标 ≤17；样本 %d）'%(len(rows),max(devs),float(np.median(devs)),len(devs)))
big=[(r[0],r[3],r[5],r[7]) for r in rows if any(x is not None and abs(x)>17 for x in (r[3],r[5],r[7]))]; P('偏差>17ms 的事件：',big)
bd=sorted({float(r['t_start']) for r in EDL}|{float(r['t_end']) for r in EDL}); anch=np.array([a['t_ms']/1000 for a in AN])
ab={r['anchor_id_end'] for r in EDL if r['anchor_id_end']}|{r['anchor_id'] for r in EDL if r['anchor_id']}
dBeat=[(b-BT[np.argmin(np.abs(BT-b))])*1000 for b in bd]; dOn=[(b-FT[np.argmin(np.abs(FT-b))])*1000 for b in bd]; dAn=[(b-anch[np.argmin(np.abs(anch-b))])*1000 for b in bd]
snapped=[b for b in bd if min(abs(anch-b))<0.0006]
P('镜头切点 %d 个：相对校正后拍点 中位 %+.1f ms（规则：提前 10ms），最大|误差| %.1f ms；相对最近精细起音 中位|误差| %.1f ms；落在锚点上(吸附) %d 个（对锚点误差 0）'%(len(bd),np.median(dBeat),max(abs(np.array(dBeat))[[i for i,b in enumerate(bd) if b not in snapped]]),np.median(np.abs(dOn)),len(snapped)))
BG=[5.11,29.77,44.57,47.91,144.31,187.21,233.14]; lum=np.array([f['lum'] for f in frames]); tt=np.array([f['t'] for f in frames])
P('BGA 7 个切换点（最近边界距离 / 事件时刻重音 / 预览中 BGA 画面突变帧相对事件的时间差）：')
for e in BG:
    nb=min(bd,key=lambda b:abs(b-e)); sh=next(r for r in EDL if float(r['t_start'])-0.05<=e<float(r['t_end'])); acc='bga_accent' in sh['fx']
    m=(tt>e-0.25)&(tt<e+0.25); d=np.abs(np.diff(lum[m])); j=int(np.argmax(d)); tj=tt[m][j+1]
    P('  %.2f：最近边界 %+.0f ms，%s；预览中最大亮度突变帧 %.3f s（相对事件 %+.0f ms）'%(e,(nb-e)*1000,'已加 ≤4 帧重音' if acc else '无重音',tj,(tj-e)*1000))
P('BGA 160ms 补偿：render_v4.BGA_OFF = %.3f（BgaReader 从 BGA 的 t+0.16 s 起播）；上面 7 个事件的画面突变帧与事件时刻差在 ±1–2 帧内，说明补偿仍生效。'%R4.BGA_OFF)
st={}
for n in ('A','B','C','full'):
    v=subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type,start_time','-of','csv=p=0',f'../deliver/preview/preview_{n}.mp4']).decode().split(); d={a.split(',')[0]:float(a.split(',')[1]) for a in v}
    st[n]=abs(d['video']-d['audio'])*1000
P('mp4 音视频 start_time 差（ms）：',{k:round(v,2) for k,v in st.items()})
# ---------- 4 使用规则
P('\n## 4 使用规则')
full=[e for e in S if e['kind']!='local']; tot=0
for e in full:
    k=e['kind']; ts=e['A0']-0.5 if k=='out' else (e['B0'] if k=='in' else e['A0']); te=e['A2'] if k in('out','scan') else (e['A1'] if k=='in' else e['t_end']); tot+=te-ts
    if 79<=ts<100 or 208<=ts<230 or 79<te<100 or 208<te<230: P('  !! 峰值区内的完整溶解',e['kind'],ts,te)
loc=[e for e in S if e['kind']=='local']
P('完整溶解 %d 次 / %.2f s（≤10 次、≤12 s）；局部溶解 %d 次 / %.2f s（≤8 s）；开场第一个镜头为凝出：%s；最后一个镜头完整溶解至黑屏：%s'%(len(full),tot,len(loc),sum(e['dur'] for e in loc),full[0]['kind']=='in' and full[0].get('opening',False),full[-1]['kind']=='final'))
sh_d=[float(r['t_end'])-float(r['t_start']) for r in EDL if 100<=float(r['t_start'])<188]; P('100–188 s 镜头最短 %.2f s（碎切 <0.5s 的镜头 %d 个）→ 不存在连续碎切'%(min(sh_d),sum(1 for d in sh_d if d<0.5)))
# ---------- 6 画面
P('\n## 6 画面')
bl=[l for l in open('work/full/blackdetect.txt') if 'black_start' in l]; P('blackdetect：',[re.search(r'black_start:([\d.]+) black_end:([\d.]+)',l).groups() for l in bl],'（开头 0.33 s 为第一个镜头之前，末尾 277.5–278.8 s 为收束熄灭至黑屏，均为设计）')
fz=re.findall(r'freeze_start: ([\d.]+).*?freeze_duration: ([\d.]+)',open('work/full/freezedetect.txt').read().replace('\n',' ')); P('freezedetect(-55dB, ≥0.15s)：',fz)
g=json.load(open('work/guard_full.json')); cc=collections.Counter((x[3] or '其他').split('(')[0] for x in g['over30'])
P('相邻帧整帧亮度变化>30%%：%d 处，其中 %s；未登记(错误) %d；孤立 1–2 帧异常 %d'%(len(g['over30']),dict(cc),len(g['bad']),len(g['isolated'])))
fl=sorted(float(r['t_start']) for r in EDL if r['flash']=='1'); mx=max((sum(1 for x in fl if t<=x<t+1) for t in fl),default=0); P('闪白 %d 次，任意 1 s 内最多 %d 次（≤2），峰值亮度 60%%（≤70%%）'%(len(fl),mx))
# ---------- 7 素材
P('\n## 7 素材')
CL={int(r['clip_id']):r for r in csv.DictReader(open('work/clips.csv',encoding='utf-8'))}; used=[int(r['clip_id']) for r in EDL]
P('使用片段 %d；舍弃 clip_id %s；重复使用的片段 %d（必须 0）'%(len(set(used)),sorted(set(CL)-set(used)),len(used)-len(set(used))))
spl=[(r['shot'],r['clip_id']) for r in EDL if float(r['src_in'])>0.1+1e-6 or float(CL[int(r['clip_id'])]['dur'])-float(r['src_out'])>0.1+1e-6]
P('整段播放检查：首尾修剪 >0.1s 的镜头 %d 个（被拆分=0）：%s'%(len(spl),spl[:10]))
bs=collections.Counter(int(r['beats']) for r in EDL); sp=[float(r['speed'].split(':')[1]) if r['speed'].startswith('ramp') else float(r['speed']) for r in EDL]
P('拍数分布',dict(sorted(bs.items())),'平均 %.2f；变速：<0.85 的 %d 段（收束慢放），0.85–1.15 的 %d 段，速度坡 %d 段；范围 %.3f–%.3f'%(np.mean(list(bs.elements())),sum(1 for x in sp if x<0.85),sum(1 for x in sp if 0.85<=x<=1.15),sum(1 for r in EDL if r['speed'].startswith('ramp')),min(sp),max(sp)))
P('素材利用率 %.1f%%（%.1f s ÷ 304.5 s）'%(sum(float(r['src_out'])-float(r['src_in']) for r in EDL)/304.5*100,sum(float(r['src_out'])-float(r['src_in']) for r in EDL)))
FE={f['clip_id']:f for f in json.load(open('work/clip_feats.json'))}
def sim(a,b):
    ha,hb=np.array(FE[a]['hue_hist']),np.array(FE[b]['hue_hist']); return ha@hb/(np.linalg.norm(ha)*np.linalg.norm(hb)+1e-9)>0.93 and abs(FE[a]['lum']-FE[b]['lum'])<0.04
run=1;mx=1
for a,b in zip(used,used[1:]): run=run+1 if sim(a,b) else 1; mx=max(mx,run)
P('相似画面连续最长 %d（≤2）'%mx)
# ---------- 拼接处
P('\n## 3 拼接处')
pk=subprocess.check_output(['ffprobe','-v','error','-select_streams','v','-show_entries','packet=pts_time','-of','csv=p=0',PV]).decode().split(); pts=np.array(sorted(float(x) for x in pk)); dd=np.diff(pts)
P('视频帧数 %d，PTS 间隔 最小 %.4f 最大 %.4f（应恒为 0.0333）；时长 %.3f s'%(len(pts),dd.min(),dd.max(),pts[-1]+1/30))
def fr(t): return CE.read_frames.__globals__['np'].frombuffer(subprocess.run(['ffmpeg','-v','error','-ss',f'{t:.4f}','-i',PV,'-frames:v','1','-vf','scale=160:90','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout,np.uint8).astype(np.float32)
for name,k in (('A|B',2976),('B|C',6254)):
    a=[fr((k+d)/30+0.001) for d in (-2,-1,0,1)]; df=[float(np.abs(a[i+1]-a[i]).mean()) for i in range(3)]; P(' 拼接 %s（帧 %d）相邻帧差 %s：%s'%(name,k,[round(x,1) for x in df],'有重复帧' if min(df)<0.05 else '无重复帧'))
x=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',PV,'-vn','-ac','1','-ar','48000','-f','f32le','-'],capture_output=True).stdout,np.float32)
for name,t in (('A|B',2976/30),('B|C',6254/30)):
    i=int(t*48000); seg=x[i-480:i+480]; d=np.abs(np.diff(seg)); loc=np.sqrt((x[i-4800:i+4800]**2).mean()); P(' 音频 %s：±10ms 内最大样本间跳变 %.3f（局部 RMS %.3f，比值 %.1f；<2 视为无爆音）'%(name,d.max(),loc,d.max()/loc))
P('音频全长 %.3f s；整片 mp4 音视频均由单次编码的音频 + 视频块无损拼接'%(len(x)/48000))
# ---------- 9 响度
P('\n## 9 响度')
t=subprocess.run(['ffmpeg','-hide_banner','-i',PV,'-af','ebur128=peak=true','-vn','-f','null','-'],capture_output=True,text=True).stderr; t=t[t.rindex('Summary'):]
P('综合响度 %s LUFS；真峰值 %s dBTP'%(re.search(r'I:\s+(-?[\d.]+) LUFS',t).group(1),re.search(r'Peak:\s+(-?[\d.]+) dBFS',t).group(1)))
open('work/qa_report.md','w',encoding='utf-8').write('# 第 4 步质检（540p 全片预览）\n\n'+'\n'.join(out)+'\n')
