"""样 E 自检：背景、边缘剖面、对位（glyph_events.csv）、封装 start_time。"""
import os,sys,json,csv,subprocess,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
W,H=960,540; lum=lambda f:f@np.array([0.299,0.587,0.114],np.float32)
def read_frames(fn,fps=None,scale=None):
    vf=[]; 
    if fps: vf.append(f'fps={fps}')
    if scale: vf.append(f'scale={scale[0]}:{scale[1]}')
    p=subprocess.run(['ffmpeg','-v','error','-i',fn,*(['-vf',','.join(vf)] if vf else []),'-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
    w,h=(scale if scale else (W,H)); return np.frombuffer(p,np.uint8).reshape(-1,h,w,3).astype(np.float32)
def background(fn,meta):
    F=read_frames(fn,fps=1); res_out=[];res_far=[]; M=meta['frames']
    for i,f in enumerate(F):
        t=meta['t0']+i+0.5; m=min(M,key=lambda e:abs(e['t']-t)); x,y,ww,wh=m['x'],m['y'],m['ww'],m['wh']
        for lst,mg in ((res_out,0),(res_far,120)):
            mask=np.ones((H,W),bool); mask[max(0,y-mg):y+wh+mg,max(0,x-mg):x+ww+mg]=False
            if mask.sum()<3000: continue
            L=lum(f); thr=np.percentile(L[mask],60); sel=mask&(L<=thr); lst.append(np.median(f[sel],0))
    return np.median(res_out,0),np.median(res_far,0),len(res_out),len(res_far)
def align(fn,meta,sched,edl_shots):
    """帧显示起点 = kf/30；与锚点比较（正=画面晚于锚点）"""
    fr=meta['frames']; rows=[]
    def first(pred):
        for e in fr:
            if pred(e): return e['k']/30.0
        return None
    def dev(vis,anchor): return None if vis is None else round((vis-anchor)*1000,1)
    shot_at={e['k']:e['shot'] for e in fr}
    def cut_after(anchor):   # 第一帧：镜头号 != 锚点前镜头
        pre=[e['shot'] for e in fr if e['t']<anchor-0.05][-1]
        return first(lambda e:e['t']>anchor-0.05 and e['shot']!=pre)
    for e in sched:
        k=e['kind']; ent=lambda ev,fr_=fr:[d for d in ev['dis'] if d['id']==e['id']]
        P=lambda ev:(ent(ev)[0]['p'] if ent(ev) else None)
        if k=='out':
            vs=first(lambda ev:ent(ev) and P(ev)>0 and ev['t']>=e['A0']-0.1); vf=first(lambda ev:ent(ev) and P(ev)>=1.0 and ev['t']>=e['A0']-0.1); vc=cut_after(e['A2'])
            rows.append([e['id'],'溶出',round(e['A0']*1000,1),dev(vs,e['A0']),round(e['A1']*1000,1),dev(vf,e['A1']),round(e['A2']*1000,1),dev(vc,e['A2']),e['anchor_ids']])
        elif k=='in':
            vr=first(lambda ev:ent(ev) and P(ev)<1.0 and ev['t']>=e['A0']-0.1); vz=first(lambda ev:ev['t']>=e['A1']-0.1 and not ent(ev))
            rows.append([e['id'],'凝出',round(e['A0']*1000,1),dev(vr,e['A0']),round(e['A1']*1000,1),dev(vz,e['A1']),round(e['B0']*1000,1),dev(first(lambda ev:ev['t']>=e['B0']-0.05 and ent(ev)),e['B0']),e['anchor_ids']])
        elif k=='scan':
            vs=first(lambda ev:ent(ev)); vc=cut_after(e['A2'])
            rows.append([e['id'],'扫描式',round(e['A0']*1000,1),dev(vs,e['A0']),None,None,round(e['A2']*1000,1),dev(vc,e['A2']),e['anchor_ids']])
        elif k=='local':
            vs=first(lambda ev:ent(ev)); ve=first(lambda ev:ev['t']>=e['t']+e['dur']-0.05 and not ent(ev))
            rows.append([e['id'],'局部溶解',round(e['t']*1000,1),dev(vs,e['t']),None,None,round((e['t']+e['dur'])*1000,1),dev(ve,e['t']+e['dur']),None])
    return rows
def profile_metrics(f,m,win_mean):
    """4 个方向的亮度剖面：沿边缘取窗口中段 60% 的平均（抑制画面内容/按钮的纹理），范围 边缘内 80px → 边缘外 120px，方向由内向外；
    返回 [(方向, 最大反弹量/窗均亮%, 每10px 最大变化/窗均亮%)]"""
    L=lum(f); x,y,ww,wh=m['x'],m['y'],m['ww'],m['wh']; res=[]
    r0,r1=int(y+0.2*wh),int(y+0.8*wh); c0,c1=int(x+0.2*ww),int(x+0.8*ww)
    lines={'左':L[r0:r1,max(0,x-120):x+80].mean(0)[::-1],'右':L[r0:r1,x+ww-80:min(W,x+ww+120)].mean(0),
           '上':L[max(0,y-120):y+80,c0:c1].mean(1)[::-1],'下':L[y+wh-80:min(H,y+wh+120),c0:c1].mean(1)}
    for nm,p in lines.items():
        if len(p)<30 or (nm=='上' and y<4) or (nm=='下' and y+wh>H-4): continue
        sm=np.convolve(p,np.ones(5)/5,'same')[2:-2]; d=np.diff(sm); rebound=float(max(0,d.max())) if len(d) else 0
        ch10=float(np.abs(sm[10:]-sm[:-10]).max()) if len(sm)>10 else 0
        res.append((nm,rebound/win_mean*100,ch10/win_mean*100))
    return res
if __name__=='__main__':
    fn='work/preview/sample_E_540.mp4'; meta=json.load(open(fn+'.meta.json')); sched=meta['sched']
    bo,bf,n1,n2=background(fn,meta); print('成片背景（非字符像素，窗口外）中位 RGB',bo.round(1),'；窗口外 120px 以外',bf.round(1),'（帧数 %d/%d）'%(n1,n2))
    rows=align(fn,meta,sched,None)
    with open('work/glyph_events.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['事件id','变体','起点锚点ms','起点画面偏差ms','完全覆盖/还原锚点ms','完全覆盖/还原偏差ms','切换/结束锚点ms','切换/结束偏差ms','锚点id']); w.writerows(rows)
    devs=[abs(x) for r in rows for x in (r[3],r[5],r[7]) if x is not None]
    print('对位偏差 |ms|：最大 %.1f 中位 %.1f（n=%d）'%(max(devs),float(np.median(devs)),len(devs)))
    for r in rows: print(r)
    st={}
    for s in subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type,start_time','-of','csv=p=0',fn]).decode().split():
        a,b=s.split(','); st[a]=float(b)
    print('mp4 start_time 视频/音频',st,'差 %.1f ms'%(abs(st['video']-st['audio'])*1000))
