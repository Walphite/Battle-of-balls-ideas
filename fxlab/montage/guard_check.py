"""通用护栏：整片（270p 探测渲染 + 全片字符溶解调度）逐帧整帧平均亮度。
 规则：相邻帧变化 >30% 只允许出现在 已登记闪白切 / 完整溶解起止 / 镜头硬切；其余视为错误。另查孤立 1–2 帧异常。"""
import json,csv,glob,sys,os,re,subprocess,numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
os.environ['PROBE']='1'
import render_c as RC
import render_v4 as R4
SH=R4.load_edl()
for _s in SH: _s['_map']=R4.src_map(_s)
_cl={}
def clip_lum(cid):
    if cid not in _cl:
        r=R4.CLIPS[cid]; p=subprocess.run(['ffmpeg','-v','error','-i','input2/clips/'+r['file'],'-an','-vf','crop=iw:ih*0.94:0:ih*0.06,scale=96:-2','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True).stdout
        h=int(round(96*int(r['h'])*0.94/int(r['w'])/2))*2; _cl[cid]=np.frombuffer(p,np.uint8).reshape(-1,h,96).astype(np.float32).mean((1,2))/255
    return _cl[cid]
def bga_change(t):
    p=subprocess.run(['ffmpeg','-v','error','-ss',f'{max(0,t+0.16-3/30):.3f}','-t','0.25','-i','input2/bga/Credits.bga.mp4','-vf','fps=30,scale=96:54','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True).stdout
    F=np.frombuffer(p,np.uint8).reshape(-1,54,96).astype(np.float32).mean((1,2))/255
    return float((np.abs(np.diff(F))/np.maximum(F[:-1],0.01)).max()) if len(F)>1 else 0.0       # ±3 帧内相邻帧最大相对变化
def src_change(t):
    s=next((x for x in SH if x['t']<=t<x['te']),None)
    if s is None: return None
    L=clip_lum(s['clip']); best=0.0
    for d in (-1,0,1):     # 输出帧 ±1 内的相邻帧对
        ta=t+d*FR
        if ta-FR<s['t'] or ta>=s['te']: continue
        off=int(round(s['src_in']*float(R4.CLIPS[s['clip']]['fps'])))      # 窗口相对片段起点的偏移（src_in）
        i1=min(RC.src_index_c(s,ta-s['t'])+off,len(L)-1); i0=min(RC.src_index_c(s,ta-FR-s['t'])+off,len(L)-1); best=max(best,float(abs(L[i1]-L[i0])/max(L[i0],0.03)))
    return best
FR=1/30
fr=[]
byk={}
for f in sorted(glob.glob('work/preview/probe/p*.mp4.meta.json'))+sorted(glob.glob('work/preview/probe/q*.mp4.meta.json')):   # q* 覆盖 p*（局部重渲）
    for e in json.load(open(f))['frames']: byk[e['k']]=e
fr=[byk[k] for k in sorted(byk)]; k0=fr[0]['k']; K=np.array([e['k'] for e in fr]); m=np.array([e['lum'] for e in fr]); sh=np.array([e['shot'] for e in fr])
sched=json.load(open('work/glyph_schedule_v5.json'))['events']; E=list(csv.DictReader(open('work/edl.csv',encoding='utf-8')))
shots=[(int(r['shot']),float(r['t_start']),float(r['t_end']),r['fx'],int(r['flash'])) for r in E]
def reg_full():
    out=[]
    for e in sched:
        if e['kind'] in('out','in','scan','final'):
            ts=e.get('t'); te=e['t']+e['dur']; out.append((ts,te,e['kind'],e.get('A0'),e.get('A1'),e.get('A2')))
    return out
FULL=reg_full()
def near(t,pts,pad): return any(abs(t-p)<=pad for p in pts)
cut_t=[s[1] for s in shots]+[s[2] for s in shots]; flash_t=[s[1] for s in shots if s[4]]
full_edges=[x for e in FULL for x in (e[0],e[1])]
def active_fx(t):
    out=[]
    for n,a,b,fx,fl in shots:
        if a<=t<b:
            tau=t-a
            for tok in fx.split():
                mm=re.match(r'(rgb|block|tear|stutter|bga_accent)(?::(\d+)f)?@([\d.]+)',tok)
                if mm and float(mm.group(3))<=tau<float(mm.group(3))+int(mm.group(2) or 4)/30: out.append(mm.group(1))
            if fl and tau<0.1: out.append('flash')
    return out
rows=[]; bad=[]
for i in range(1,len(m)):
    a,b=m[i-1],m[i]; rel=abs(b-a)/max(a,0.08)          # 分母下限 0.08：近乎全黑时（淡入）相对变化无意义
    if rel<=0.30: continue
    t=fr[i]['t']; cat=None
    if sh[i]!=sh[i-1]: cat='镜头硬切'+('(含登记闪白)' if near(t,flash_t,0.1) else '')
    elif near(t,flash_t,0.12): cat='登记闪白'
    elif near(t,full_edges,0.12): cat='完整溶解起止'
    else:
        sc=src_change(t); bc=bga_change(t)
        if sc is not None and sc>=0.25: cat='素材自身亮度突变(源帧间 %.0f%%)'%(sc*100)
        elif bc>=0.25: cat='BGA 内容切换(BGA 帧间 %.0f%%)'%(bc*100)
    rows.append((t,a,b,rel,cat,fr[i]['dis'],active_fx(t)))
    if cat is None: bad.append(rows[-1])
print('相邻帧亮度变化>30%% 共 %d 处：其中已登记（镜头硬切/闪白/完整溶解起止）%d，其余 %d'%(len(rows),len(rows)-len(bad),len(bad)))
for r in bad: print('  错误候选 t=%.3f  亮度 %.3f→%.3f（%+.0f%%） 溶解=%s 特效=%s'%(r[0],r[1],r[2],(r[2]-r[1])/max(r[1],0.03)*100,[d['kind']+':'+str(d['p']) for d in r[5]],r[6]))
# 孤立 1–2 帧异常
iso=[]
for L in (1,2):
    for i in range(1,len(m)-L):
        a,b=m[i-1],m[i+L]; seg=m[i:i+L]
        if abs(a-b)/max(a,b,0.03)<0.10 and np.all(np.abs(seg-a)/max(a,0.03)>0.20) and sh[i-1]==sh[i+L]:
            iso.append((fr[i]['t'],L,a,float(seg.mean()),b,[d['kind'] for d in fr[i]['dis']],active_fx(fr[i]['t'])))
print('孤立 1–2 帧异常（与前后帧差异>20%% 而前后帧彼此相差<10%%）共 %d 处'%len(iso))
for r in iso: print('  t=%.3f %d帧 前%.3f 中%.3f 后%.3f 溶解=%s 特效=%s'%r)
json.dump(dict(over30=[(r[0],r[1],r[2],r[4],[d['kind'] for d in r[5]],r[6]) for r in rows],bad=[(r[0],r[1],r[2],[d['kind'] for d in r[5]],r[6]) for r in bad],isolated=iso,frames=len(m)),open('work/guard_report.json','w'),default=float)
print('帧数',len(m),'范围 %.2f–%.2f s'%(fr[0]['t'],fr[-1]['t']))
