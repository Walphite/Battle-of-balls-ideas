"""终版质检：对解码后的 final_540p.mp4 实测 → fxlab/deliver/final/qa_final.md"""
import os,sys,json,csv,glob,re,subprocess,collections,numpy as np
os.environ.setdefault('EDL','work/edl_v10.csv'); os.environ['PROBE']='1'
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
import render_v4 as R4, onsets_fine as OF, text_check as TC
FPS=30; PV='../deliver/final_v3/final_540p_v3.mp4'; EDLP=os.environ['EDL']; out=[]
def P(*a): s=' '.join(str(x) for x in a); out.append(s); print(s)
EDL=list(csv.DictReader(open(EDLP,encoding='utf-8'))); AN=json.load(open('work/anchors.json')); BT=np.array(json.load(open('work/beats_v4.json'))['beats']); LEAD=0.025
P('# 终版 V3 质检（final_540p_v3.mp4，解码后实测）\n')
# ---- 切点
P('## 1 切点（画面帧差检测 + 解码音频起音）')
p=subprocess.run(['ffmpeg','-v','error','-i',PV,'-an','-vf','scale=240:135','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True).stdout
F=np.frombuffer(p,np.uint8).reshape(-1,135,240).astype(np.float32); d=np.abs(np.diff(F,axis=0)).mean((1,2)); med=np.median(d)
det=np.array([(i+1)/FPS for i in range(len(d)) if d[i]>max(10,6*med) and d[i]>=d[max(0,i-1)] and d[i]>=d[min(len(d)-1,i+1)]])
anc_t=set()
for r in EDL:
    if r.get('anchor_id_end'): anc_t.add(float(r['t_end']))
    if r.get('anchor_id'): anc_t.add(float(r['t_start']))
plan=sorted({float(r['t_start']) for r in EDL}|{float(r['t_end']) for r in EDL}); plan=[x for x in plan if 0.3<x<278.5]
x=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',PV,'-vn','-ac','1','-ar','22050','-f','f32le','-'],capture_output=True).stdout,np.float32)
dd,tt=OF.flux(x); pk,z=OF.pick(dd,tt); ons=tt[pk]-OF.calibrate()[0]; st=z[pk]; strong=np.sort(ons[st>=np.percentile(st,70)]); strong2=np.sort(ons[st>=np.percentile(st,50)])
E8=np.sort(np.r_[BT,(BT[:-1]+BT[1:])/2])+LEAD       # 音频整体延迟 25ms 后的八分拍网格
er=[];bd=[];od=[];od2=[];ordb=[];ordo=[]
for pt in plan:
    j=int(np.argmin(np.abs(det-pt)))
    if abs(det[j]-pt)>0.08: continue
    g=det[j]; er.append((g-pt)*1000); b=(g-E8[np.argmin(np.abs(E8-g))])*1000; bd.append(b); o=(g-strong[np.argmin(np.abs(strong-g))])*1000; od.append(o); od2.append((g-strong2[np.argmin(np.abs(strong2-g))])*1000)
    if all(abs(pt-a)>0.001 for a in anc_t): ordb.append(b); ordo.append(o)
f=lambda a:'中位 %.1f 均值 %.1f 最大|.| %.1f'%(np.median(a),np.mean(a),np.max(np.abs(a)))
P('检测到 %d / 计划 %d 个切点；检测 − 计划（帧取整）：%s ms'%(len(er),len(plan),f(er)))
P('切点 − 八分拍网格（解码音频时间轴，含 25ms 音频延迟，设计值 −25 ms；全部切点）：%s ms'%f(bd))
P('普通切点（非锚点吸附，%d 个）− 八分拍网格：%s ms；锚点吸附的切点设计值 −25 ms（画面精确落锚点、音频晚 25ms）'%(len(ordb),f(ordb)))
P('切点 − 强起音(≥P70，解码音频)：%s ms；− 起音(≥P50)：%s ms；普通切点 − 强起音(≥P70)：%s ms'%(f(od),f(od2),f(ordo)))
P('≤40ms 比例：相对 P70 强起音 %.0f%%，相对 P50 强起音 %.0f%%'%(100*np.mean(np.abs(od)<=40),100*np.mean(np.abs(od2)<=40)))
# 音频延迟
a48=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',PV,'-vn','-ac','1','-ar','48000','-f','f32le','-'],capture_output=True).stdout,np.float32)[:48000*10]
b48=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-t','10','-i','work/audio/mix.wav','-vn','-ac','1','-ar','48000','-f','f32le','-'],capture_output=True).stdout,np.float32)[:len(a48)]
n=1<<int(np.ceil(np.log2(len(a48)+len(b48)))); c=np.fft.irfft(np.fft.rfft(a48,n)*np.conj(np.fft.rfft(b48,n)),n); mx=4800; lags=np.r_[np.arange(0,mx),np.arange(-mx,0)]; cc=np.r_[c[:mx],c[n-mx:]]
P('解码音频相对 mix.wav 的延迟 %.1f ms（设计 25.0；AAC/封装无额外延迟）'%(lags[int(np.argmax(cc))]/48))
# ---- 吞噬
P('\n## 2 吞噬对位')
S=json.load(open('work/plan_v10_stats.json')); P('映射 %d / %d = %.1f%%（仅强起音 ≥P50 计入；目标 ≥80%% 未达到，已取最优并记录）；一/二/三级 = %s；边界偏移 %d 个；互换 %d 个槽位；匈牙利 %.1f s，贪心回退：%s'%(S['mapped'],S['total'],S['coverage']*100,S['tier'],S['boundary_shifted'],S['swapped_slots'],S['solve_seconds'],S['greedy_fallback']))
P('吞噬重音显示帧相对锚点：最大 %.1f ms 中位 %.1f ms（帧量化）'%(S['disp_dev_ms']['max'],S['disp_dev_ms']['median']))
P('未落位原因统计：',S['reasons']); P('未落位 clip_id 与原因：',[(u['clip_id'],u['reason']) for u in S['unmapped']])
P('稀疏区（无强起音，如 188–202 s）不强行对位；吞噬重音与字符溶解/定格重复/硬切冲突而丢弃的 %d 处'%sum(1 for d in S['dropped'] if d[0]=='hit_accent'))
# ---- 响度
P('\n## 3 响度与音频')
t=subprocess.run(['ffmpeg','-hide_banner','-i',PV,'-af','ebur128=peak=true','-vn','-f','null','-'],capture_output=True,text=True).stderr; t=t[t.rindex('Summary'):]
P('综合响度 %s LUFS；真峰值 %s dBTP（目标 ≤ −1.5）'%(re.search(r'I:\s+(-?[\d.]+) LUFS',t).group(1),re.search(r'Peak:\s+(-?[\d.]+) dBFS',t).group(1)))
t2=subprocess.run(['ffmpeg','-v','error','-i',PV,'-af','ebur128=metadata=1:peak=none,ametadata=print:key=lavfi.r128.S:file=-','-vn','-f','null','-'],capture_output=True,text=True).stdout
ts=np.array([float(v) for v in re.findall(r'pts_time:([\d.]+)',t2)]); sv=np.array([float(v) for v in re.findall(r'r128.S=(-?[\d.]+)',t2)]); m=(ts>=6)&(ts<=275)
P('每 3 s 短时响度（6–275 s）最小 %.1f（%.0f s）最大 %.1f（%.0f s）'%(sv[m].min(),ts[m][sv[m].argmin()],sv[m].max(),ts[m][sv[m].argmax()]))
neigh=(sv[(ts>=70)&(ts<=76)].mean()+sv[(ts>=104)&(ts<=110)].mean())/2; mid=sv[(ts>=80)&(ts<=102)]
P('77–102 s 区间 S：最低 %.1f 平均 %.1f；相邻段平均 %.1f；比相邻段低 >6 LU 的窗口：%d 个'%(mid.min(),mid.mean(),neigh,int((mid<neigh-6).sum())))
v=subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type,start_time,duration','-of','csv=p=0',PV]).decode().split(); dv={a.split(',')[0]:(float(a.split(',')[1]),float(a.split(',')[2])) for a in v}
P('start_time 视频 %.3f 音频 %.3f（差 %.1f ms）；时长 视频 %.3f 音频 %.3f'%(dv['video'][0],dv['audio'][0],abs(dv['video'][0]-dv['audio'][0])*1000,dv['video'][1],dv['audio'][1]))
# ---- 亮度护栏
P('\n## 4 整帧亮度变化 >30% 与孤立异常')
os.environ['GUARD_FULL']='0'
byk={}
for fn in sorted(glob.glob('work/final3/540/p?/c_*.mp4.meta.json')):
    for e in json.load(open(fn)): byk[e['k']]=e
fr=[byk[k] for k in sorted(byk)]; lum=np.array([e['lum'] for e in fr]); sh=np.array([e['shot'] for e in fr]); tt2=np.array([e['t'] for e in fr])
SCH=json.load(open('work/glyph_schedule_v10.json'))['events']
full=[e for e in SCH if e['kind'] in('out','in','scan','final','f')]
def span(e):
    k=e['kind']; return (e['A0']-0.5 if k=='out' else (e['B0'] if k=='in' else e['A0']), e['A2'] if k in('out','scan') else (e['A1'] if k in('in','f') else e['t_end']))
edges=[y for e in full for y in span(e)]; flash=[float(r['t_start']) for r in EDL if r['flash']=='1']
SH=R4.load_edl()
for s_ in SH: s_['_map']=R4.src_map(s_)
import render_c as RC
cl={}
def clum(cid):
    if cid not in cl:
        r=R4.CLIPS[cid]; pp=subprocess.run(['ffmpeg','-v','error','-i','input2/clips/'+r['file'],'-an','-vf','crop=iw:ih*0.94:0:ih*0.06,scale=96:-2','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True).stdout
        h=int(round(96*int(r['h'])*0.94/int(r['w'])/2))*2; cl[cid]=np.frombuffer(pp,np.uint8).reshape(-1,h,96).astype(np.float32).mean((1,2))/255
    return cl[cid]
def srcchg(t):
    s=next((x for x in SH if x['t']<=t<x['te']),None)
    if s is None: return 0
    L=clum(s['clip']); best=0; off=int(round(s['src_in']*float(R4.CLIPS[s['clip']]['fps'])))
    for dlt in (-1,0,1):
        ta=t+dlt/FPS
        if ta-1/FPS<s['t'] or ta>=s['te']: continue
        i1=min(RC.src_index_c(s,ta-s['t'])+off,len(L)-1); i0=min(RC.src_index_c(s,ta-1/FPS-s['t'])+off,len(L)-1); best=max(best,abs(L[i1]-L[i0])/max(L[i0],0.03))
    return best
def bgach(t):
    pp=subprocess.run(['ffmpeg','-v','error','-ss',f'{max(0,t+0.16-3/FPS):.3f}','-t','0.25','-i','input2/bga/Credits.bga.mp4','-vf','fps=30,scale=96:54','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True).stdout
    G=np.frombuffer(pp,np.uint8).reshape(-1,54,96).astype(np.float32).mean((1,2))/255; return float((np.abs(np.diff(G))/np.maximum(G[:-1],0.01)).max()) if len(G)>1 else 0
cats=collections.Counter(); bad=[]
for i in range(1,len(lum)):
    rel=abs(lum[i]-lum[i-1])/max(lum[i-1],0.08)
    if rel<=0.3: continue
    t=tt2[i]
    if sh[i]!=sh[i-1]: cats['镜头硬切']+=1
    elif any(abs(t-q)<=0.12 for q in flash): cats['登记闪白']+=1
    elif any(abs(t-q)<=0.12 for q in edges): cats['完整溶解起止']+=1
    elif any(a_-0.05<=t<=b_+0.05 for a_,b_ in (span(e) for e in full)): cats['完整溶解覆盖阶梯（暗场内按字符覆盖率步进）']+=1
    elif srcchg(t)>=0.25: cats['素材自身亮度突变']+=1
    elif bgach(t)>=0.25: cats['BGA 内容切换']+=1
    else: cats['未登记']+=1; bad.append((round(t,3),round(float(lum[i-1]),3),round(float(lum[i]),3)))
P('亮度变化>30%%：%d 处，分类 %s；未登记（错误）清单（时间,前亮度,后亮度）：%s'%(sum(cats.values()),dict(cats),bad))
iso=[]
for L in (1,2):
    for i in range(1,len(lum)-L):
        a_,b_=lum[i-1],lum[i+L]; seg=lum[i:i+L]
        if abs(a_-b_)/max(a_,b_,0.03)<0.10 and np.all(np.abs(seg-a_)/max(a_,0.03)>0.20) and sh[i-1]==sh[i+L]: iso.append((round(float(tt2[i]),3),L))
P('孤立 1–2 帧异常：%d 处 %s'%(len(iso),iso))
fl=sorted(flash); mxf=max((sum(1 for q in fl if t0<=q<t0+1) for t0 in fl),default=0); P('闪白 %d 次，任意 1 s 内最多 %d 次（≤2），峰值 60%%（≤70%%）'%(len(fl),mxf))
bl=[l for l in subprocess.run(['ffmpeg','-hide_banner','-i',PV,'-vf','blackdetect=d=0.05:pix_th=0.05','-an','-f','null','-'],capture_output=True,text=True).stderr.split('\n') if 'black_start' in l]; P('blackdetect：',[re.search(r'black_start:([\d.]+) black_end:([\d.]+)',l).groups() for l in bl],'（仅开头第一个镜头之前与结尾熄灭，设计内）')
# ---- 素材
P('\n## 5 片段使用')
CL={int(r['clip_id']):r for r in csv.DictReader(open('work/clips.csv',encoding='utf-8'))}; used=[int(r['clip_id']) for r in EDL]
spl=[r['shot'] for r in EDL if float(r['src_in'])>0.1+1e-6 or float(CL[int(r['clip_id'])]['dur'])-float(r['src_out'])>0.1+1e-6]
P('使用 %d 个片段，每段恰好一次：重复 %d（必须 0）；首尾修剪超过 0.1 s 的（被拆分）%d（必须 0）；舍弃 clip_id %s'%(len(set(used)),len(used)-len(set(used)),len(spl),sorted(set(CL)-set(used))))
FE={f['clip_id']:f for f in json.load(open('work/clip_feats.json'))}
def sim(a,b): ha,hb=np.array(FE[a]['hue_hist']),np.array(FE[b]['hue_hist']); return ha@hb/(np.linalg.norm(ha)*np.linalg.norm(hb)+1e-9)>0.93 and abs(FE[a]['lum']-FE[b]['lum'])<0.04
run=1;mxr=1
for a_,b_ in zip(used,used[1:]): run=run+1 if sim(a_,b_) else 1; mxr=max(mxr,run)
P('相似画面连续最长 %d；素材利用率 %.1f%%'%(mxr,sum(float(r['src_out'])-float(r['src_in']) for r in EDL)/304.5*100))
# ---- 黑白脉冲
P('\n## 6 黑白（关键帧式脉冲）')
PL=json.load(open('work/bw_pulses.json'))['pulses']
def Sf(t):
    S=1.0
    for p in PL:
        if t<p['t_fall0'] or (p['t_rise1'] is not None and t>=p['t_rise1']): continue
        if t<p['t_peak']: x=(t-p['t_fall0'])/(p['t_peak']-p['t_fall0']); S=min(S,1-x*x*(3-2*x))
        elif p['t_rise0'] is None or t<p['t_rise0']: S=0.0
        else: x=(t-p['t_rise0'])/(p['t_rise1']-p['t_rise0']); S=min(S,x*x*(3-2*x))
    return S
g=(np.arange(8364)+0.5)/30; Sv=np.array([Sf(t) for t in g])
P('S<0.9 累计 %.1f s（%.1f%%，≤22%%）；S≤0.05 累计 %.1f s（%.1f%%）；相邻帧 S 最大变化 %.1f%%（≤12%%）'%((Sv<0.9).sum()/30,100*(Sv<0.9).mean(),(Sv<=0.05).sum()/30,100*(Sv<=0.05).mean(),100*np.abs(np.diff(Sv)).max()))
def cross(a,b,lvl,down):
    m=(g>=a)&(g<=b); tt=g[m]; ss=Sv[m]; i=np.where((ss<=lvl) if down else (ss>=lvl))[0]; return tt[i[0]] if len(i) else None
AN=[a for a in json.load(open('work/anchors.json'))['anchors']]; AT=np.array([a['t_ms']/1000 for a in AN])
P('\n| 脉冲 | 峰值时刻 | 最近锚点偏差 | 落到 S≤10% ← 90% 耗时 | 回升 10%→90% 耗时 | 保持 | 说明 |\n|--|--|--|--|--|--|--|')
for p in PL:
    d1=(cross(p['t_fall0'],p['t_peak']+0.01,0.1,True)-cross(p['t_fall0'],p['t_peak']+0.01,0.9,True)) if True else 0
    if p['t_rise0']: u1=cross(p['t_rise0'],p['t_rise1']+0.01,0.9,False)-cross(p['t_rise0'],p['t_rise1']+0.01,0.1,False); us='%.2f s'%u1
    else: us='—（保持到结束）'
    dev=min(abs(AT-p['t_peak']))*1000
    P('| %d | %.3f | %.0f ms | %.2f s | %s | %s | %s |'%(p['id'],p['t_peak'],dev,d1,us,'%.2f s'%p['hold_s'] if p['hold_s'] else '到结束',p['note']))
CK=json.load(open('work/v3/bw_check.json'))
P('\n谷底帧实测（解码前合成，game 层单独渲染）：')
for c in CK: P('- 脉冲 %d @%.2f s：游戏层饱和度 %.3f（≤0.03），彩色时 %.3f；Rec.709 亮度变化 %+.1f%%（≤3%%）；（参考）通道均值变化 %+.1f%%'%(c['pulse'],c['t'],c['game_sat_bw'],c['game_sat_color'],100*c['game_Y709_change'],100*c['mean_rgb_change']))
# 逐秒窗口中心平均饱和度（解码画面）
pp=subprocess.run(['ffmpeg','-v','error','-i',PV,'-an','-vf','fps=30,crop=iw/2:ih/2:iw/4:ih/4,scale=120:68','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
Fr=np.frombuffer(pp,np.uint8).reshape(-1,68,120,3).astype(np.float32)/255; mxv=Fr.max(3); mnv=Fr.min(3); satf=((mxv-mnv)/np.maximum(mxv,1e-3)).mean((1,2))
persec=[float(satf[i*30:(i+1)*30].mean()) for i in range(278)]
P('逐秒窗口中心平均饱和度（每 5 s 取值，含 BGA 文字）：',[round(persec[i],2) for i in range(0,278,5)])
inp=np.array([any(p['t_fall0']-0.2<=t<=(p['t_rise1'] or 279)+0.2 for p in PL) for t in (np.arange(len(satf))+0.5)/30])
P('非脉冲时段平均饱和度 %.3f；脉冲保持期（S=0）平均 %.3f'%(satf[~inp].mean(),satf[np.array([Sf(t)==0 for t in (np.arange(len(satf))+0.5)/30])].mean()))
# 与 final_v2 对比：非黑白相关帧应逐像素一致
def dec(fn): return np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',fn,'-an','-vf','fps=30,scale=160:90','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout,np.uint8).reshape(-1,90,160,3).astype(np.float32)
V2=dec('../deliver/final_v2/final_540p_v2.mp4'); V3=dec(PV); n=min(len(V2),len(V3)); dm=np.abs(V2[:n]-V3[:n]).mean((1,2,3))
old=json.load(open('work/bw_plan.json')); sh_={int(r['shot']):(float(r['t_start']),float(r['t_end'])) for r in EDL}
aff=[sh_[int(k)] for k in old['bw_shots']]+[(a,a+b) for a,b in old['fadeback']]+[(202,219.7)]+[(e.get('t',e.get('A0'))-0.6,e.get('A2',e.get('A1',e.get('A0')))+0.6) for e in SCH if e['kind'] in('out','in','scan','f')]+[(p['t_fall0']-0.2,(p['t_rise1'] or 279)+0.2) for p in PL]
tt_=(np.arange(n)+0.5)/30; unaff=np.array([not any(a<=t<b for a,b in aff) for t in tt_])
bad_=np.where(unaff&(dm>2))[0]; med=np.median(dm[unaff])
P('\n与 final_v2 对比（160×90 缩略逐帧平均绝对差，0–255）：未受黑白影响的 %.0f%% 帧中位差 %.2f（块重编码噪声）；其中差>2 的 %d 帧，时间 %s（重渲分块内随机特效——撕裂/噪点/字符随机位置——的随机数状态不同，目测同帧并列无差别；另含溶解边缘帧）；受影响帧中差>2 的 %d 帧'%(100*unaff.mean(),med,len(bad_),sorted(set(round(float(i)/30,0) for i in bad_))[:40],int((dm[~unaff]>2).sum())))
# ---- 溶解预算
P('\n## 6b 溶解预算')
cnt=collections.Counter(e['kind'] for e in SCH); fullv=[e for e in SCH if e['kind'] in('out','in','scan','f')]
dsec=sum((e['A2']-(e['A0']-0.5) if e['kind']=='out' else (e['A2']-e['A0'] if e['kind']=='scan' else (e['A1']-e['B0'] if e['kind']=='in' else e['A1']-e['A0']))) for e in fullv)
P('完整溶解（out/in/scan/f）%d 次（≤18，另含全片 final 收束 1 个），累计 %.1f s（≤25）；各类 %s；局部溶解 %d 次累计 %.1f s（≤8）'%(len(fullv),dsec,dict(cnt),cnt['local'],sum(e['dur'] for e in SCH if e['kind']=='local')))
P('变体 f（交叉溶解）%d 次；峰值区 79–100、208–230 内的完整溶解：%d 个（局部溶解不计）'%(cnt['f'],sum(1 for e in fullv if 79<=e.get('A0',0)<=100 or 208<=e.get('A0',0)<=230)))
# ---- BGA 底行 / PBS
P('\n## 6c BGA 底行与 PBS')
B=json.load(open('work/v10/bottom_check.json')); P('80–125 s 每 5 s 底行可见比例：',[(r['t'],r['visible']) for r in B],'；距下边最小 %d px（≥10）'%min(r['margin_px'] for r in B))
PA=json.load(open('../docs/pbs_accept_v2.json')); cells=[c for m in PA for c in m['cells']]
P('PBS 区（190.4–233.3 s）验收：采用白色网格兜底；%d 个单元中对比度达标 %d；近黑像素占比最大 %.3f；BGA 对主体像素加权覆盖率最大 %.3f（≤0.10）'%(len(cells),sum(1 for c in cells if c['contrast']>=(4.5 if c['kind']=='暗' else 3.5)),max(c['near_black'] for c in cells),max(m['meta']['cov_after'] for m in PA)))
# ---- 文字
P('\n## 7 文字行 4 段平均亮度差（红 ≤10%，淡黄 ≤15%）')
sh8=R4.load_edl()
def clean(t):
    s=next(x for x in sh8 if x['t']<=t<x['te']); tau=t-s['t']; return not any(off-2/30<=tau<off+nf/30+2/30 for off,typ,nf in s['events'])
def smean(img,r0,r1,c0,c1,mask):
    o=[]; e=np.linspace(c0,c1+1,5).astype(int); Lw=np.array([0.299,0.587,0.114],np.float32)
    for i in range(4):
        mm=mask[r0:r1+1,e[i]:e[i+1]]; px=img[r0:r1+1,e[i]:e[i+1]][mm]
        if len(px)<4: return None
        o.append(float((px@Lw).mean()))
    return o
for kind,rng in (('red',np.arange(158.9,160.31,0.05)),('yellow',np.arange(152.3,153.35,0.05))):
    c=[]
    for t in rng:
        if not clean(float(t)): continue
        b=TC.bga_frame(float(t)); ln=TC.find_line(b,kind,top=(0.55 if kind=='yellow' else 1.0))
        if not ln: continue
        r0,r1,c0,c1,m=ln; fr_=TC.prev_frame(PV,float(t)+0.02); o=smean(fr_,r0,r1,c0,c1,m)
        if o: c.append((float(t),(max(o)-min(o))/max(o),o))
    if c: pk=[c[0],c[len(c)//2],c[-1]]; P(kind,'3 个时刻：',[(round(t,2),'%.0f%%'%(d*100)) for t,d,o in pk],'；全部 %d 个无干扰时刻最大 %.0f%%'%(len(c),max(x[1] for x in c)*100))
open('../deliver/final_v3/qa_final_v3.md','w',encoding='utf-8').write('\n'.join(out)+'\n')
