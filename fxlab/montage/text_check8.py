import sys,os,json,numpy as np
os.environ['PROBE']='1'; sys.path.insert(0,'.')
import render_v4 as R4, text_check as T
sh=R4.load_edl(); sp=lambda v:(max(v)-min(v))/max(max(v),1e-6)
def clean(t):
    s=next(x for x in sh if x['t']<=t<x['te']); tau=t-s['t']
    return not any(off-2/30<=tau<off+nf/30+2/30 for off,typ,nf in s['events'])
def stats_mean(img,r0,r1,c0,c1,mask):
    out=[]; edges=np.linspace(c0,c1+1,5).astype(int); Lw=np.array([0.299,0.587,0.114],np.float32)
    for i in range(4):
        mm=mask[r0:r1+1,edges[i]:edges[i+1]]; px=img[r0:r1+1,edges[i]:edges[i+1]][mm]
        if len(px)<4: out.append((None,None)); continue
        mx=px.max(1); mn=px.min(1); out.append((float((px@Lw).mean()),float(np.where(mx>0,(mx-mn)/np.maximum(mx,1),0).mean())))
    return out
H1,H2=29.8333,152.1667; rows=[]
for kind,rng,t0,vid in (('red',np.arange(158.9,160.31,0.05),H2,'work/preview/sample_H2_540.mp4'),('yellow',np.arange(152.3,153.35,0.05),H2,'work/preview/sample_H2_540.mp4')):
    c=[]
    for t in rng:
        if not clean(float(t)): continue
        b=T.bga_frame(float(t)); ln=T.find_line(b,kind,top=(0.55 if kind=='yellow' else 1.0))
        if not ln: continue
        r0,r1,c0,c1,m=ln; f=T.prev_frame(vid,float(t)-t0+0.02); a=stats_mean(f,r0,r1,c0,c1,m)
        if any(x[0] is None for x in a): continue
        bg=(f[r0:r1+1,c0:c1+1]@np.array([0.299,0.587,0.114],np.float32))[~m[r0:r1+1,c0:c1+1]].mean()
        c.append((float(t),sp([x[0] for x in a]),sp([x[1] for x in a]),float(bg),a))
    if kind=='red': pick=[c[0],c[len(c)//2],c[-1]]
    else: b=max(c,key=lambda x:x[3]); oth=[x for x in c if abs(x[0]-b[0])>0.7]; pick=[b,oth[0],oth[len(oth)//2]]
    for t,dl,ds,bg,a in pick:
        print(kind,'%.2f s'%t,'4 段(平均亮度,平均饱和度)',[(round(x[0]),round(x[1],2)) for x in a],'亮度差 %.0f%% 饱和度差 %.0f%% 行带内游戏层亮度 %.0f'%(dl*100,ds*100,bg)); rows.append(dict(kind=kind,t=round(t,2),seg=[(round(x[0]),round(x[1],2)) for x in a],dL=round(dl*100,1),dS=round(ds*100,1),bgL=round(bg)))
    print(kind,'全部无干扰时刻(%d 个)的最大亮度差 %.0f%%'%(len(c),max(x[1] for x in c)*100))
json.dump(rows,open('work/text_check_v8.json','w'))
