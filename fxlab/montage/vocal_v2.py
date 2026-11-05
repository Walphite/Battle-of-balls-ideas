"""第 4 条：100–188s 重检人声/采样起音；202–219s 沿用 analyze_v2_seg 候选。输出 work/vocal_onsets.json"""
import numpy as np,sys,os,pickle,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import analyze_v2 as V
FPS=V.FPS
x=V.load(V.MUS); mag=V.stft(x); n=len(mag); t=np.arange(n)/FPS
f=np.fft.rfftfreq(V.NFFT,1/V.SR)
vb=(f>=300)&(f<3400); lo=(f<200); hi=(f>=4000)
lm=np.log1p(10*mag)
voc_e=lm[:,vb].mean(1); lo_e=lm[:,lo].mean(1); hi_e=lm[:,hi].mean(1)
on=np.maximum(np.diff(lm[:,vb],axis=0),0).mean(1); on=np.r_[0,on]
# 组合分：人声频段起音(z) + 人声频段能量相对低/高频的占比
ratio=voc_e-(lo_e+hi_e)/2
score=V.z(on)+0.7*V.z(ratio)
score=np.convolve(score,np.ones(3)/3,'same')
def top(a,b,k,thr=1.0,mind=0.15):
    m=(t>=a)&(t<b); idx=np.where(m)[0]
    s=score.copy(); s[~m]=-9
    pk=V.peaks(s,thr,int(mind*FPS))
    pk=pk[np.argsort(-s[pk])][:k]
    return sorted([(round(float(t[i]),2),round(float(s[i]),2)) for i in pk])
corp=top(100,188,40)
P=pickle.load(open('work/an4.pkl','rb'))
frag=[(float(a),float(b)) for a,b in P['voc'] if 202<=a<=219.5]
# 202–219 补充：用同一组合分补到密度合理（每 0.3s 最多 1 个）
frag2=top(202,219.5,40,thr=0.8,mind=0.3)
frag_all=sorted(set([a for a,_ in frag]+[a for a,_ in frag2]))
early=[(float(a),float(b)) for a,b in P['voc'] if a<80]
json.dump(dict(corp_100_188=corp,frag_202_219_old=frag,frag_202_219_new=frag2,frag_merged=frag_all,early_old=early),open('work/vocal_onsets.json','w'),indent=0)
print('corp',len(corp),corp)
print('frag old',len(frag),'new',len(frag2),'merged',len(frag_all),frag_all)
