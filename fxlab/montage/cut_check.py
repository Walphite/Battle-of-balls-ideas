"""切点实测：对样 mp4 解码，用画面帧差检测切点，用解码后的音频检测起音；报告
 ① 检测切点 vs 计划切点（帧取整）；② 切点 vs 校正后八分拍网格（拍点）；③ 切点 vs 解码音频中的强起音；④ AAC/封装延迟（解码音频与 mix.wav 互相关）。"""
import os,sys,json,subprocess,csv,numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,'.')
import onsets_fine as OF
def video_cuts(fn):
    p=subprocess.run(['ffmpeg','-v','error','-i',fn,'-an','-vf','scale=240:135','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True).stdout
    F=np.frombuffer(p,np.uint8).reshape(-1,135,240).astype(np.float32); d=np.abs(np.diff(F,axis=0)).mean((1,2)); med=np.median(d)
    idx=[i+1 for i in range(len(d)) if d[i]>max(10,6*med) and d[i]>=d[max(0,i-1)] and d[i]>=d[min(len(d)-1,i+1)]]; return idx,d
def audio(fn):
    p=subprocess.run(['ffmpeg','-v','error','-i',fn,'-vn','-ac','1','-ar','22050','-f','f32le','-'],capture_output=True).stdout; return np.frombuffer(p,np.float32)
def aac_lag(fn,t0):
    a=subprocess.run(['ffmpeg','-v','error','-i',fn,'-vn','-ac','1','-ar','48000','-f','f32le','-'],capture_output=True).stdout; a=np.frombuffer(a,np.float32)[:48000*8]
    b=subprocess.run(['ffmpeg','-v','error','-ss',f'{t0:.4f}','-t','8','-i','work/audio/mix.wav','-vn','-ac','1','-ar','48000','-f','f32le','-'],capture_output=True).stdout; b=np.frombuffer(b,np.float32)[:len(a)]
    n=1<<int(np.ceil(np.log2(len(a)+len(b)))); c=np.fft.irfft(np.fft.rfft(a,n)*np.conj(np.fft.rfft(b,n)),n); mx=2400; lags=np.r_[np.arange(0,mx),np.arange(-mx,0)]; cc=np.r_[c[:mx],c[n-mx:]]; return lags[int(np.argmax(cc))]/48000*1000
def run(name,t0,t1,edl='work/edl_v8.csv'):
    fn=f'work/preview/sample_{name}_540.mp4'; BT=np.array(json.load(open('work/beats_v4.json'))['beats']); E8=np.sort(np.r_[BT,(BT[:-1]+BT[1:])/2])
    R=list(csv.DictReader(open(edl,encoding='utf-8'))); anc_t=set()
    for r in R:
        if r.get('anchor_id_end'): anc_t.add(float(r['t_end']))
        if r.get('anchor_id'): anc_t.add(float(r['t_start']))
    plan=sorted({float(r['t_start']) for r in R}|{float(r['t_end']) for r in R}); plan=[p for p in plan if t0+0.1<p<t1-0.1]
    idx,d=video_cuts(fn); det=np.array([i/30+t0 for i in idx]); 
    x=audio(fn); dd,tt=OF.flux(x); pk,z=OF.pick(dd,tt); ons=tt[pk]-OF.calibrate()[0]+t0; st=z[pk]; strong=np.sort(ons[st>=np.percentile(st,70)])
    errs=[];grid=[];onsd=[];beatd=[];ordb=[];ordo=[]
    for p in plan:
        j=int(np.argmin(np.abs(det-p)))
        if abs(det[j]-p)>0.08: continue
        errs.append((det[j]-p)*1000); g=det[j]; beatd.append((g-BT[np.argmin(np.abs(BT-g))])*1000); grid.append((g-E8[np.argmin(np.abs(E8-g))])*1000); onsd.append((g-strong[np.argmin(np.abs(strong-g))])*1000)
        if all(abs(p-a)>0.001 for a in anc_t): ordb.append(beatd[-1]); ordo.append(onsd[-1])
    f=lambda a:(round(float(np.median(a)),1),round(float(np.mean(a)),1),round(float(np.max(np.abs(a))),1))
    print(name,'检测到切点 %d / 计划 %d'%(len(errs),len(plan)))
    print('  检测切点 − 计划切点 (中位,均值,最大|.|) ms',f(errs)); print('  切点 − 拍点 (中位,均值,最大|.|) ms',f(beatd)); print('  切点 − 八分拍网格(就近) ms',f(grid)); print('  普通切点(非锚点吸附 %d 个) − 拍点 ms'%len(ordb),f(ordb) if ordb else None,' 设计值 −25'); print('  普通切点 − 强起音 ms',f(ordo) if ordo else None); print('  切点 − 解码音频强起音(最近) ms',f(onsd)); lag=aac_lag(fn,t0); print('  解码音频相对 mix.wav 的延迟 %.1f ms'%lag)
    return dict(name=name,n=len(errs),plan_err=f(errs),beat=f(beatd),grid=f(grid),onset=f(onsd),ordinary_beat=f(ordb) if ordb else None,ordinary_onset=f(ordo) if ordo else None,ordb=ordb,aac_lag_ms=lag)
if __name__=='__main__':
    out=[run('H1',29.8333,44.9),run('H2',152.1667,168.2333)]; json.dump(out,open('work/cut_check.json','w'))
    allb=sum([o['ordb'] for o in out],[]); print('H1+H2 普通切点 %d 个：相对拍点 中位 %.1f 均值 %.1f ms（设计 −25）'%(len(allb),np.median(allb),np.mean(allb)))
