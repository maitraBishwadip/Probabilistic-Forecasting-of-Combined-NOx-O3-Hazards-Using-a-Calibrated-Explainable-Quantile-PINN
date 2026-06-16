# -*- coding: utf-8 -*-
"""
05_mlp_pinn.py — MLP-PINN (feed-forward physics-informed) for compound PM2.5-O3
extremes. Same physics box-residual + regime-lift as the LSTM variant but with a
feed-forward encoder (the recurrent encoder gave no gain on a 3-yr record).
Writes results_mlp_pinn.json + figs/.
"""
import json, copy, numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve, f1_score
from sklearn.isotonic import IsotonicRegression
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
ROOT=Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
FIG=ROOT/"figs"; FIG.mkdir(exist_ok=True); rng=np.random.default_rng(0)
df=pd.read_csv(ROOT/"modelling_daily.csv",parse_dates=["date"]).sort_values(["Station","date"]).reset_index(drop=True)
art=joblib.load(ROOT/"artefacts"/"artefacts.joblib"); names=art["names"]; K=5
L=14; SC=dict(pm=100.0,o3=40.0,vc=3000.0,p=20.0,photo=8000.0)
SEQ=["PM2.5_ugm3","O3_8h","NO2_ppb","NOx_ppb","CO_ppm","SO2_ppb","PM10_ugm3","temp_C","RH_pct",
     "wind_speed_ms","solar_rad_Wm2","precip_mm","boundary_layer_height_m","ventilation_coefficient",
     "photochemical_activitiy_index","surface_pressure_hPa","wind_speed_u_ms","wind_speed_v_ms",
     "geopotential_height_500hPa","modis_aod_550nm","doy_sin","doy_cos"]
for i in range(K): df[f"regime_oh{i}"]=(df["regime"]==i).astype(float)
SEQF=SEQ+[f"regime_oh{i}" for i in range(K)]
stations=sorted(df["Station"].dropna().unique()); SOH={s:i for i,s in enumerate(stations)}
trmask=df["year"].isin([2014,2015]); mu=df.loc[trmask,SEQ].mean(); sd=df.loc[trmask,SEQ].std().replace(0,1)
dfs=df.copy()
for c in SEQ: dfs[c]=(df[c]-mu[c])/sd[c]
dfs["PM_isna"]=df["PM2.5_ugm3"].isna().astype(float); dfs["O3_isna"]=df["O3_8h"].isna().astype(float)
SEQF=SEQF+["PM_isna","O3_isna"]
for c in SEQF: dfs[c]=dfs[c].fillna(0.0)
Xseq=[];stat=[];y24=[];y48=[];y72=[];reg=[];yr=[];st=[];pm_t=[];o3_t=[];vc=[];pr=[];ph=[];pm_n=[];o3_n=[];pmx=[];o3x=[];pers=[]
arr={c:{s:dfs[dfs.Station==s][c].values for s in stations} for c in SEQF}
raw={c:{s:df[df.Station==s][c].values for s in stations} for c in
     ["PM2.5_ugm3","O3_8h","ventilation_coefficient","precip_mm","photochemical_activitiy_index",
      "pm_next","o3_next","pm_ex_next","o3_ex_next","y24","y48","y72","regime","year","compound_today"]}
for s in stations:
    n=len(raw["y24"][s]); Fmat=np.stack([arr[c][s] for c in SEQF],axis=1)
    for i in range(L-1,n):
        if not np.isfinite(raw["y24"][s][i]): continue
        Xseq.append(Fmat[i-L+1:i+1]); st.append(s); stat.append(SOH[s])
        y24.append(raw["y24"][s][i]); y48.append(raw["y48"][s][i]); y72.append(raw["y72"][s][i])
        reg.append(int(raw["regime"][s][i]) if np.isfinite(raw["regime"][s][i]) else 0); yr.append(int(raw["year"][s][i]))
        pm_t.append(raw["PM2.5_ugm3"][s][i]); o3_t.append(raw["O3_8h"][s][i])
        vc.append(raw["ventilation_coefficient"][s][i]); pr.append(raw["precip_mm"][s][i]); ph.append(raw["photochemical_activitiy_index"][s][i])
        pmx.append(raw["pm_ex_next"][s][i]); o3x.append(raw["o3_ex_next"][s][i]); pm_n.append(raw["pm_next"][s][i]); o3_n.append(raw["o3_next"][s][i])
        pers.append(raw["compound_today"][s][i])
Xseq=np.array(Xseq,dtype="float32"); stat=np.array(stat); reg=np.array(reg); yr=np.array(yr); st=np.array(st)
y24=np.array(y24,dtype="float32"); y48=np.array(y48,dtype="float32"); y72=np.array(y72,dtype="float32")
f=lambda a: np.array(a,dtype="float32")
pm_t=f(pm_t)/SC["pm"]; o3_t=f(o3_t)/SC["o3"]; vc=f(vc)/SC["vc"]; pr=f(pr)/SC["p"]; ph=f(ph)/SC["photo"]
for a in (pm_t,o3_t,vc,pr,ph): np.nan_to_num(a,copy=False)
pmx=f(pmx); o3x=f(o3x); obm=np.isfinite(pmx).astype("float32"); obo=np.isfinite(o3x).astype("float32"); pmx=np.nan_to_num(pmx); o3x=np.nan_to_num(o3x)
cpm_n=f(pm_n)/SC["pm"]; co3_n=f(o3_n)/SC["o3"]; opm=np.isfinite(cpm_n).astype("float32"); oo3=np.isfinite(co3_n).astype("float32"); cpm_n=np.nan_to_num(cpm_n); co3_n=np.nan_to_num(co3_n); pers=np.nan_to_num(f(pers))
Fdim=Xseq.shape[2]; Sdim=len(stations)
print("sequences:",Xseq.shape,"| pos rate y24:",round(float(np.nanmean(y24)),3))
T=lambda a: torch.tensor(a); sp=torch.nn.functional.softplus

class PINN(nn.Module):
    def __init__(self,F,S,K,use_lstm=True,use_regime=True,use_lift=True):
        super().__init__(); self.use_lstm=use_lstm; self.use_regime=use_regime; self.use_lift=use_lift
        if use_lstm: self.enc=nn.LSTM(F,40,batch_first=True); hid=40
        else: self.enc=nn.Sequential(nn.Linear(F,40),nn.ReLU()); hid=40
        self.encdrop=nn.Dropout(0.35); self.stat=nn.Embedding(S,8)
        self.head=nn.Sequential(nn.Linear(hid+F+8,64),nn.ReLU(),nn.Dropout(0.3))
        self.h_pm=nn.Linear(64,1); self.h_o3=nn.Linear(64,1); self.c_head=nn.Linear(64,2)
        n=K if use_regime else 1
        self.a=nn.Parameter(torch.zeros(n)); self.b=nn.Parameter(torch.zeros(n)); self.d=nn.Parameter(torch.zeros(n))
        self.p=nn.Parameter(torch.zeros(n)-1.0); self.q=nn.Parameter(torch.zeros(n)-1.0)
        nl=K if (use_regime and use_lift) else 1; self.lift=nn.Parameter(torch.zeros(nl)+0.541)
    def forward(self,x,sid,ridx):
        if self.use_lstm: o,_=self.enc(x); h=self.encdrop(o[:,-1,:])
        else: h=self.enc(x[:,-1,:])
        z=self.head(torch.cat([h,x[:,-1,:],self.stat(sid)],1))
        p_pm=torch.sigmoid(self.h_pm(z)).squeeze(-1); p_o3=torch.sigmoid(self.h_o3(z)).squeeze(-1); c=sp(self.c_head(z))
        if self.use_lift:
            sel=ridx if self.use_regime else torch.zeros_like(ridx)
            lift=sp(self.lift)[sel] if self.lift.numel()>1 else sp(self.lift)[0]
        else: lift=torch.ones_like(p_pm)
        return p_pm,p_o3,c,torch.clamp(p_pm*p_o3*lift,1e-6,1-1e-6)
    def phys(self,cpt,cot,vct,prt,pht,ridx):
        sel=ridx if self.use_regime else torch.zeros_like(ridx)
        a,b,d,p,q=sp(self.a)[sel],sp(self.b)[sel],sp(self.d)[sel],sp(self.p)[sel],sp(self.q)[sel]
        return torch.stack([cpt*torch.exp(-(a*vct+b*prt+d))+q, cot*torch.exp(-(a*vct+d))+p*pht],1)

def focal_p(p,yv,al=0.75,g=2.0):
    p=torch.clamp(p,1e-6,1-1e-6); ce=-(yv*torch.log(p)+(1-yv)*torch.log(1-p))
    pt=torch.where(yv==1,p,1-p); w=torch.where(yv==1,al,1-al); return (w*(1-pt)**g*ce).mean()

def fit(target="y24",use_lstm=False,use_regime=True,use_lift=True,use_phys=True,itr=None,ite=None,ep=200,bs=256,patience=25):
    yt_all={"y24":y24,"y48":y48,"y72":y72}[target]; valid=np.isfinite(yt_all)
    if itr is None: itr=valid&np.isin(yr,[2014,2015]); ite=valid&(yr==2016)
    net=PINN(Fdim,Sdim,K,use_lstm,use_regime,use_lift); opt=torch.optim.Adam(net.parameters(),lr=1.5e-3,weight_decay=1e-4)
    gidx=np.where(itr)[0]; rs=np.random.default_rng(1).permutation(len(gidx)); nval=max(50,int(0.15*len(gidx)))
    vsel=gidx[rs[:nval]]; idx=gidx[rs[nval:]]
    Xt=T(Xseq[idx]); sid=torch.tensor(stat[idx]); rid=torch.tensor(reg[idx]); yt=T(yt_all[idx].astype("float32"))
    pmt=T(pmx[idx]); o3t=T(o3x[idx]); mbm=T(obm[idx]); mbo=T(obo[idx]); cpn=T(cpm_n[idx]); con=T(co3_n[idx]); mpm=T(opm[idx]); mo3=T(oo3[idx])
    cpt=T(pm_t[idx]); cot=T(o3_t[idx]); vct=T(vc[idx]); prt=T(pr[idx]); pht=T(ph[idx])
    Xv=T(Xseq[vsel]); sidv=torch.tensor(stat[vsel]); ridv=torch.tensor(reg[vsel]); yv_=yt_all[vsel].astype(int)
    wpm=((mbm*(1-pmt)).sum()/((mbm*pmt).sum()+1e-6)); wo3=((mbo*(1-o3t)).sum()/((mbo*o3t).sum()+1e-6))
    N=len(idx); best=-1; best_state=copy.deepcopy(net.state_dict()); bad=0
    for e in range(ep):
        net.train(); perm=torch.randperm(N)
        for j in range(0,N,bs):
            b=perm[j:j+bs]; opt.zero_grad(); p_pm,p_o3,c,pc=net(Xt[b],sid[b],rid[b])
            bpm=nn.functional.binary_cross_entropy(p_pm,pmt[b],reduction="none",weight=torch.where(pmt[b]==1,wpm,1.0))
            bo3=nn.functional.binary_cross_entropy(p_o3,o3t[b],reduction="none",weight=torch.where(o3t[b]==1,wo3,1.0))
            Lm=(bpm*mbm[b]).sum()/(mbm[b].sum()+1e-6)+(bo3*mbo[b]).sum()/(mbo[b].sum()+1e-6); Lc=focal_p(pc,yt[b])
            Lreg=((c[:,0]-cpn[b])**2*mpm[b]).sum()/(mpm[b].sum()+1e-6)+((c[:,1]-con[b])**2*mo3[b]).sum()/(mo3[b].sum()+1e-6)
            lam=min(0.5,e/60*0.5); Lph=((c-net.phys(cpt[b],cot[b],vct[b],prt[b],pht[b],rid[b]))**2).mean() if use_phys else torch.tensor(0.0)
            (Lm+Lc+0.3*Lreg+(lam*Lph if use_phys else 0.0)).backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        net.eval()
        with torch.no_grad(): _,_,_,pcv=net(Xv,sidv,ridv)
        try: vscore=average_precision_score(yv_,pcv.numpy()) if yv_.sum()>0 else 0
        except Exception: vscore=0
        if vscore>best: best=vscore; best_state=copy.deepcopy(net.state_dict()); bad=0
        else:
            bad+=1
            if bad>=patience: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        jdx=np.where(ite)[0]
        _,_,_,pc=net(T(Xseq[jdx]),torch.tensor(stat[jdx]),torch.tensor(reg[jdx]))
        _,_,_,ptr=net(T(Xseq[gidx]),torch.tensor(stat[gidx]),torch.tensor(reg[gidx]))
    return net, pc.numpy(), yt_all[ite].astype(int), ptr.numpy(), yt_all[gidx].astype(int)

def M(yv,p): return dict(PR_AUC=float(average_precision_score(yv,p)),ROC_AUC=float(roc_auc_score(yv,p)),
                         Brier=float(brier_score_loss(yv,np.clip(p,0,1))),base=float(yv.mean()),n=int(len(yv)),n_pos=int(yv.sum()))
def ci(yv,p,B=1000):
    n=len(yv);s=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        if yv[i].sum()>0: s.append(average_precision_score(yv[i],p[i]))
    return [round(float(np.percentile(s,2.5)),3),round(float(np.percentile(s,97.5)),3)]

RES={}
net,p,yv,ptr,ytr=fit("y24",use_lstm=False,use_regime=True,use_lift=True)
m=M(yv,p); m["PR_AUC_CI95"]=ci(yv,p)
prec,rec,thr=precision_recall_curve(ytr,ptr); f1=2*prec*rec/(prec+rec+1e-9); tstar=thr[max(0,np.argmax(f1)-1)]
m["F1_test"]=float(f1_score(yv,(p>=tstar).astype(int))); m["thr_star"]=float(tstar)
pP,rP,_=precision_recall_curve(yv,p); ok=pP>=0.3; m["recall_at_prec0.3"]=float(rP[ok].max()) if ok.any() else 0.0
iso=IsotonicRegression(out_of_bounds="clip").fit(ptr,ytr); pcal=iso.transform(p); m["Brier_calibrated"]=float(brier_score_loss(yv,pcal))
RES["MLP_PINN"]=m
RES["reference"]=dict(base_rate=float(yv.mean()),persistence_PR_AUC=float(average_precision_score(yv, pers[(np.isfinite(y24))&(yr==2016)])))
def quick(**k):
    _,pp,yy,_,_=fit("y24",ep=110,**k); return round(float(average_precision_score(yy,pp)),3), round(float(roc_auc_score(yy,pp)),3)
RES["ablation"]={"MLP_PINN(full)":dict(PR_AUC=round(m["PR_AUC"],3),ROC_AUC=round(m["ROC_AUC"],3))}
ap,ro=quick(use_lstm=True);  RES["ablation"]["with_LSTM_encoder"]=dict(PR_AUC=ap,ROC_AUC=ro)
ap,ro=quick(use_lstm=False,use_phys=False); RES["ablation"]["no_physics"]=dict(PR_AUC=ap,ROC_AUC=ro)
ap,ro=quick(use_lstm=False,use_lift=False); RES["ablation"]["no_regime_lift(independence)"]=dict(PR_AUC=ap,ROC_AUC=ro)
ap,ro=quick(use_lstm=False,use_regime=False); RES["ablation"]["no_regime_params"]=dict(PR_AUC=ap,ROC_AUC=ro)
RES["lead_times"]={"24h":round(m["PR_AUC"],3)}
for tg in ["y48","y72"]:
    _,pp,yy,_,_=fit(tg,use_lstm=False,ep=110); RES["lead_times"][tg.replace("y","")+"h"]=round(float(average_precision_score(yy,pp)),3)
ite=(np.isfinite(y24))&(yr==2016); regte=reg[ite]; per={}
for i in range(K):
    sel=regte==i
    if y24[ite][sel].sum()>=3:
        per[names[i]]=dict(PR_AUC=round(float(average_precision_score(y24[ite][sel].astype(int),p[sel])),3),n=int(sel.sum()),n_pos=int(y24[ite][sel].sum()))
RES["per_regime_PR_AUC"]=per
with torch.no_grad():
    RES["learned_regime_lift"]={names[i]:round(float(sp(net.lift)[i]),3) for i in range(K)}
    RES["learned_physics_by_regime"]={names[i]:dict(dilution_a=round(float(sp(net.a)[i]),3),wet_b=round(float(sp(net.b)[i]),3),
        base_loss_d=round(float(sp(net.d)[i]),3),o3_prod_p=round(float(sp(net.p)[i]),3),pm_emit_q=round(float(sp(net.q)[i]),3)) for i in range(K)}
loso=[]
for s in stations:
    v=np.isfinite(y24); itr=v&(st!=s); ite2=v&(st==s)
    if y24[ite2].sum()<5: continue
    _,pp,yy,_,_=fit("y24",use_lstm=False,itr=itr,ite=ite2,ep=90); loso.append(float(average_precision_score(yy,pp)))
RES["LOSO"]=dict(mean_PR_AUC=round(float(np.mean(loso)),3),std=round(float(np.std(loso)),3),folds=len(loso),per_fold=[round(x,3) for x in loso])
plt.figure(figsize=(5,4)); pp,rr,_=precision_recall_curve(yv,p); plt.plot(rr,pp,label=f"MLP-PINN (AP={m['PR_AUC']:.2f})")
plt.axhline(yv.mean(),ls=':',c='grey',label=f"base rate={yv.mean():.2f}")
plt.xlabel("recall");plt.ylabel("precision");plt.legend();plt.title("MLP-PINN — compound 24h PR curve (test 2016)")
plt.tight_layout();plt.savefig(FIG/"pr_mlp_pinn.png",dpi=130);plt.close()
bins=np.linspace(0,1,11); idx=np.digitize(pcal,bins)-1; xs=[];ys=[]
for bn in range(10):
    sb=idx==bn
    if sb.sum()>5: xs.append(pcal[sb].mean()); ys.append(yv[sb].mean())
plt.figure(figsize=(4,4)); plt.plot([0,1],[0,1],'k--',lw=1); plt.plot(xs,ys,'o-'); plt.xlabel("predicted");plt.ylabel("observed");plt.title("MLP-PINN reliability (calibrated)")
plt.tight_layout();plt.savefig(FIG/"reliability_mlp_pinn.png",dpi=130);plt.close()
np.save(ROOT/"artefacts"/"pred_mlp_pinn.npy", np.vstack([yv, p]))
json.dump(RES,open(ROOT/"results_mlp_pinn.json","w"),indent=2)
print(json.dumps(RES,indent=2))
