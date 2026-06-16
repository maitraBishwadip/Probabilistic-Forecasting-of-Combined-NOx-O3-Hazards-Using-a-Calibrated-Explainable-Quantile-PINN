# -*- coding: utf-8 -*-
"""
03_pinn.py  — Physics-Informed NN for compound PM2.5-O3 extremes.

Design (encodes the thesis: regime-conditioned joint dependence):
  trunk MLP -> marginal exceedance heads  p_PM, p_O3
            -> concentration heads ĉ_PM, ĉ_O3 (softplus, >=0)  [physics-regularised]
  compound prob  p_comp = clamp( p_PM * p_O3 * lift_r , 0,1 )
     lift_r = softplus(θ_r)  : LEARNED regime-indexed joint-dependence multiplier
              (lift_r>1 -> co-occur more than independence in regime r)

Physics: daily box mass-balance ties ĉ(t+1) to c(t) via learned, regime-indexed
deposition/dilution (a,b,d), O3 production (p), PM emission (q).

Loss = BCE(marginals) + focal(compound) + 0.3*MSE(ĉ,c_obs) + λ*MSE(ĉ,ĉ_phys)
Ablations: full / no-physics / no-regime / no-lift(independence).
Robust eval: PR-AUC(+bootstrap CI), ROC, Brier, per-regime, LOSO.
"""
import json, numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
FIG = ROOT/"figs"; FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
df = pd.read_csv(ROOT/"modelling_daily.csv", parse_dates=["date"])
K = 5
art = joblib.load(ROOT/"artefacts"/"artefacts.joblib")

EXCLUDE = {"y24","y48","y72","pm_next","o3_next","pm_ex_next","o3_ex_next",
           "compound","pm_ex","o3_ex","thr_PM","thr_O3","date","year","regime"}
CAT=["Station","season","regime_name"]
num=[c for c in df.columns if c not in EXCLUDE and c not in CAT and df[c].dtype!=object]
dums=pd.get_dummies(df[CAT].astype("category"),dummy_na=False).astype(float)
miss=df[["PM2.5_ugm3","O3_8h","modis_aod_550nm"]].isna().astype(float).add_suffix("_isna")
X=pd.concat([df[num],miss,dums],axis=1)

reg_idx=df["regime"].fillna(0).astype(int).clip(0,K-1).values
SC=dict(pm=100.0,o3=40.0,vc=3000.0,p=20.0,photo=8000.0)
def masks(t):
    m=df[t].notna().values
    return m&df["year"].isin([2014,2015]).values, m&(df["year"]==2016).values
tr,te=masks("y24")
mu=X[tr].mean(); sd=X[tr].std().replace(0,1)
Xs=((X-mu)/sd).fillna(0.0).values.astype("float32")

y=df["y24"].values.astype("float32")
pmx=df["pm_ex_next"].values.astype("float32"); o3x=df["o3_ex_next"].values.astype("float32")
obm=df["pm_ex_next"].notna().values.astype("float32"); obo=df["o3_ex_next"].notna().values.astype("float32")
cpm_n=(df["pm_next"]/SC["pm"]).values.astype("float32"); co3_n=(df["o3_next"]/SC["o3"]).values.astype("float32")
opm=df["pm_next"].notna().values.astype("float32"); oo3=df["o3_next"].notna().values.astype("float32")
cpm_t=(df["PM2.5_ugm3"]/SC["pm"]).fillna(0).values.astype("float32")
co3_t=(df["O3_8h"]/SC["o3"]).fillna(0).values.astype("float32")
vc=(df["ventilation_coefficient"]/SC["vc"]).fillna(0).values.astype("float32")
pr=(df["precip_mm"]/SC["p"]).fillna(0).values.astype("float32")
ph=(df["photochemical_activitiy_index"]/SC["photo"]).fillna(0).values.astype("float32")
T=lambda a: torch.tensor(a)
sp=torch.nn.functional.softplus

class PINN(nn.Module):
    def __init__(self,d,K,use_regime=True,use_lift=True):
        super().__init__(); self.use_regime=use_regime; self.use_lift=use_lift
        self.trunk=nn.Sequential(nn.Linear(d,96),nn.ReLU(),nn.Dropout(0.25),
                                 nn.Linear(96,64),nn.ReLU(),nn.Dropout(0.1))
        self.h_pm=nn.Linear(64,1); self.h_o3=nn.Linear(64,1); self.c_head=nn.Linear(64,2)
        n=K if use_regime else 1
        self.a=nn.Parameter(torch.zeros(n)); self.b=nn.Parameter(torch.zeros(n))
        self.d=nn.Parameter(torch.zeros(n)); self.p=nn.Parameter(torch.zeros(n)-1.0)
        self.q=nn.Parameter(torch.zeros(n)-1.0)
        nlift=K if (use_regime and use_lift) else 1
        self.lift=nn.Parameter(torch.zeros(nlift)+0.541)   # softplus(0.541)=1.0
    def forward(self,x,ridx):
        h=self.trunk(x)
        p_pm=torch.sigmoid(self.h_pm(h)).squeeze(-1); p_o3=torch.sigmoid(self.h_o3(h)).squeeze(-1)
        c=sp(self.c_head(h))
        if self.use_lift:
            sel=ridx if (self.use_regime) else torch.zeros_like(ridx)
            lift=sp(self.lift)[sel] if self.lift.numel()>1 else sp(self.lift)[0]
        else:
            lift=torch.ones_like(p_pm)
        p_comp=torch.clamp(p_pm*p_o3*lift,1e-6,1-1e-6)
        return p_pm,p_o3,c,p_comp
    def phys(self,cpt,cot,vct,prt,pht,ridx):
        sel=ridx if self.use_regime else torch.zeros_like(ridx)
        a,b,d,p,q=sp(self.a)[sel],sp(self.b)[sel],sp(self.d)[sel],sp(self.p)[sel],sp(self.q)[sel]
        return torch.stack([cpt*torch.exp(-(a*vct+b*prt+d))+q, cot*torch.exp(-(a*vct+d))+p*pht],1)

def focal_p(p,yv,alpha=0.75,gamma=2.0):
    p=torch.clamp(p,1e-6,1-1e-6); ce=-(yv*torch.log(p)+(1-yv)*torch.log(1-p))
    pt=torch.where(yv==1,p,1-p); w=torch.where(yv==1,alpha,1-alpha)
    return (w*(1-pt)**gamma*ce).mean()

def fit(use_phys=True,use_regime=True,use_lift=True,itr=None,ite=None,ep=400):
    itr=tr if itr is None else itr; ite=te if ite is None else ite
    net=PINN(Xs.shape[1],K,use_regime,use_lift)
    opt=torch.optim.Adam(net.parameters(),lr=3e-3,weight_decay=1e-4)
    idx=np.where(itr)[0]
    Xt=T(Xs[idx]); yt=T(y[idx]); rid=torch.tensor(reg_idx[idx])
    pmt=T(pmx[idx]); o3t=T(o3x[idx]); mbm=T(obm[idx]); mbo=T(obo[idx])
    cpn=T(cpm_n[idx]); con=T(co3_n[idx]); mpm=T(opm[idx]); mo3=T(oo3[idx])
    cpt=T(cpm_t[idx]); cot=T(co3_t[idx]); vct=T(vc[idx]); prt=T(pr[idx]); pht=T(ph[idx])
    # marginal pos_weights
    wpm=((mbm*(1-pmt)).sum()/((mbm*pmt).sum()+1e-6)); wo3=((mbo*(1-o3t)).sum()/((mbo*o3t).sum()+1e-6))
    for e in range(ep):
        net.train(); opt.zero_grad()
        p_pm,p_o3,c,p_comp=net(Xt,rid)
        bpm=nn.functional.binary_cross_entropy(p_pm,pmt,reduction="none",weight=torch.where(pmt==1,wpm,1.0))
        bo3=nn.functional.binary_cross_entropy(p_o3,o3t,reduction="none",weight=torch.where(o3t==1,wo3,1.0))
        Lm=(bpm*mbm).sum()/mbm.sum()+(bo3*mbo).sum()/mbo.sum()
        Lc=focal_p(p_comp,yt)
        Lreg=((c[:,0]-cpn)**2*mpm).sum()/mpm.sum()+((c[:,1]-con)**2*mo3).sum()/mo3.sum()
        Lph=((c-net.phys(cpt,cot,vct,prt,pht,rid))**2).mean() if use_phys else torch.tensor(0.0)
        lam=min(0.5,e/150*0.5)
        (Lm+Lc+0.3*Lreg+lam*Lph).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        jdx=np.where(ite)[0]
        _,_,_,pc=net(T(Xs[jdx]),torch.tensor(reg_idx[jdx]))
    return net, pc.numpy()

def M(yv,p): return dict(PR_AUC=float(average_precision_score(yv,p)),ROC_AUC=float(roc_auc_score(yv,p)),
                         Brier=float(brier_score_loss(yv,np.clip(p,0,1))),base=float(yv.mean()),
                         n=int(len(yv)),n_pos=int(yv.sum()))
def ci(yv,p,B=1000):
    n=len(yv);s=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        if yv[i].sum()>0: s.append(average_precision_score(yv[i],p[i]))
    return [round(float(np.percentile(s,2.5)),3),round(float(np.percentile(s,97.5)),3)]

RES={}; yv=y[te].astype(int)
net,p_full=fit(True,True,True);   m=M(yv,p_full); m["PR_AUC_CI95"]=ci(yv,p_full); RES["PINN_full"]=m
_,p_np=fit(False,True,True);      RES["PINN_no_physics"]=M(yv,p_np)
_,p_nr=fit(True,False,True);      RES["PINN_no_regime"]=M(yv,p_nr)
_,p_nl=fit(True,True,False);      RES["PINN_no_lift(independence)"]=M(yv,p_nl)

with torch.no_grad():
    names=art["names"]
    RES["learned_regime_lift"]={names[i]:round(float(sp(net.lift)[i]),3) for i in range(K)}
    RES["learned_physics_by_regime"]={names[i]:dict(
        dilution_a=round(float(sp(net.a)[i]),3), wet_b=round(float(sp(net.b)[i]),3),
        base_loss_d=round(float(sp(net.d)[i]),3), o3_prod_p=round(float(sp(net.p)[i]),3),
        pm_emit_q=round(float(sp(net.q)[i]),3)) for i in range(K)}

dte=df[te].copy(); dte["p"]=p_full; dte["yy"]=yv; per={}
for rn,sub in dte.groupby("regime_name"):
    if sub["yy"].sum()>=3:
        per[rn]=dict(PR_AUC=round(float(average_precision_score(sub["yy"],sub["p"])),3),n=int(len(sub)),n_pos=int(sub["yy"].sum()))
RES["per_regime_PR_AUC"]=per

loso=[]
for st in df["Station"].dropna().unique():
    mm=df["y24"].notna().values; itr=mm&(df["Station"]!=st).values; ite=mm&(df["Station"]==st).values
    if df["y24"][ite].sum()<5: continue
    _,pp=fit(True,True,True,itr,ite,ep=300)
    loso.append(float(average_precision_score(df["y24"][ite].astype(int).values,pp)))
RES["LOSO"]=dict(mean_PR_AUC=round(float(np.mean(loso)),3),std=round(float(np.std(loso)),3),folds=len(loso))

plt.figure(figsize=(5,4))
for nm,p in [("PINN full",p_full),("PINN no-physics",p_np),("PINN no-lift(indep)",p_nl)]:
    pp,rr,_=precision_recall_curve(yv,p); plt.plot(rr,pp,label=f"{nm} (AP={average_precision_score(yv,p):.2f})")
plt.axhline(yv.mean(),ls=':',c='grey',label=f"base={yv.mean():.2f}")
plt.xlabel("recall");plt.ylabel("precision");plt.legend(fontsize=7);plt.title("PINN — PR curves (test 2016)")
plt.tight_layout();plt.savefig(FIG/"pr_curves_pinn.png",dpi=130);plt.close()

json.dump(RES,open(ROOT/"results_pinn.json","w"),indent=2)
print(json.dumps(RES,indent=2))
