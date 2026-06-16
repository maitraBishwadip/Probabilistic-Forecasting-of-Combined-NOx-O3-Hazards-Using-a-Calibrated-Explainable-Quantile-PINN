# -*- coding: utf-8 -*-
"""
06_rf_purelstm.py — Random Forest and a PURE LSTM (no physics) for the compound
PM2.5-O3 24h forecast, evaluated on the SAME sample set as the PINN (05) so the
three models are directly comparable. Writes results_rf.json, results_pure_lstm.json,
and saved test predictions for the combined PR figure.
"""
import json, copy, numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve, f1_score
from sklearn.isotonic import IsotonicRegression
torch.manual_seed(0); np.random.seed(0)
ROOT=Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
(ROOT/"artefacts").mkdir(exist_ok=True); rng=np.random.default_rng(0)
df=pd.read_csv(ROOT/"modelling_daily.csv",parse_dates=["date"]).sort_values(["Station","date"]).reset_index(drop=True)
art=joblib.load(ROOT/"artefacts"/"artefacts.joblib"); names=art["names"]; K=5; L=14
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
SEQF2=SEQF+["PM_isna","O3_isna"]
for c in SEQF2: dfs[c]=dfs[c].fillna(0.0)

Xseq=[]; Xtab=[]; stat=[]; reg=[]; yr=[]; st=[]; y24=[]; y48=[]; y72=[]; pers=[]
arrs={c:{s:dfs[dfs.Station==s][c].values for s in stations} for c in SEQF2}
rawseq={c:{s:df[df.Station==s][c].values for s in stations} for c in SEQ}
rawreg={s:df[df.Station==s][[f"regime_oh{i}" for i in range(K)]].values for s in stations}
meta={c:{s:df[df.Station==s][c].values for s in stations} for c in ["y24","y48","y72","regime","year","compound_today"]}
for s in stations:
    n=len(meta["y24"][s]); Fmat=np.stack([arrs[c][s] for c in SEQF2],axis=1)
    rawM=np.stack([rawseq[c][s] for c in SEQ],axis=1)
    for i in range(L-1,n):
        if not np.isfinite(meta["y24"][s][i]): continue
        Xseq.append(Fmat[i-L+1:i+1])
        # tabular: last-day raw + regime one-hot + PM/O3 lags (1,3,7) + station idx
        lastraw=rawM[i]
        lags=[rawseq["PM2.5_ugm3"][s][i-1],rawseq["PM2.5_ugm3"][s][i-3],rawseq["PM2.5_ugm3"][s][i-7],
              rawseq["O3_8h"][s][i-1],rawseq["O3_8h"][s][i-3],rawseq["O3_8h"][s][i-7]]
        Xtab.append(np.concatenate([lastraw, rawreg[s][i], lags, [SOH[s]]]))
        stat.append(SOH[s]); reg.append(int(meta["regime"][s][i]) if np.isfinite(meta["regime"][s][i]) else 0); yr.append(int(meta["year"][s][i])); st.append(s)
        y24.append(meta["y24"][s][i]); y48.append(meta["y48"][s][i]); y72.append(meta["y72"][s][i]); pers.append(meta["compound_today"][s][i])
Xseq=np.array(Xseq,dtype="float32"); Xtab=np.array(Xtab,dtype="float32")
stat=np.array(stat); reg=np.array(reg); yr=np.array(yr); st=np.array(st)
y24=np.array(y24); y48=np.array(y48); y72=np.array(y72); pers=np.nan_to_num(np.array(pers,dtype=float))
Fdim=Xseq.shape[2]; Sdim=len(stations)
print("samples:",Xseq.shape,"tab:",Xtab.shape,"pos y24:",round(float(np.nanmean(y24)),3))

def M(yv,p): return dict(PR_AUC=float(average_precision_score(yv,p)),ROC_AUC=float(roc_auc_score(yv,p)),
                         Brier=float(brier_score_loss(yv,np.clip(p,0,1))),base=float(yv.mean()),n=int(len(yv)),n_pos=int(yv.sum()))
def ci(yv,p,B=1000):
    n=len(yv);s=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        if yv[i].sum()>0: s.append(average_precision_score(yv[i],p[i]))
    return [round(float(np.percentile(s,2.5)),3),round(float(np.percentile(s,97.5)),3)]
def extras(yv,p,ptr,ytr):
    prec,rec,thr=precision_recall_curve(ytr,ptr); f1=2*prec*rec/(prec+rec+1e-9); tstar=thr[max(0,np.argmax(f1)-1)]
    out=dict(F1_test=float(f1_score(yv,(p>=tstar).astype(int))),thr_star=float(tstar))
    pP,rP,_=precision_recall_curve(yv,p); ok=pP>=0.3; out["recall_at_prec0.3"]=float(rP[ok].max()) if ok.any() else 0.0
    iso=IsotonicRegression(out_of_bounds="clip").fit(ptr,ytr); out["Brier_calibrated"]=float(brier_score_loss(yv,iso.transform(p)))
    return out

# ============ Random Forest ============
def rf_fit(Xt,yt):
    return RandomForestClassifier(n_estimators=500,min_samples_leaf=5,max_features="sqrt",
            class_weight="balanced_subsample",random_state=0,n_jobs=-1).fit(Xt,yt)
def split(tgt):
    yv_all={"y24":y24,"y48":y48,"y72":y72}[tgt]; m=np.isfinite(yv_all)
    tr=m&np.isin(yr,[2014,2015]); te=m&(yr==2016); return tr,te,yv_all
imp=SimpleImputer(strategy="median")
tr,te,_=split("y24"); Xti=imp.fit_transform(Xtab[tr]); Xtei=imp.transform(Xtab[te])
rf=rf_fit(Xti,y24[tr].astype(int)); p_rf=rf.predict_proba(Xtei)[:,1]; ptr_rf=rf.predict_proba(Xti)[:,1]
yv=y24[te].astype(int)
RF={"RandomForest":{**M(yv,p_rf),"PR_AUC_CI95":ci(yv,p_rf),**extras(yv,p_rf,ptr_rf,y24[tr].astype(int))}}
RF["reference"]=dict(base_rate=float(yv.mean()),persistence_PR_AUC=float(average_precision_score(yv,pers[te])))
# per-regime
per={}
for i in range(K):
    sel=reg[te]==i
    if yv[sel].sum()>=3: per[names[i]]=dict(PR_AUC=round(float(average_precision_score(yv[sel],p_rf[sel])),3),n=int(sel.sum()),n_pos=int(yv[sel].sum()))
RF["per_regime_PR_AUC"]=per
# lead times
RF["lead_times"]={"24h":round(RF["RandomForest"]["PR_AUC"],3)}
for tg in ["y48","y72"]:
    trL,teL,yvA=split(tg); Xi=imp.fit_transform(Xtab[trL]); Xei=imp.transform(Xtab[teL])
    r=rf_fit(Xi,yvA[trL].astype(int)); pp=r.predict_proba(Xei)[:,1]
    RF["lead_times"][tg.replace("y","")+"h"]=round(float(average_precision_score(yvA[teL].astype(int),pp)),3)
# LOSO
loso=[]
for s in stations:
    m=np.isfinite(y24); itr=m&(st!=s); ite=m&(st==s)
    if y24[ite].sum()<5: continue
    Xi=imp.fit_transform(Xtab[itr]); Xei=imp.transform(Xtab[ite])
    r=rf_fit(Xi,y24[itr].astype(int)); pp=r.predict_proba(Xei)[:,1]
    loso.append(float(average_precision_score(y24[ite].astype(int),pp)))
RF["LOSO"]=dict(mean_PR_AUC=round(float(np.mean(loso)),3),std=round(float(np.std(loso)),3),folds=len(loso))
# feature importances (top)
fnames=SEQ+[f"regime_oh{i}" for i in range(K)]+["PM_lag1","PM_lag3","PM_lag7","O3_lag1","O3_lag3","O3_lag7","station"]
imp_sorted=sorted(zip(fnames,rf.feature_importances_),key=lambda x:-x[1])[:12]
RF["top_features"]={k:round(float(v),4) for k,v in imp_sorted}
np.save(ROOT/"artefacts"/"pred_rf.npy",np.vstack([yv,p_rf]))
json.dump(RF,open(ROOT/"results_rf.json","w"),indent=2)
print("RF done: PR-AUC",round(RF["RandomForest"]["PR_AUC"],3))

# ============ Pure LSTM (no physics) ============
T=lambda a: torch.tensor(a)
class PureLSTM(nn.Module):
    def __init__(self,F,S):
        super().__init__(); self.lstm=nn.LSTM(F,48,batch_first=True); self.drop=nn.Dropout(0.3)
        self.stat=nn.Embedding(S,8); self.fc=nn.Sequential(nn.Linear(48+8,32),nn.ReLU(),nn.Dropout(0.2),nn.Linear(32,1))
    def forward(self,x,sid):
        o,_=self.lstm(x); h=self.drop(o[:,-1,:]); return self.fc(torch.cat([h,self.stat(sid)],1)).squeeze(-1)
def focal_logit(logit,yv,al=0.75,g=2.0):
    p=torch.sigmoid(logit); ce=nn.functional.binary_cross_entropy_with_logits(logit,yv,reduction="none")
    pt=torch.where(yv==1,p,1-p); w=torch.where(yv==1,al,1-al); return (w*(1-pt)**g*ce).mean()
def lstm_fit(tgt="y24",itr=None,ite=None,ep=160,bs=256,patience=22):
    yv_all={"y24":y24,"y48":y48,"y72":y72}[tgt]; valid=np.isfinite(yv_all)
    if itr is None: itr=valid&np.isin(yr,[2014,2015]); ite=valid&(yr==2016)
    net=PureLSTM(Fdim,Sdim); opt=torch.optim.Adam(net.parameters(),lr=1.5e-3,weight_decay=1e-4)
    gidx=np.where(itr)[0]; rs=np.random.default_rng(1).permutation(len(gidx)); nval=max(50,int(0.15*len(gidx)))
    vsel=gidx[rs[:nval]]; idx=gidx[rs[nval:]]
    Xt=T(Xseq[idx]); sid=torch.tensor(stat[idx]); yt=T(yv_all[idx].astype("float32"))
    Xv=T(Xseq[vsel]); sidv=torch.tensor(stat[vsel]); yv_=yv_all[vsel].astype(int)
    N=len(idx); best=-1; bs_=copy.deepcopy(net.state_dict()); bad=0
    for e in range(ep):
        net.train(); perm=torch.randperm(N)
        for j in range(0,N,bs):
            b=perm[j:j+bs]; opt.zero_grad(); lg=net(Xt[b],sid[b]); focal_logit(lg,yt[b]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        net.eval()
        with torch.no_grad(): pv=torch.sigmoid(net(Xv,sidv)).numpy()
        try: vs=average_precision_score(yv_,pv) if yv_.sum()>0 else 0
        except Exception: vs=0
        if vs>best: best=vs; bs_=copy.deepcopy(net.state_dict()); bad=0
        else:
            bad+=1
            if bad>=patience: break
    net.load_state_dict(bs_); net.eval()
    with torch.no_grad():
        jdx=np.where(ite)[0]; p=torch.sigmoid(net(T(Xseq[jdx]),torch.tensor(stat[jdx]))).numpy()
        ptr=torch.sigmoid(net(T(Xseq[gidx]),torch.tensor(stat[gidx]))).numpy()
    return p, yv_all[ite].astype(int), ptr, yv_all[gidx].astype(int)

p_l,yv,ptr_l,ytr_l=lstm_fit("y24")
LS={"PureLSTM":{**M(yv,p_l),"PR_AUC_CI95":ci(yv,p_l),**extras(yv,p_l,ptr_l,ytr_l)}}
LS["reference"]=dict(base_rate=float(yv.mean()),persistence_PR_AUC=float(average_precision_score(yv,pers[(np.isfinite(y24))&(yr==2016)])))
per={}
te=(np.isfinite(y24))&(yr==2016)
for i in range(K):
    sel=reg[te]==i
    if y24[te][sel].sum()>=3: per[names[i]]=dict(PR_AUC=round(float(average_precision_score(y24[te][sel].astype(int),p_l[sel])),3),n=int(sel.sum()),n_pos=int(y24[te][sel].sum()))
LS["per_regime_PR_AUC"]=per
LS["lead_times"]={"24h":round(LS["PureLSTM"]["PR_AUC"],3)}
for tg in ["y48","y72"]:
    pp,yy,_,_=lstm_fit(tg,ep=140); LS["lead_times"][tg.replace("y","")+"h"]=round(float(average_precision_score(yy,pp)),3)
loso=[]
for s in stations:
    m=np.isfinite(y24); itr=m&(st!=s); ite=m&(st==s)
    if y24[ite].sum()<5: continue
    pp,yy,_,_=lstm_fit("y24",itr=itr,ite=ite,ep=110); loso.append(float(average_precision_score(yy,pp)))
LS["LOSO"]=dict(mean_PR_AUC=round(float(np.mean(loso)),3),std=round(float(np.std(loso)),3),folds=len(loso))
np.save(ROOT/"artefacts"/"pred_pure_lstm.npy",np.vstack([yv,p_l]))
json.dump(LS,open(ROOT/"results_pure_lstm.json","w"),indent=2)
print("PureLSTM done: PR-AUC",round(LS["PureLSTM"]["PR_AUC"],3))
