# -*- coding: utf-8 -*-
"""
02_train_eval.py  — baselines + HistGradientBoosting compound classifier.
Robust eval: temporal hold-out (train 2014-15, test 2016), per-regime PR-AUC,
bootstrap CIs, calibration, LOSO, ablations, single-pollutant (independent) baseline,
lead times 24/48/72 h. Writes results_gbt.json + figs/.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             f1_score, precision_recall_curve)
from sklearn.isotonic import IsotonicRegression
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
FIG = ROOT/"figs"; FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
df = pd.read_csv(ROOT/"modelling_daily.csv", parse_dates=["date"])

# ---------- feature matrix ----------
EXCLUDE = {"y24","y48","y72","pm_next","o3_next","pm_ex_next","o3_ex_next",
           "compound","pm_ex","o3_ex","thr_PM","thr_O3","date","year"}
CAT = ["Station","season","regime_name"]
num = [c for c in df.columns if c not in EXCLUDE and c not in CAT
       and df[c].dtype != object and c != "regime"]
dums = pd.get_dummies(df[CAT].astype("category"), dummy_na=False)
X_all = pd.concat([df[num], dums], axis=1)
feat_names = X_all.columns.tolist()
# feature groups for ablation
regime_cols = [c for c in feat_names if c.startswith("regime_name_")]
o3_today_cols = [c for c in feat_names if c.startswith("O3_8h") or "o3_" in c.lower()]
pm_today_cols = [c for c in feat_names if c.startswith("PM2.5") or "pm_" in c.lower()]

def split(target):
    m = df[target].notna()
    tr = m & df["year"].isin([2014,2015]); te = m & (df["year"]==2016)
    return (X_all[tr].values, df.loc[tr,target].values.astype(int), df[tr],
            X_all[te].values, df.loc[te,target].values.astype(int), df[te])

def metrics(y, p):
    return dict(PR_AUC=float(average_precision_score(y,p)),
                ROC_AUC=float(roc_auc_score(y,p)),
                Brier=float(brier_score_loss(y,p)),
                base_rate=float(y.mean()), n=int(len(y)), n_pos=int(y.sum()))

def boot_ci(y, p, fn=average_precision_score, B=1000):
    n=len(y); s=[]
    for _ in range(B):
        idx=rng.integers(0,n,n)
        if y[idx].sum()==0: continue
        s.append(fn(y[idx],p[idx]))
    return float(np.percentile(s,2.5)), float(np.percentile(s,97.5))

def hgb():
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
            max_iter=400, l2_regularization=1.0, min_samples_leaf=30,
            class_weight="balanced", random_state=0, early_stopping=True,
            validation_fraction=0.15)

RES = {}

# ===== primary: 24 h compound forecast =====
Xtr,ytr,dtr, Xte,yte,dte = split("y24")
RES["pos_rate"] = dict(train=float(ytr.mean()), test=float(yte.mean()),
                       n_train=len(ytr), n_test=len(yte))

# pooled vs within-regime dependence (Simpson check) on full labelled data
val = df.dropna(subset=["compound"])
pooled = val["compound"].mean()/(val["pm_ex"].mean()*val["o3_ex"].mean())
RES["pooled_daily_lift"] = round(float(pooled),3)
RES["pooled_daily_pearson_r"] = round(float(val["PM2.5_ugm3"].corr(val["O3_8h"])),3)

# --- baselines ---
# persistence: today's compound state
p_pers = dte["compound_today"].fillna(0).values
# climatology: train base rate by (season,regime_name)
clim = dtr.groupby(["season","regime_name"])["y24"].mean()
glob = ytr.mean()
p_clim = np.array([clim.get((s,r), glob) for s,r in zip(dte["season"],dte["regime_name"])])
# single-pollutant INDEPENDENT baseline: P(pm_ex_next)*P(o3_ex_next)
def fit_predict(target):
    m=df[target].notna(); tr=m&df["year"].isin([2014,2015]); te=m&(df["year"]==2016)
    g=hgb().fit(X_all[tr].values, df.loc[tr,target].astype(int).values)
    out=np.full(len(df),np.nan); out[te.values]=g.predict_proba(X_all[te].values)[:,1]
    return out
p_pm = fit_predict("pm_ex_next"); p_o3 = fit_predict("o3_ex_next")
te_idx = (df[df["y24"].notna()]["year"]==2016).values
mask_full = df["y24"].notna() & (df["year"]==2016)
p_indep = (p_pm*p_o3)[mask_full.values]

RES["baselines"] = {
 "persistence": metrics(yte,p_pers),
 "climatology_season_regime": metrics(yte,p_clim),
 "single_pollutant_independent": metrics(yte,p_indep),
}

# --- main GBT (joint, regime-aware) ---
clf = hgb().fit(Xtr,ytr)
p_gbt = clf.predict_proba(Xte)[:,1]
mm = metrics(yte,p_gbt); lo,hi = boot_ci(yte,p_gbt)
mm["PR_AUC_CI95"]=[round(lo,3),round(hi,3)]
# F1 threshold tuned on TRAIN
ptr = clf.predict_proba(Xtr)[:,1]
prec,rec,thrs = precision_recall_curve(ytr,ptr)
f1s = 2*prec*rec/(prec+rec+1e-9); tstar = thrs[max(0,np.argmax(f1s)-1)]
yhat = (p_gbt>=tstar).astype(int)
mm["F1_test"]=float(f1_score(yte,yhat)); mm["thr_star"]=float(tstar)
# recall @ precision>=0.3 on test PR curve
pP,rP,tP = precision_recall_curve(yte,p_gbt)
ok=pP>=0.3; mm["recall_at_prec0.3"]=float(rP[ok].max()) if ok.any() else 0.0
RES["GBT_joint_regime"]=mm

# --- ablation: GBT without regime one-hots ---
keep=[i for i,c in enumerate(feat_names) if c not in regime_cols]
clf2=hgb().fit(Xtr[:,keep],ytr); p2=clf2.predict_proba(Xte[:,keep])[:,1]
a=metrics(yte,p2); lo,hi=boot_ci(yte,p2); a["PR_AUC_CI95"]=[round(lo,3),round(hi,3)]
RES["GBT_no_regime"]=a

# --- lead-time degradation (48/72 h) ---
RES["lead_times"]={"24h":mm["PR_AUC"]}
for h in ["y48","y72"]:
    Xh,yh,_,Xhe,yhe,_=split(h)
    ch=hgb().fit(Xh,yh); ph=ch.predict_proba(Xhe)[:,1]
    RES["lead_times"][h.replace("y","")+"h"]=float(average_precision_score(yhe,ph))

# --- per-regime PR-AUC (test) ---
perreg={}
for rn,sub in dte.assign(p=p_gbt,y=yte).groupby("regime_name"):
    if sub["y"].sum()>=3:
        perreg[rn]=dict(PR_AUC=round(float(average_precision_score(sub["y"],sub["p"])),3),
                        base=round(float(sub["y"].mean()),3), n=int(len(sub)), n_pos=int(sub["y"].sum()))
RES["per_regime_PR_AUC"]=perreg

# --- LOSO (all years, leave-one-station-out) ---
loso=[]
for st in df["Station"].dropna().unique():
    m=df["y24"].notna()
    tr=m&(df["Station"]!=st); te=m&(df["Station"]==st)
    if df.loc[te,"y24"].sum()<5: continue
    c=hgb().fit(X_all[tr].values, df.loc[tr,"y24"].astype(int).values)
    pp=c.predict_proba(X_all[te].values)[:,1]
    loso.append(float(average_precision_score(df.loc[te,"y24"].astype(int).values,pp)))
RES["LOSO"]=dict(mean_PR_AUC=round(float(np.mean(loso)),3),
                 std=round(float(np.std(loso)),3), folds=len(loso),
                 per_fold=[round(x,3) for x in loso])

# --- calibration (isotonic) + reliability figure ---
iso=IsotonicRegression(out_of_bounds="clip").fit(ptr,ytr)
p_cal=iso.transform(p_gbt)
RES["GBT_joint_regime"]["Brier_calibrated"]=float(brier_score_loss(yte,p_cal))
bins=np.linspace(0,1,11); idx=np.digitize(p_cal,bins)-1
xs=[];ys=[]
for b in range(10):
    s=idx==b
    if s.sum()>5: xs.append(p_cal[s].mean()); ys.append(yte[s].mean())
plt.figure(figsize=(4,4)); plt.plot([0,1],[0,1],'k--',lw=1)
plt.plot(xs,ys,'o-'); plt.xlabel("predicted"); plt.ylabel("observed")
plt.title("Reliability (GBT, calibrated)"); plt.tight_layout(); plt.savefig(FIG/"reliability_gbt.png",dpi=130); plt.close()

# --- PR curves figure ---
plt.figure(figsize=(5,4))
for name,p in [("GBT joint",p_gbt),("no-regime",p2),("indep PM·O3",p_indep),("climatology",p_clim),("persistence",p_pers)]:
    pp,rr,_=precision_recall_curve(yte,p)
    plt.plot(rr,pp,label=f"{name} (AP={average_precision_score(yte,p):.2f})")
plt.axhline(yte.mean(),ls=':',c='grey',label=f"base={yte.mean():.2f}")
plt.xlabel("recall"); plt.ylabel("precision"); plt.legend(fontsize=7); plt.title("Compound 24h — PR curves (test 2016)")
plt.tight_layout(); plt.savefig(FIG/"pr_curves_gbt.png",dpi=130); plt.close()

json.dump(RES, open(ROOT/"results_gbt.json","w"), indent=2)
print(json.dumps(RES, indent=2))
