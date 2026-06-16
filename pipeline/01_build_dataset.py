# -*- coding: utf-8 -*-
"""
01_build_dataset.py
QC + cleaning + daily aggregation + weather regimes + seasonal-75th compound
labels (thresholds fit on TRAIN years only) + forecasting feature engineering.
Outputs: modelling_daily.csv, artefacts/{scaler,kmeans,thresholds}.joblib, qc_report.json
"""
import json, warnings, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
ART = ROOT / "artefacts"; ART.mkdir(exist_ok=True)
RAW = ROOT / "BD_DOE_2014-16.csv"
TRAIN_YEARS = [2014, 2015]          # thresholds + regimes fit here only (no leakage)
Q = 0.75                            # extreme percentile
K_REGIMES = 5
report = {}

# ---------- load ----------
df = pd.read_csv(RAW, low_memory=False)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
df = df.dropna(subset=["dt"]).sort_values(["Station", "dt"]).reset_index(drop=True)

POLL = ["SO2_ppb","NO_ppb","NO2_ppb","NOx_ppb","CO_ppm","O3_ppb","PM2.5_ugm3","PM10_ugm3"]
MET = ["temp_C","RH_pct","wind_speed_ms","wind_dir_deg","solar_rad_Wm2","precip_mm",
       "surface_pressure_hPa","boundary_layer_height_m","wind_speed_u_ms","wind_speed_v_ms",
       "dilution_volume_index","ventilation_coefficient","photochemical_activitiy_index",
       "geopotential_height_1000hPa","geopotential_height_850hPa","geopotential_height_500hPa"]
SAT = ["modis_aod_550nm","modis_ndvi","firms_fire_count","firms_frp_mw"]
for c in POLL + MET + SAT:
    df[c] = pd.to_numeric(df[c], errors="coerce")

report["raw_rows"] = int(len(df))

# ---------- QC: physical range ----------
RANGES = {"SO2_ppb":(0,1000),"NO_ppb":(0,1000),"NO2_ppb":(0,1000),"NOx_ppb":(0,2000),
          "CO_ppm":(0,50),"O3_ppb":(0,300),"PM2.5_ugm3":(0,1500),"PM10_ugm3":(0,2000),
          "temp_C":(-5,55),"RH_pct":(0,100),"wind_speed_ms":(0,60),"solar_rad_Wm2":(0,1400),
          "precip_mm":(0,300),"surface_pressure_hPa":(850,1080),"boundary_layer_height_m":(0,5000)}
qc_range = {}
for c,(lo,hi) in RANGES.items():
    bad = ((df[c] < lo) | (df[c] > hi))
    qc_range[c] = int(bad.sum()); df.loc[bad, c] = np.nan
report["qc_range_removed"] = qc_range

# ---------- QC: flat-line (>=24 identical consecutive, value>0) ----------
def flatline_mask(s, run=24):
    g = (s != s.shift()).cumsum()
    sz = s.groupby(g).transform("size")
    return (sz >= run) & (s > 0) & s.notna()
qc_flat = {}
for c in ["PM2.5_ugm3","O3_ppb","NO2_ppb","NOx_ppb","CO_ppm","SO2_ppb","PM10_ugm3"]:
    m = df.groupby("Station")[c].transform(lambda s: flatline_mask(s))
    qc_flat[c] = int(m.sum()); df.loc[m, c] = np.nan
report["qc_flatline_removed"] = qc_flat

# ---------- 8-h rolling O3 then daily aggregation ----------
df["O3_8h"] = df.groupby("Station")["O3_ppb"].transform(lambda s: s.rolling(8, min_periods=6).mean())
df["date"] = df["dt"].dt.normalize()

# valid-hour counts for completeness gating
cnt = df.groupby(["Station","date"]).agg(pm_n=("PM2.5_ugm3","count"),
                                         o3_n=("O3_ppb","count")).reset_index()

aggmap = {**{c:"mean" for c in MET if c != "precip_mm"}, "precip_mm":"sum",
          "PM2.5_ugm3":"mean","O3_8h":"max","PM10_ugm3":"mean","NO2_ppb":"mean",
          "NOx_ppb":"mean","CO_ppm":"mean","SO2_ppb":"mean",
          **{c:"mean" for c in SAT}}
g = df.groupby(["Station","date"]).agg(aggmap).reset_index().merge(cnt, on=["Station","date"])

meta = df.groupby("Station")[["latitude","longtitude"]].first()
g = g.merge(meta, on="Station")

# completeness gate: need enough valid hours for a trustworthy daily value
g.loc[g["pm_n"] < 18, "PM2.5_ugm3"] = np.nan
g.loc[g["o3_n"] < 12, "O3_8h"] = np.nan

# reindex each station to a full daily calendar so lags are true calendar lags
full = pd.date_range("2014-01-01", "2016-12-31", freq="D")
g = (g.set_index("date").groupby("Station")
       .apply(lambda d: d.reindex(full))
       .drop(columns="Station").rename_axis(["Station","date"]).reset_index())
g["latitude"] = g.groupby("Station")["latitude"].ffill().bfill()
g["longtitude"] = g.groupby("Station")["longtitude"].ffill().bfill()

# season + year
m = g["date"].dt.month
g["season"] = np.select([m.isin([12,1,2]), m.isin([3,4,5]), m.isin([6,7,8,9])],
                        ["winter","pre","mon"], default="post")
g["year"] = g["date"].dt.year
report["station_days"] = int(g["date"].notna().sum())

# ---------- weather regimes (fit on TRAIN years only) ----------
REG_FEATS = ["temp_C","RH_pct","wind_speed_ms","solar_rad_Wm2","precip_mm",
             "boundary_layer_height_m","ventilation_coefficient",
             "photochemical_activitiy_index","surface_pressure_hPa"]
tr = g[g["year"].isin(TRAIN_YEARS)].dropna(subset=REG_FEATS)
scaler = StandardScaler().fit(tr[REG_FEATS])
km = KMeans(K_REGIMES, n_init=10, random_state=0).fit(scaler.transform(tr[REG_FEATS]))
ok = g[REG_FEATS].notna().all(axis=1)
g["regime"] = np.nan
g.loc[ok, "regime"] = km.predict(scaler.transform(g.loc[ok, REG_FEATS]))

# name regimes by centroid character (back in original units)
cent = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=REG_FEATS)
names = {}
for i, r in cent.iterrows():
    if r["precip_mm"] > cent["precip_mm"].median() and r["RH_pct"] > 85:
        names[i] = "Monsoon-Wet-Windy"
    elif r["ventilation_coefficient"] == cent["ventilation_coefficient"].max():
        names[i] = "Ventilated-Stormy"
    elif r["photochemical_activitiy_index"] == cent["photochemical_activitiy_index"].max() and r["RH_pct"] < 75:
        names[i] = "Dry-Sunny-Photochemical"
    elif r["ventilation_coefficient"] == cent["ventilation_coefficient"].min():
        names[i] = "Stagnant-Trapping"
    else:
        names[i] = "Humid-Transition"
# guarantee uniqueness
seen={};
for k_ in list(names):
    n=names[k_];
    if n in seen: names[k_]=f"{n}-{k_}"
    seen[names[k_]]=1
g["regime_name"] = g["regime"].map(names)
report["regime_names"] = {int(k): v for k, v in names.items()}

# ---------- compound labels (seasonal x station 75th, TRAIN-only thresholds) ----------
def fit_thr(frame):
    thr = {}
    for (st, se), sub in frame.groupby(["Station","season"]):
        thr[(st,se,"PM")] = sub["PM2.5_ugm3"].quantile(Q)
        thr[(st,se,"O3")] = sub["O3_8h"].quantile(Q)
    return thr
thr = fit_thr(g[g["year"].isin(TRAIN_YEARS)])
joblib.dump({"scaler":scaler,"kmeans":km,"names":names,"thr":thr,"Q":Q}, ART/"artefacts.joblib")

g["thr_PM"] = [thr.get((st,se,"PM"), np.nan) for st,se in zip(g["Station"],g["season"])]
g["thr_O3"] = [thr.get((st,se,"O3"), np.nan) for st,se in zip(g["Station"],g["season"])]
g["pm_ex"] = (g["PM2.5_ugm3"] >= g["thr_PM"]).astype("float")
g["o3_ex"] = (g["O3_8h"] >= g["thr_O3"]).astype("float")
g.loc[g["PM2.5_ugm3"].isna(), "pm_ex"] = np.nan
g.loc[g["O3_8h"].isna(), "o3_ex"] = np.nan
g["compound"] = ((g["pm_ex"]==1) & (g["o3_ex"]==1)).astype("float")
g.loc[g["pm_ex"].isna() | g["o3_ex"].isna(), "compound"] = np.nan

# ---------- feature engineering (predict day t+1/t+2/t+3) ----------
g = g.sort_values(["Station","date"]).reset_index(drop=True)
gb = g.groupby("Station")
# circular calendar
doy = g["date"].dt.dayofyear
g["doy_sin"] = np.sin(2*np.pi*doy/365.25); g["doy_cos"] = np.cos(2*np.pi*doy/365.25)
g["weekday"] = g["date"].dt.weekday

LAGW = ["PM2.5_ugm3","O3_8h","temp_C","RH_pct","wind_speed_ms","solar_rad_Wm2",
        "boundary_layer_height_m","ventilation_coefficient","photochemical_activitiy_index","precip_mm"]
for c in LAGW:
    for L in (1,2,3):
        g[f"{c}_lag{L}"] = gb[c].shift(L)
for c in ["PM2.5_ugm3","O3_8h"]:
    g[f"{c}_roll3"] = gb[c].shift(1).rolling(3).mean().reset_index(0,drop=True)
    g[f"{c}_roll7"] = gb[c].shift(1).rolling(7).mean().reset_index(0,drop=True)
g["precip_roll3"] = gb["precip_mm"].shift(1).rolling(3).sum().reset_index(0,drop=True)

# persistence states (today)
g["pm_ex_today"]=g["pm_ex"]; g["o3_ex_today"]=g["o3_ex"]; g["compound_today"]=g["compound"]

# targets: leads
for h,lab in [(1,"y24"),(2,"y48"),(3,"y72")]:
    g[lab] = gb["compound"].shift(-h)
g["pm_next"] = gb["PM2.5_ugm3"].shift(-1)
g["o3_next"] = gb["O3_8h"].shift(-1)
g["pm_ex_next"] = gb["pm_ex"].shift(-1)
g["o3_ex_next"] = gb["o3_ex"].shift(-1)

g.to_csv(ROOT/"modelling_daily.csv", index=False)

# ---------- EDA: regime lift table (full data, for the report) ----------
val = g.dropna(subset=["compound","regime_name"])
rows=[]
for rn, sub in val.groupby("regime_name"):
    rate=sub["compound"].mean(); exp=sub["pm_ex"].mean()*sub["o3_ex"].mean()
    rows.append(dict(regime=rn, n_days=int(len(sub)),
                     RH=round(sub["RH_pct"].mean(),0), solar=round(sub["solar_rad_Wm2"].mean(),0),
                     photochem=round(sub["photochemical_activitiy_index"].mean(),0),
                     ventilation=round(sub["ventilation_coefficient"].mean(),0),
                     compound_pct=round(100*rate,1), lift=round(rate/exp,2) if exp>0 else None))
lift_tbl = sorted(rows, key=lambda r: -(r["lift"] or 0))
report["regime_lift_table"] = lift_tbl
report["compound_rate_overall_pct"] = round(100*val["compound"].mean(),2)
report["n_compound_days"] = int(val["compound"].sum())
report["compound_by_season_pct"] = {k: round(100*v,1) for k,v in val.groupby("season")["compound"].mean().items()}
report["n_model_rows_y24"] = int(g["y24"].notna().sum())
report["pos_rate_y24_pct"] = round(100*g["y24"].mean(),2)

json.dump(report, open(ROOT/"qc_report.json","w"), indent=2, default=str)
print(json.dumps(report, indent=2, default=str))
print("\nSaved modelling_daily.csv with", g.shape[1], "columns,", len(g), "rows")
