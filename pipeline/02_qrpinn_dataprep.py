# -*- coding: utf-8 -*-
"""
02_qrpinn_dataprep.py  (E0)
Hourly QC + cleaning + feature engineering for the QR-PINN combined NOx-O3 study.

ANTI-SYNTHETIC-DATA POLICY (see CLAUDE.md):
  * No pollutant value is ever interpolated/imputed. Out-of-range and flat-line
    values become NaN and stay NaN.
  * The target H is computed ONLY for hours where BOTH NOx and O3 are observed.
  * Missing MODEL INPUTS are represented by a mask channel (1=observed, 0=missing)
    plus a neutral 0 placeholder AFTER scaling. This is masking, not data creation;
    the network is told which inputs are real. Targets/statistics use observed data only.
  * All scalers and the extreme threshold are fit on TRAIN years (2014-15) only.

Output: artefacts/qrpinn_data.npz  (+ artefacts/qrpinn_meta.json)
"""
import json, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
ART = ROOT / "artefacts"; ART.mkdir(exist_ok=True)
RAW = ROOT / "BD_DOE_2014-16.csv"
TRAIN_YEARS = [2014, 2015]
Q_EXTREME = 0.30          # cutoff at 30th pct of H  => >=70% of obs are the hazardous class
meta = {"policy": "no imputation of pollutants or target; mask+zero for missing inputs only"}

# ---------------- load ----------------
df = pd.read_csv(RAW, low_memory=False)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
df = df.dropna(subset=["dt"]).sort_values(["Station", "dt"]).reset_index(drop=True)

POLL = ["NO_ppb", "NO2_ppb", "NOx_ppb", "O3_ppb"]
MET = ["temp_C", "RH_pct", "wind_speed_ms", "wind_speed_u_ms", "wind_speed_v_ms",
       "solar_rad_Wm2", "precip_mm", "surface_pressure_hPa", "boundary_layer_height_m",
       "ventilation_coefficient", "photochemical_activitiy_index"]
SAT = ["modis_aod_550nm"]
for c in POLL + MET + SAT:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------------- QC: physical range (out-of-range -> NaN, never replaced) ----------------
RANGES = {"NO_ppb": (0, 1000), "NO2_ppb": (0, 1000), "NOx_ppb": (0, 2000), "O3_ppb": (0, 300)}
qc_range = {}
for c, (lo, hi) in RANGES.items():
    bad = (df[c] < lo) | (df[c] > hi)
    qc_range[c] = int(bad.sum()); df.loc[bad, c] = np.nan
meta["qc_range_to_nan"] = qc_range

# ---------------- QC: flat-line (>=24 identical consecutive, value>0) -> NaN ----------------
def flatline_mask(s, run=24):
    g = (s != s.shift()).cumsum()
    sz = s.groupby(g).transform("size")
    return (sz >= run) & (s > 0) & s.notna()
qc_flat = {}
for c in POLL:
    m = df.groupby("Station")[c].transform(lambda s: flatline_mask(s))
    qc_flat[c] = int(m.sum()); df.loc[m, c] = np.nan
meta["qc_flatline_to_nan"] = qc_flat

# ---------------- reindex each station to a true continuous hourly calendar ----------------
df["date"] = df["dt"]
full = pd.date_range("2014-01-01 00:00", "2016-12-31 23:00", freq="h")
stations = sorted(df["Station"].unique().tolist())
st2i = {s: i for i, s in enumerate(stations)}

per = {}
for s in stations:
    d = df[df["Station"] == s].set_index("dt").reindex(full)
    per[s] = d
meta["stations"] = stations
meta["hours_per_station"] = len(full)

# ---------------- fit TRAIN-only scalers ----------------
train_mask_full = full.year.isin(TRAIN_YEARS)
def stack_train(col):
    vals = []
    for s in stations:
        v = per[s][col].values[train_mask_full]
        vals.append(v)
    a = np.concatenate(vals)
    return a[~np.isnan(a)]

robust = {}   # for pollutants: median + IQR
for c in POLL + SAT:
    a = stack_train(c)
    med = float(np.median(a)); iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
    robust[c] = {"median": med, "iqr": iqr if iqr > 1e-6 else 1.0}
standard = {}  # for met: mean + std
for c in MET:
    a = stack_train(c)
    mu = float(np.mean(a)); sd = float(np.std(a))
    standard[c] = {"mean": mu, "std": sd if sd > 1e-6 else 1.0}
meta["robust_scaler"] = robust; meta["standard_scaler"] = standard

def rs(col, x):   # robust scale
    return (x - robust[col]["median"]) / robust[col]["iqr"]
def ss(col, x):   # standard scale
    return (x - standard[col]["mean"]) / standard[col]["std"]

# ---------------- combined hazard index H (only where BOTH NOx & O3 observed) ----------------
# H = 0.5 * robustscale(NOx) + 0.5 * robustscale(O3)
H_all = {}
for s in stations:
    nox = per[s]["NOx_ppb"].values; o3 = per[s]["O3_ppb"].values
    both = ~np.isnan(nox) & ~np.isnan(o3)
    h = np.full(len(full), np.nan)
    h[both] = 0.5 * rs("NOx_ppb", nox[both]) + 0.5 * rs("O3_ppb", o3[both])
    H_all[s] = h

# threshold from TRAIN H only
Htr = np.concatenate([H_all[s][train_mask_full] for s in stations])
Htr = Htr[~np.isnan(Htr)]
thr = float(np.quantile(Htr, Q_EXTREME))
meta["H_threshold_Q30_train"] = thr
meta["train_frac_extreme_by_construction"] = float((Htr >= thr).mean())

# ---------------- build per-station feature matrices ----------------
# feature order (documented in meta)
poll_feat = POLL                      # scaled + mask each
met_feat = MET                        # scaled (100% present, but keep mask for safety)
sat_feat = SAT                        # scaled + mask
cal_feat = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "weekday"]

feat_names = []
for c in poll_feat: feat_names += [c, c + "_mask"]
for c in met_feat:  feat_names += [c]
for c in sat_feat:  feat_names += [c, c + "_mask"]
feat_names += cal_feat
F = len(feat_names)

hour = full.hour.values.astype(float)
doy = full.dayofyear.values.astype(float)
wd = full.weekday.values.astype(float)
cal = {
    "hour_sin": np.sin(2*np.pi*hour/24), "hour_cos": np.cos(2*np.pi*hour/24),
    "doy_sin": np.sin(2*np.pi*doy/365.25), "doy_cos": np.cos(2*np.pi*doy/365.25),
    "weekday": wd/6.0,
}

T = len(full)
X = np.zeros((len(stations), T, F), dtype=np.float32)
# physics drivers (scaled met) we will reuse in the model: VC, BLH, photochem, precip, temp
phys_cols = ["ventilation_coefficient", "boundary_layer_height_m",
             "photochemical_activitiy_index", "precip_mm", "temp_C"]
PHYS = np.zeros((len(stations), T, len(phys_cols)), dtype=np.float32)
# observed scaled concentrations + masks for NOx, O3 (aux-head data fit)
COBS = np.zeros((len(stations), T, 2), dtype=np.float32)      # [NOx, O3] scaled
CMASK = np.zeros((len(stations), T, 2), dtype=np.float32)
Hmat = np.full((len(stations), T), np.nan, dtype=np.float32)
year = full.year.values.astype(np.int16)

for s in stations:
    si = st2i[s]; d = per[s]
    j = 0
    for c in poll_feat:
        v = d[c].values.astype(float)
        m = (~np.isnan(v)).astype(np.float32)
        vs = rs(c, v); vs[np.isnan(vs)] = 0.0          # mask+zero placeholder (NOT data)
        X[si, :, j] = vs; X[si, :, j+1] = m; j += 2
    for c in met_feat:
        v = d[c].values.astype(float)
        vs = ss(c, v); vs[np.isnan(vs)] = 0.0           # met is ~100% present
        X[si, :, j] = vs; j += 1
    for c in sat_feat:
        v = d[c].values.astype(float)
        m = (~np.isnan(v)).astype(np.float32)
        vs = rs(c, v); vs[np.isnan(vs)] = 0.0
        X[si, :, j] = vs; X[si, :, j+1] = m; j += 2
    for c in cal_feat:
        X[si, :, j] = cal[c].astype(np.float32); j += 1
    assert j == F
    for k, c in enumerate(phys_cols):
        v = d[c].values.astype(float); vs = ss(c, v); vs[np.isnan(vs)] = 0.0
        PHYS[si, :, k] = vs
    for k, c in enumerate(["NOx_ppb", "O3_ppb"]):
        v = d[c].values.astype(float); m = (~np.isnan(v)).astype(np.float32)
        vs = rs(c, v); vs[np.isnan(vs)] = 0.0
        COBS[si, :, k] = vs; CMASK[si, :, k] = m
    Hmat[si, :] = H_all[s].astype(np.float32)

# ---------------- save ----------------
np.savez_compressed(ART / "qrpinn_data.npz",
    X=X, PHYS=PHYS, COBS=COBS, CMASK=CMASK, H=Hmat, year=year,
    feat_names=np.array(feat_names), phys_cols=np.array(phys_cols),
    stations=np.array(stations), thr=np.array([thr]))
meta["F"] = F; meta["feat_names"] = feat_names; meta["phys_cols"] = phys_cols
meta["T"] = int(T)
# coverage summary (observed only)
for nm, msk in [("H_overall", ~np.isnan(Hmat)),
                ("H_train", (~np.isnan(Hmat)) & (year[None, :] != 2016)),
                ("H_test2016", (~np.isnan(Hmat)) & (year[None, :] == 2016))]:
    meta[f"n_{nm}"] = int(msk.sum())
json.dump(meta, open(ART / "qrpinn_meta.json", "w"), indent=2, default=str)

print("Saved", ART / "qrpinn_data.npz")
print("F =", F, "| stations =", len(stations), "| hours/station =", T)
print("thr (Q30 train) =", round(thr, 4),
      "| train frac extreme =", round(meta["train_frac_extreme_by_construction"], 3))
print("valid H: overall", meta["n_H_overall"], "| train", meta["n_H_train"], "| test2016", meta["n_H_test2016"])
print("feature names:", feat_names)
