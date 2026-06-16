# -*- coding: utf-8 -*-
"""
gis_station_summary.py  (descriptive EDA / profiling — NOT a model experiment)

Builds a GIS-ready summary of the 9 DoE stations from BD_DOE_2014-16.csv for map
plotting: per-station lat/lon, mean/median/tail O3 and NOx, the physics-relevant
oxidant Ox=O3+NO2, an O3/NOx photochemical indicator, data completeness, a
cross-station combined-hazard score, and a transparent (threshold-based, no model
fit) "pollutant regime" label for marker colouring.

Outputs (ROOT):
  - gis_station_pollutant_regimes.geojson   (FeatureCollection, Point geometry)
  - gis_station_pollutant_regimes.json      (flat records array, same content)

Physical-range QC is applied (consistent with pipeline/01_build_dataset.py) so the
means are not contaminated by out-of-range spikes. This is descriptive only — it
fits no model and produces no paper headline metric.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
RAW = ROOT / "BD_DOE_2014-16.csv"

# physical plausible ranges (same as pipeline/01) -> clip out-of-range to NaN
RANGES = {"NO_ppb": (0, 1000), "NO2_ppb": (0, 1000), "NOx_ppb": (0, 2000),
          "O3_ppb": (0, 300)}

POLL = ["NO_ppb", "NO2_ppb", "NOx_ppb", "O3_ppb"]

# ---------- load + QC ----------
df = pd.read_csv(RAW, low_memory=False)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
for c in POLL:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c, (lo, hi) in RANGES.items():
    bad = (df[c] < lo) | (df[c] > hi)
    df.loc[bad, c] = np.nan

# physics-relevant combined oxidant (slowly-varying, conserved through NO<->NO2<->O3 titration)
df["Ox_ppb"] = df["O3_ppb"] + df["NO2_ppb"]

n_total_per_station = df.groupby("Station").size()


def stat_block(s):
    s = s.dropna()
    if len(s) == 0:
        return dict(mean=None, median=None, p95=None, max=None, n_valid=0)
    return dict(mean=round(float(s.mean()), 2),
                median=round(float(s.median()), 2),
                p95=round(float(s.quantile(0.95)), 2),
                max=round(float(s.max()), 2),
                n_valid=int(len(s)))


records = []
for st, sub in df.groupby("Station"):
    lat = float(sub["latitude"].dropna().iloc[0])
    lon = float(sub["longtitude"].dropna().iloc[0])
    o3 = stat_block(sub["O3_ppb"])
    nox = stat_block(sub["NOx_ppb"])
    no2 = stat_block(sub["NO2_ppb"])
    no = stat_block(sub["NO_ppb"])
    ox = stat_block(sub["Ox_ppb"])
    n_total = int(n_total_per_station[st])
    # O3/NOx ratio of the means: low => fresh/titrated (traffic), high => aged/photochemical
    ratio = round(o3["mean"] / nox["mean"], 3) if (o3["mean"] and nox["mean"]) else None
    records.append(dict(
        station=st, latitude=round(lat, 6), longitude=round(lon, 6),
        n_hours_total=n_total,
        o3_completeness_pct=round(100 * o3["n_valid"] / n_total, 1),
        nox_completeness_pct=round(100 * nox["n_valid"] / n_total, 1),
        mean_O3_ppb=o3["mean"], median_O3_ppb=o3["median"], p95_O3_ppb=o3["p95"], max_O3_ppb=o3["max"],
        mean_NOx_ppb=nox["mean"], median_NOx_ppb=nox["median"], p95_NOx_ppb=nox["p95"], max_NOx_ppb=nox["max"],
        mean_NO2_ppb=no2["mean"], mean_NO_ppb=no["mean"], mean_Ox_ppb=ox["mean"],
        O3_to_NOx_ratio=ratio,
        n_O3_valid=o3["n_valid"], n_NOx_valid=nox["n_valid"],
    ))

rec = pd.DataFrame(records)

# ---------- cross-station combined-hazard score + transparent regime label ----------
# percentile rank (0-1) of each station's mean within the 9 stations, averaged across the
# two pollutants -> spirit of the H index but at the station level (relative, for colouring).
rec["o3_rank"] = rec["mean_O3_ppb"].rank(pct=True)
rec["nox_rank"] = rec["mean_NOx_ppb"].rank(pct=True)
rec["combined_hazard_score"] = ((rec["o3_rank"] + rec["nox_rank"]) / 2).round(3)

o3_med = rec["mean_O3_ppb"].median()
nox_med = rec["mean_NOx_ppb"].median()


def regime(r):
    hi_o3 = r["mean_O3_ppb"] >= o3_med
    hi_nox = r["mean_NOx_ppb"] >= nox_med
    if hi_nox and hi_o3:
        return "High-burden (NOx & O3 both high)"
    if hi_nox and not hi_o3:
        return "NOx-dominated (traffic / titration)"
    if hi_o3 and not hi_nox:
        return "O3-dominated (photochemical)"
    return "Background / low"


rec["pollutant_regime"] = rec.apply(regime, axis=1)
rec = rec.drop(columns=["o3_rank", "nox_rank"])
rec = rec.sort_values("combined_hazard_score", ascending=False).reset_index(drop=True)

# ---------- write flat records JSON ----------
metadata = dict(
    source="BD_DOE_2014-16.csv (9 DoE stations, hourly, 2014-01-01..2016-12-31)",
    units="O3/NOx/NO2/NO/Ox in ppb",
    definitions=dict(
        Ox_ppb="O3 + NO2 (combined oxidant; slowly-varying, conserved through NO<->NO2<->O3 titration)",
        O3_to_NOx_ratio="ratio of station-mean O3 to station-mean NOx; low=fresh/titrated (traffic), high=aged/photochemical",
        combined_hazard_score="mean of cross-station percentile ranks of mean O3 and mean NOx (0-1); higher = jointly more hazardous",
        pollutant_regime="transparent 2x2 label vs cross-station medians of mean O3 and mean NOx (descriptive, no model fit)",
    ),
    qc="physical-range clip applied (O3 0-300, NOx 0-2000, NO/NO2 0-1000 ppb)",
    note="Descriptive EDA for GIS plotting only; not a QR-PINN experiment and not a paper headline metric.",
    n_stations=int(len(rec)),
)
flat = dict(metadata=metadata, stations=rec.to_dict(orient="records"))
(ROOT / "gis_station_pollutant_regimes.json").write_text(json.dumps(flat, indent=2), encoding="utf-8")

# ---------- write GeoJSON (GIS-native) ----------
features = []
for r in rec.to_dict(orient="records"):
    lon, lat = r["longitude"], r["latitude"]
    features.append(dict(type="Feature",
                         geometry=dict(type="Point", coordinates=[lon, lat]),
                         properties=r))
geojson = dict(type="FeatureCollection", metadata=metadata, features=features)
(ROOT / "gis_station_pollutant_regimes.geojson").write_text(json.dumps(geojson, indent=2), encoding="utf-8")

# ---------- console summary ----------
cols = ["station", "latitude", "longitude", "mean_O3_ppb", "mean_NOx_ppb",
        "O3_to_NOx_ratio", "combined_hazard_score", "pollutant_regime"]
print(rec[cols].to_string(index=False))
print("\nWrote: gis_station_pollutant_regimes.geojson  and  gis_station_pollutant_regimes.json")
