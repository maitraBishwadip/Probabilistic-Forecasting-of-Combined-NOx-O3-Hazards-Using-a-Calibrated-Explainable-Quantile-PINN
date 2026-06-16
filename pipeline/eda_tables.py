# -*- coding: utf-8 -*-
"""Descriptive tables: annual and regime-wise mean PM2.5 & O3 per station."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT=Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")

# ---- raw hourly (for true means) ----
df=pd.read_csv(ROOT/"BD_DOE_2014-16.csv",low_memory=False)
df=df.loc[:,~df.columns.str.contains("^Unnamed")]
for c in ["PM2.5_ugm3","O3_ppb"]: df[c]=pd.to_numeric(df[c],errors="coerce")
df["dt"]=pd.to_datetime(df["Date"]+" "+df["Time"],errors="coerce")
df=df.dropna(subset=["dt"])
# physical-range QC
df.loc[(df["PM2.5_ugm3"]<0)|(df["PM2.5_ugm3"]>1500),"PM2.5_ugm3"]=np.nan
df.loc[(df["O3_ppb"]<0)|(df["O3_ppb"]>300),"O3_ppb"]=np.nan
df["year"]=df["dt"].dt.year; df["date"]=df["dt"].dt.normalize()

# ---- TABLE 1: annual mean per station per year ----
a=(df.groupby(["Station","year"])
     .agg(PM2_5_ugm3_mean=("PM2.5_ugm3","mean"),
          O3_ppb_mean=("O3_ppb","mean"),
          n_hours_PM=("PM2.5_ugm3","count"),
          n_hours_O3=("O3_ppb","count")).reset_index())
allyr=(df.groupby("year").agg(PM2_5_ugm3_mean=("PM2.5_ugm3","mean"),O3_ppb_mean=("O3_ppb","mean"),
       n_hours_PM=("PM2.5_ugm3","count"),n_hours_O3=("O3_ppb","count")).reset_index())
allyr.insert(0,"Station","ALL_STATIONS")
a=pd.concat([a,allyr],ignore_index=True)
for c in ["PM2_5_ugm3_mean","O3_ppb_mean"]: a[c]=a[c].round(1)
a.to_csv(ROOT/"annual_mean_pm25_o3_by_station_year.csv",index=False)

# ---- TABLE 2: regime-wise mean per station (merge daily regime label onto hourly) ----
md=pd.read_csv(ROOT/"modelling_daily.csv",parse_dates=["date"])[["Station","date","regime_name"]]
md["date"]=md["date"].dt.normalize()
h=df.merge(md,on=["Station","date"],how="inner").dropna(subset=["regime_name"])
r=(h.groupby(["Station","regime_name"])
     .agg(PM2_5_ugm3_mean=("PM2.5_ugm3","mean"),
          O3_ppb_mean=("O3_ppb","mean"),
          n_days=("date","nunique"),
          n_hours=("PM2.5_ugm3","count")).reset_index())
allre=(h.groupby("regime_name").agg(PM2_5_ugm3_mean=("PM2.5_ugm3","mean"),O3_ppb_mean=("O3_ppb","mean"),
       n_days=("date","nunique"),n_hours=("PM2.5_ugm3","count")).reset_index())
allre.insert(0,"Station","ALL_STATIONS")
r=pd.concat([r,allre],ignore_index=True)
for c in ["PM2_5_ugm3_mean","O3_ppb_mean"]: r[c]=r[c].round(1)
r.to_csv(ROOT/"regime_mean_pm25_o3_by_station.csv",index=False)

print("=== annual_mean_pm25_o3_by_station_year.csv ===")
print(a.to_string(index=False))
print("\n=== regime_mean_pm25_o3_by_station.csv (head) ===")
print(r.head(20).to_string(index=False))
print("\nrows: annual",len(a),"| regime",len(r))
