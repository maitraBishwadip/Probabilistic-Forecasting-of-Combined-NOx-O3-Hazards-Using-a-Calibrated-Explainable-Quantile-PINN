# -*- coding: utf-8 -*-
"""
10_qrpinn_e9_xai.py  (E9)  -- Explainability of the frozen calibrated QR-PINN.
Method & citations: METHODS_explainability.md ; design: EXPERIMENT_DESIGN.md sec.9.

POST-HOC on the already-trained, already-approved reference checkpoint
artefacts/qrpinn_full_rank_ext.pt (rank index H, physics on, extended 13-tau grid; E6).
NO retraining, no change to any locked decision.

Two complementary, cross-validating attributions:
  (1) Integrated Gradients (Sundararajan, Taly & Yan 2017) -- axiomatic, completeness-checked,
      hand-rolled (no captum). Baseline = "missing/no-information" input (values=0, masks=0).
      Attributed scalars expose the DISTRIBUTION, not a point, at several leads:
        med_h1/h6/h12/h24 = Q0.5(H_{t+h})  (central hazard; driver shift with lead)
        tail_h1           = Q0.95(H_{t+1}) (severe / hazardous upper tail)
        width_h1          = Q0.95-Q0.05    (predictive uncertainty / spread)
      Aggregated per-feature, per-lag, per physics-group, group x lag, and day vs night.
  (2) Permutation / occlusion importance (Breiman 2001; Fisher, Rudin & Dominici 2019) --
      model-agnostic, in the model's own loss currency: mean-occlude each feature/group across the
      window on TEST-2016 and measure dPinball / dCRPS / dBrier(exceedance); plus station permutation.

Outputs: results_e9.json, publication-grade figs e9_*.{png,pdf}, artefacts/e9_ig_sample.npz,
append RESULTS_LOG.md.  No synthetic data; observed H only. seed=0.
"""
import json, time, platform, numpy as np, torch, torch.nn as nn
from pathlib import Path
from datetime import datetime

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
ART = ROOT / "artefacts"; FIGS = ROOT / "figs"; FIGS.mkdir(exist_ok=True)
torch.manual_seed(0); np.random.seed(0); rng = np.random.default_rng(0)

L, HZ = 48, 24
TAUS = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99], dtype=np.float32)
nT = len(TAUS)
def ti(t): return int(np.argmin(np.abs(TAUS - t)))
I_MED, I_Q05, I_Q95 = ti(0.5), ti(0.05), ti(0.95)
CKPT = ART / "qrpinn_full_rank_ext.pt"
M_IG = 32            # integrated-gradients path steps (midpoint Riemann sum)
N_IG = 1280         # stratified sample size for IG (day/night balanced)
B_ANCH = 64         # anchors per IG chunk
BATCH_EVAL = 2048   # batch for occlusion forward passes

# ---------------- load E0 tensors + meta ----------------
d = np.load(ART / "qrpinn_data.npz", allow_pickle=True)
X = np.nan_to_num(d["X"].astype(np.float32))          # [S,T,F]  (missing already mask+zero in E0)
COBS = d["COBS"].astype(np.float32); CMASK = d["CMASK"].astype(np.float32)
Hrob = d["H"].astype(np.float32); year = d["year"].astype(np.int32)
S, T, F = X.shape
meta = json.load(open(ART / "qrpinn_meta.json"))
FEAT = meta["feat_names"]; assert len(FEAT) == F, (len(FEAT), F)
name2i = {n: k for k, n in enumerate(FEAT)}
import pandas as pd
hour = pd.date_range("2014-01-01 00:00", periods=T, freq="h").hour.values

# ---------------- rebuild rank index H (identical to E5/E6) ----------------
both = (CMASK[:, :, 0] == 1) & (CMASK[:, :, 1] == 1) & (~np.isnan(Hrob))
yr2d = np.broadcast_to(year[None, :], (S, T)); tr_both = both & (yr2d <= 2015)
sN = np.sort(COBS[:, :, 0][tr_both]); sO = np.sort(COBS[:, :, 1][tr_both])
def Fcdf(s, v): return np.searchsorted(s, v, side="right") / len(s)
H = np.full((S, T), np.nan, np.float32)
H[both] = (0.5 * Fcdf(sN, COBS[:, :, 0][both]) + 0.5 * Fcdf(sO, COBS[:, :, 1][both])).astype(np.float32)
thr = float(np.quantile(H[tr_both], 0.30))
print(f"rank thr(Q30 train)={thr:.4f}", flush=True)

def build_index():
    tr, te = [], []
    for si in range(S):
        Hs = H[si]
        for t in range(L - 1, T - HZ):
            if np.isnan(Hs[t]): continue
            if not np.any(~np.isnan(Hs[t + 1:t + 1 + HZ])): continue
            if year[t + HZ] <= 2015: tr.append((si, t))
            elif year[t] == 2016: te.append((si, t))
    return np.array(tr, np.int32), np.array(te, np.int32)
idx_tr, idx_te = build_index()
print(f"anchors train={len(idx_tr)} test2016={len(idx_te)}", flush=True)

# ---------------- model (identical arch to E6 ext checkpoint) ----------------
class QRPINN(nn.Module):
    def __init__(self, F, n_station, hidden=64, emb=8):
        super().__init__()
        self.emb = nn.Embedding(n_station, emb); self.lstm = nn.LSTM(F, hidden, batch_first=True)
        self.base = nn.Linear(hidden + emb, HZ); self.inc = nn.Linear(hidden + emb, HZ * (nT - 1))
        self.chead = nn.Linear(hidden, 2); self.praw = nn.Parameter(torch.zeros(7))
    def forward(self, x, si):
        out, (hn, cn) = self.lstm(x); trunk = torch.cat([hn[-1], self.emb(si)], 1)
        base = self.base(trunk); inc = torch.nn.functional.softplus(self.inc(trunk)).view(-1, HZ, nT - 1)
        q = torch.cat([base.unsqueeze(-1), base.unsqueeze(-1) + torch.cumsum(inc, -1)], -1)
        return q, self.chead(out)

model = QRPINN(F, S)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()
for p in model.parameters(): p.requires_grad_(False)   # only input grads needed for IG
print(f"loaded frozen checkpoint {CKPT.name}", flush=True)

# ---------------- physics groups (METHODS_explainability sec.3) ----------------
GROUPS = {
    "NOx_side":    ["NO_ppb", "NO_ppb_mask", "NO2_ppb", "NO2_ppb_mask", "NOx_ppb", "NOx_ppb_mask"],
    "O3_side":     ["O3_ppb", "O3_ppb_mask"],
    "photochem":   ["solar_rad_Wm2", "photochemical_activitiy_index", "temp_C"],
    "dispersion":  ["boundary_layer_height_m", "ventilation_coefficient", "wind_speed_ms",
                    "wind_speed_u_ms", "wind_speed_v_ms"],
    "wet_removal": ["precip_mm", "RH_pct"],
    "pressure":    ["surface_pressure_hPa"],
    "satellite":   ["modis_aod_550nm", "modis_aod_550nm_mask"],
    "calendar":    ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "weekday"],
}
GLIST = list(GROUPS.keys())
GROUP_IDX = {g: [name2i[n] for n in cols] for g, cols in GROUPS.items()}

# ================= Integrated Gradients =================
HZ_OF = {"med_h1": 0, "med_h6": 5, "med_h12": 11, "med_h24": 23}
TARGETS = ["med_h1", "med_h6", "med_h12", "med_h24", "tail_h1", "width_h1"]
def tgt_value(q, name):
    if name.startswith("med_h"): return q[:, HZ_OF[name], I_MED]
    if name == "tail_h1":        return q[:, 0, I_Q95]
    if name == "width_h1":       return q[:, 0, I_Q95] - q[:, 0, I_Q05]
    raise KeyError(name)

def window(si, t): return np.nan_to_num(X[si, t - L + 1:t + 1, :]).astype(np.float32)

def ig_chunk(xt, sib):
    """xt torch [b,L,F], sib [b] np.int -> dict name->IG [b,L,F]; + (gx,g0) for med_h1 completeness."""
    b = xt.shape[0]
    alphas = ((np.arange(M_IG) + 0.5) / M_IG).astype(np.float32)         # midpoints
    path = (torch.from_numpy(alphas).view(1, M_IG, 1, 1) * xt.view(b, 1, L, F)).reshape(b * M_IG, L, F)
    si_rep = torch.from_numpy(np.repeat(sib, M_IG)).long()
    out = {}
    for name in TARGETS:
        inp = path.clone().requires_grad_(True)
        g = tgt_value(model(inp, si_rep)[0], name).sum()
        if inp.grad is not None: inp.grad = None
        g.backward()
        grad = inp.grad.detach().reshape(b, M_IG, L, F).mean(1)          # avg gradient along path
        out[name] = (xt * grad).numpy()                                  # (x-0)*avg_grad
    with torch.no_grad():
        sit = torch.from_numpy(sib).long()
        gx = tgt_value(model(xt, sit)[0], "med_h1").numpy()
        g0 = tgt_value(model(torch.zeros_like(xt), sit)[0], "med_h1").numpy()
    return out, gx, g0

# ---- stratified day/night sample of test anchors ----
anc_hour = hour[idx_te[:, 1]]
is_day_all = np.isin(anc_hour, np.arange(9, 17))           # solar-on hours (local)
day_pos = np.where(is_day_all)[0]; night_pos = np.where(~is_day_all)[0]
ne = N_IG // 2
sel = np.concatenate([rng.choice(day_pos, min(ne, len(day_pos)), replace=False),
                      rng.choice(night_pos, min(ne, len(night_pos)), replace=False)])
rng.shuffle(sel)
samp = idx_te[sel]; samp_isday = is_day_all[sel]
Xs = np.stack([window(si, t) for si, t in samp]).astype(np.float32)      # [N,L,F] precomputed
np.savez(ART / "e9_ig_sample.npz", sample_anchors=samp, is_day=samp_isday, sel_into_idx_te=sel)
print(f"IG sample: {len(samp)} anchors ({samp_isday.sum()} day / {(~samp_isday).sum()} night), M={M_IG}", flush=True)

feat_abs = {n: np.zeros(F) for n in TARGETS}
feat_sgn = {n: np.zeros(F) for n in TARGETS}
lag_abs_med = np.zeros(L)
lag_group_abs = np.zeros((len(GLIST), L))
day_abs = np.zeros(F); night_abs = np.zeros(F); n_day = n_night = 0
comp_gaps = []
t0 = time.time()
for s in range(0, len(samp), B_ANCH):
    xt = torch.from_numpy(Xs[s:s + B_ANCH]); isd = samp_isday[s:s + B_ANCH]
    sib = samp[s:s + B_ANCH, 0].astype(np.int64)
    igs, gx, g0 = ig_chunk(xt, sib)
    for name in TARGETS:
        ig = igs[name]
        feat_abs[name] += np.abs(ig).sum(1).sum(0); feat_sgn[name] += ig.sum(1).sum(0)
    absm = np.abs(igs["med_h1"])                                  # [b,L,F]
    lag_abs_med += absm.sum(2).sum(0)
    for gi, g in enumerate(GLIST):
        lag_group_abs[gi] += absm[:, :, GROUP_IDX[g]].sum(2).sum(0)
    fa = absm.sum(1)                                              # [b,F]
    day_abs += fa[isd].sum(0); night_abs += fa[~isd].sum(0)
    n_day += int(isd.sum()); n_night += int((~isd).sum())
    igsum = igs["med_h1"].reshape(xt.shape[0], -1).sum(1)
    comp_gaps.append(np.abs(igsum - (gx - g0)) / (np.abs(gx - g0) + 1e-6))
    if s % (B_ANCH * 8) == 0: print(f"  IG {s+xt.shape[0]}/{len(samp)} ({time.time()-t0:.0f}s)", flush=True)
nA = len(samp)
for name in TARGETS: feat_abs[name] /= nA; feat_sgn[name] /= nA
lag_abs_med /= nA; lag_group_abs /= nA
day_abs /= max(n_day, 1); night_abs /= max(n_night, 1)
comp_gap = float(np.concatenate(comp_gaps).mean())
print(f"IG done. mean completeness gap (med_h1) = {comp_gap:.4f}", flush=True)

def group_frac(vec):
    tot = vec.sum() + 1e-12
    return {g: float(vec[idx].sum() / tot) for g, idx in GROUP_IDX.items()}
ig_out = {}
for name in TARGETS:
    tot = feat_abs[name].sum() + 1e-12
    ig_out[name] = {"feat_abs": feat_abs[name].tolist(), "feat_signed": feat_sgn[name].tolist(),
                    "feat_abs_frac": (feat_abs[name] / tot).tolist(), "group_abs_frac": group_frac(feat_abs[name])}
ig_out["lag_abs_med_h1"] = lag_abs_med.tolist()
ig_out["lag_group_abs_med_h1"] = lag_group_abs.tolist()
ig_out["completeness_gap_rel_med_h1"] = comp_gap
ig_out["daynight_med_h1"] = {"day_group_frac": group_frac(day_abs), "night_group_frac": group_frac(night_abs),
                             "n_day": n_day, "n_night": n_night}

# ================= Occlusion / permutation importance =================
trmask = year <= 2015
mu = X[:, trmask, :].reshape(-1, F).mean(0).astype(np.float32)        # train-mean per channel
print("assembling test windows for occlusion ...", flush=True)
Xte = np.stack([window(si, t) for si, t in idx_te]).astype(np.float32)   # [Nte,L,F]
Site = idx_te[:, 0].astype(np.int64)
Y = np.zeros((len(idx_te), HZ), np.float32); Mk = np.zeros((len(idx_te), HZ), np.float32)
for i, (si, t) in enumerate(idx_te):
    fut = H[si, t + 1:t + 1 + HZ]; m = ~np.isnan(fut); Y[i, m] = fut[m]; Mk[i, m] = 1.0

def cdf_at_thr(Q, thrv):
    N, m = Q.shape; k = (Q < thrv).sum(1); lo = np.clip(k - 1, 0, m - 1); hi = np.clip(k, 0, m - 1); ar = np.arange(N)
    qlo, qhi, tlo, thi = Q[ar, lo], Q[ar, hi], TAUS[lo], TAUS[hi]
    gap = qhi - qlo; den = np.where(gap > 1e-9, gap, 1.0); frac = np.where(gap > 1e-9, (thrv - qlo) / den, 0.0)
    cdf = tlo + frac * (thi - tlo); cdf = np.where(k == 0, 0.0, cdf); cdf = np.where(k == m, 1.0, cdf)
    return np.clip(cdf, 0, 1)

def score_occ(occ_feats=None, occ_station=None):
    taus = TAUS.reshape(1, 1, nT); spin = nm = 0.0; ptau = np.zeros(nT); pe_sq = ne = 0.0
    for st in range(0, len(idx_te), BATCH_EVAL):
        xb = Xte[st:st + BATCH_EVAL].copy(); sib = Site[st:st + BATCH_EVAL].copy()
        if occ_feats is not None:
            for f in occ_feats: xb[:, :, f] = mu[f]
        if occ_station is not None: sib[:] = occ_station
        with torch.no_grad():
            q = model(torch.from_numpy(xb), torch.from_numpy(sib))[0].numpy()
        yb = Y[st:st + BATCH_EVAL]; mb = Mk[st:st + BATCH_EVAL]; m3 = mb[..., None]
        err = yb[..., None] - q; pb = np.maximum(taus * err, (taus - 1) * err)
        spin += (pb * m3).sum(); nm += m3.sum(); ptau += (pb * m3).sum((0, 1))
        Qf, Yf, Mf = q.reshape(-1, nT), yb.reshape(-1), mb.reshape(-1)
        pexc = 1.0 - cdf_at_thr(Qf, thr); ind = (Yf >= thr).astype(float); sl = Mf > 0
        pe_sq += ((pexc - ind) ** 2)[sl].sum(); ne += sl.sum()
    return float(spin / nm), float(2 * (ptau / nm).mean()), float(pe_sq / ne)

print("\n== occlusion: intact baseline ==", flush=True)
base_pin, base_crps, base_brier = score_occ()
print(f"  intact: pinball={base_pin:.4f} crps={base_crps:.4f} brier={base_brier:.4f}", flush=True)
occ_feat = {}
for f, fname in enumerate(FEAT):
    p, c, b = score_occ(occ_feats=[f])
    occ_feat[fname] = {"dpinball": p - base_pin, "dcrps": c - base_crps, "dbrier": b - base_brier}
occ_grp = {}
for g, idx in GROUP_IDX.items():
    p, c, b = score_occ(occ_feats=idx)
    occ_grp[g] = {"dpinball": p - base_pin, "dcrps": c - base_crps, "dbrier": b - base_brier}
    print(f"  occ GROUP {g:14s} dPin={p-base_pin:+.4f} dBrier={b-base_brier:+.4f}", flush=True)
ps, cs, bs = score_occ(occ_station=0)
occ_station = {"dpinball": ps - base_pin, "dcrps": cs - base_crps, "dbrier": bs - base_brier, "ref_station": 0}
print(f"  perm STATION (->id0)  dPin={ps-base_pin:+.4f} dBrier={bs-base_brier:+.4f}", flush=True)

# agreement IG vs occlusion across groups
igv = np.array([ig_out["med_h1"]["group_abs_frac"][g] for g in GLIST])
ocv = np.array([occ_grp[g]["dpinball"] for g in GLIST])
agree_r = float(np.corrcoef(igv, ocv)[0, 1])

# ================= assemble + save =================
R = {"subject_checkpoint": CKPT.name,
     "config": {"L": L, "HZ": HZ, "taus": TAUS.tolist(), "thr_Q30": thr, "M_IG": M_IG, "N_IG": int(nA),
                "ig_baseline": "missing/no-information (values=0, masks=0)", "n_test_anchors": int(len(idx_te)),
                "seed": 0, "tau_idx": {"median": I_MED, "q05": I_Q05, "q95": I_Q95}},
     "feat_names": FEAT, "groups": GROUPS,
     "intact_metrics_test2016": {"pinball": base_pin, "crps": base_crps, "brier_exceed": base_brier},
     "integrated_gradients": ig_out,
     "occlusion": {"per_feature": occ_feat, "per_group": occ_grp, "station": occ_station},
     "ig_vs_occlusion_group_r": agree_r,
     "env": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
json.dump(R, open(ROOT / "results_e9.json", "w"), indent=2)
print(f"wrote results_e9.json (IG-vs-occlusion group r={agree_r:.3f})", flush=True)

# ================= publication-grade figures =================
try:
    import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6})
    OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
          "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "gray": "#7F7F7F"}
    GC = {"NOx_side": OI["verm"], "O3_side": OI["blue"], "photochem": OI["orange"], "dispersion": OI["green"],
          "wet_removal": OI["sky"], "pressure": OI["purple"], "satellite": OI["yellow"], "calendar": OI["gray"]}
    def save(fig, name):
        for ext in ("png", "pdf"): fig.savefig(FIGS / f"{name}.{ext}")
        plt.close(fig)

    # Fig 1 — group importance (a) bars + (b) IG-vs-occlusion agreement scatter
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    order = np.argsort(igv)
    ax[0].barh([GLIST[i] for i in order], igv[order], color=[GC[GLIST[i]] for i in order], edgecolor="k", linewidth=0.4)
    ax[0].set_xlabel("Integrated-Gradients importance (fraction)"); ax[0].set_title("(a) Driver importance by physics group")
    ax[0].grid(axis="y", visible=False)
    ax[1].scatter(igv, ocv, c=[GC[g] for g in GLIST], s=90, edgecolor="k", linewidth=0.6, zorder=3)
    for g, xx, yy in zip(GLIST, igv, ocv):
        ax[1].annotate(g, (xx, yy), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    if np.ptp(igv) > 0:
        sl, ic = np.polyfit(igv, ocv, 1); xs = np.linspace(igv.min(), igv.max(), 20)
        ax[1].plot(xs, sl * xs + ic, "--", color=OI["gray"], lw=1, zorder=1)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("IG importance (median, h=1)"); ax[1].set_ylabel(r"Occlusion $\Delta$Pinball (test-2016)")
    ax[1].set_title(f"(b) Two independent methods agree (r = {agree_r:.2f})")
    fig.suptitle("Explainability of the combined NOx–O3 hazard QR-PINN", y=1.02, fontsize=12, fontweight="bold")
    save(fig, "e9_group_importance")

    # Fig 2 — distributional attribution: centre vs tail vs uncertainty
    x = np.arange(len(GLIST)); w = 0.26
    series = [("med_h1", "Median (centre)", OI["blue"]), ("tail_h1", r"Tail $Q_{0.95}$ (severe)", OI["verm"]),
              ("width_h1", "Interval width (uncertainty)", OI["green"])]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for k, (nm_, lab, col) in enumerate(series):
        ax.bar(x + (k - 1) * w, [ig_out[nm_]["group_abs_frac"][g] for g in GLIST], w, label=lab, color=col, edgecolor="k", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(GLIST, rotation=22, ha="right"); ax.set_ylabel("IG importance (fraction)")
    ax.grid(axis="x", visible=False); ax.legend(ncol=3, loc="upper right")
    ax.set_title("Distributional attribution: what drives the centre, the severe tail, and the uncertainty")
    save(fig, "e9_distributional")

    # Fig 3 — temporal saliency heatmap (group x lag)
    Mtx = lag_group_abs / (lag_group_abs.max() + 1e-12)
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    im = ax.imshow(Mtx, aspect="auto", cmap="magma", extent=[-L + 0.5, 0.5, len(GLIST) - 0.5, -0.5])
    ax.set_yticks(range(len(GLIST))); ax.set_yticklabels(GLIST)
    ax.axvline(-24, color="w", ls=":", lw=1); ax.text(-24, -0.65, "−24 h (diurnal)", color="k", fontsize=7.5, ha="center")
    ax.set_xlabel("lag from forecast time (h)"); ax.grid(False)
    ax.set_title("Temporal saliency over the 48 h window (median, h=1)")
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("IG importance (normalised)", fontsize=9)
    save(fig, "e9_temporal_heatmap")

    # Fig 4 — day vs night (dumbbell)
    dn = ig_out["daynight_med_h1"]; dv = np.array([dn["day_group_frac"][g] for g in GLIST])
    nv = np.array([dn["night_group_frac"][g] for g in GLIST]); yy = np.arange(len(GLIST))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i in range(len(GLIST)): ax.plot([dv[i], nv[i]], [i, i], color=OI["gray"], lw=1.4, zorder=1)
    ax.scatter(dv, yy, s=80, color=OI["yellow"], edgecolor="k", linewidth=0.6, label=f"day (n={dn['n_day']})", zorder=3)
    ax.scatter(nv, yy, s=80, color=OI["blue"], edgecolor="k", linewidth=0.6, label=f"night (n={dn['n_night']})", zorder=3)
    ax.set_yticks(yy); ax.set_yticklabels(GLIST); ax.grid(axis="y", visible=False)
    ax.set_xlabel("IG importance (fraction)"); ax.legend(loc="lower right")
    ax.set_title("Diurnal physics audit: day vs night attribution (median, h=1)")
    save(fig, "e9_daynight")

    # Fig 5 — lead-time driver shift
    hs = [1, 6, 12, 24]; keys = ["med_h1", "med_h6", "med_h12", "med_h24"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for g in GLIST:
        ax.plot(hs, [ig_out[k]["group_abs_frac"][g] for k in keys], "-o", ms=5, lw=1.8, color=GC[g], label=g)
    ax.set_xticks(hs); ax.set_xlabel("forecast lead time (h)"); ax.set_ylabel("IG importance (fraction)")
    ax.legend(ncol=2, loc="upper right"); ax.set_title("Driver shift with lead time (persistence → meteorology)")
    save(fig, "e9_leadtime")

    # Fig 6 (supp) — top individual features: IG vs occlusion
    fa = np.array(ig_out["med_h1"]["feat_abs_frac"]); o1 = np.argsort(fa)[::-1][:12]
    dpf = np.array([occ_feat[FEAT[i]]["dpinball"] for i in range(F)]); o2 = np.argsort(dpf)[::-1][:12]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].barh(range(12), fa[o1][::-1], color=OI["blue"], edgecolor="k", linewidth=0.3)
    ax[0].set_yticks(range(12)); ax[0].set_yticklabels([FEAT[i] for i in o1][::-1], fontsize=8)
    ax[0].set_xlabel("IG importance (fraction, median h=1)"); ax[0].set_title("(a) Integrated Gradients"); ax[0].grid(axis="y", visible=False)
    ax[1].barh(range(12), dpf[o2][::-1], color=OI["orange"], edgecolor="k", linewidth=0.3)
    ax[1].set_yticks(range(12)); ax[1].set_yticklabels([FEAT[i] for i in o2][::-1], fontsize=8)
    ax[1].set_xlabel(r"Occlusion $\Delta$Pinball (test-2016)"); ax[1].set_title("(b) Occlusion importance"); ax[1].grid(axis="y", visible=False)
    fig.suptitle("Top individual features (supplementary)", y=1.0, fontsize=11, fontweight="bold")
    save(fig, "e9_feature_importance")
    print("saved 6 publication-grade figs (png+pdf): e9_group_importance, e9_distributional, "
          "e9_temporal_heatmap, e9_daynight, e9_leadtime, e9_feature_importance", flush=True)
except Exception as e:
    import traceback; print("figure skipped:", e); traceback.print_exc()

# ================= console summary + RESULTS_LOG =================
def topk(vec, k=5): return [(FEAT[i], float(vec[i])) for i in np.argsort(vec)[::-1][:k]]
top_ig = topk(np.array(ig_out["med_h1"]["feat_abs_frac"]))
top_occ = topk(np.array([occ_feat[FEAT[i]]["dpinball"] for i in range(F)]))
grp_ig_med = sorted(ig_out["med_h1"]["group_abs_frac"].items(), key=lambda kv: -kv[1])
grp_occ = sorted(occ_grp.items(), key=lambda kv: -kv[1]["dpinball"])
print("\n=============== E9 explainability (test-2016, frozen ext model) ===============")
print(f"intact: pinball={base_pin:.4f} crps={base_crps:.4f} brier={base_brier:.4f} | IG completeness gap={comp_gap:.4f} | IG~occ group r={agree_r:.3f}")
print("top-5 IG (median h1):", [f"{n}={v:.3f}" for n, v in top_ig])
print("top-5 occlusion dPin :", [f"{n}={v:+.4f}" for n, v in top_occ])
print("group IG (median h1):", [f"{g}={v:.3f}" for g, v in grp_ig_med])
print("group occlusion dPin:", [f"{g}={v['dpinball']:+.4f}" for g, v in grp_occ])

run_id = "E9-" + datetime.now().strftime("%Y%m%d-%H%M%S")
def fmt_g(name): return " / ".join(f"{g} {ig_out[name]['group_abs_frac'][g]:.3f}" for g in GLIST)
dn = ig_out["daynight_med_h1"]
lines = [
    f"\n## Run {run_id}  (E9: explainability — Integrated Gradients + occlusion, frozen ext model)",
    f"- date: {datetime.now().isoformat(timespec='seconds')} | code: pipeline/10_qrpinn_e9_xai.py | method: METHODS_explainability.md",
    f"- subject: FROZEN artefacts/{CKPT.name} (rank index, physics on, 13-tau); POST-HOC, no retrain; seed=0",
    f"- IG: baseline=missing(values=0,masks=0), M={M_IG}, N={nA} stratified test-2016 anchors ({n_day} day/{n_night} night); completeness gap(med h1)={comp_gap:.4f}",
    f"- occlusion: mean-occlude on all {len(idx_te)} test-2016 anchors; intact pinball={base_pin:.4f} crps={base_crps:.4f} brier(exc)={base_brier:.4f}",
    f"- **IG vs occlusion agreement across groups: Pearson r = {agree_r:.3f}** (independent corroboration)",
    "",
    f"- top-5 IG (median h=1): " + ", ".join(f"{n} {v:.3f}" for n, v in top_ig),
    f"- top-5 occlusion dPinball: " + ", ".join(f"{n} {v:+.4f}" for n, v in top_occ),
    "",
    "| group | IG median_h1 | IG tail_h1 | IG width_h1 | IG median_h24 | occ dPinball | occ dBrier |",
    "|---|---|---|---|---|---|---|",
]
for g in GLIST:
    lines.append(f"| {g} | {ig_out['med_h1']['group_abs_frac'][g]:.3f} | {ig_out['tail_h1']['group_abs_frac'][g]:.3f} | "
                 f"{ig_out['width_h1']['group_abs_frac'][g]:.3f} | {ig_out['med_h24']['group_abs_frac'][g]:.3f} | "
                 f"{occ_grp[g]['dpinball']:+.4f} | {occ_grp[g]['dbrier']:+.4f} |")
lines += [
    "",
    f"- station permutation (->id0): dPinball={occ_station['dpinball']:+.4f}, dBrier={occ_station['dbrier']:+.4f}",
    f"- day/night (median h1, IG frac): photochem day={dn['day_group_frac']['photochem']:.3f}/night={dn['night_group_frac']['photochem']:.3f}; "
    f"NOx_side day={dn['day_group_frac']['NOx_side']:.3f}/night={dn['night_group_frac']['NOx_side']:.3f}; "
    f"O3_side day={dn['day_group_frac']['O3_side']:.3f}/night={dn['night_group_frac']['O3_side']:.3f}",
    f"- figs (png+pdf, 300dpi): e9_group_importance, e9_distributional, e9_temporal_heatmap, e9_daynight, e9_leadtime, e9_feature_importance",
    f"- NOTE: saliency-on-time-series caveat (Ismail et al. 2020) — IG (axiomatic, completeness-checked) cross-validated with occlusion in loss currency; agreement (r above) = robustness. Interpretation only, frozen model.",
]
with open(ROOT / "RESULTS_LOG.md", "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nWrote results_e9.json and appended {run_id} to RESULTS_LOG.md", flush=True)
