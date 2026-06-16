# -*- coding: utf-8 -*-
"""
11_qrpinn_e10.py  (E10)  -- Physics-GUIDED HYBRID QR-PINN.

Fix for the two failed physics formulations:
  * E3/E8 soft penalty  -> physics decoupled from the forecast (neutral; PHYSICS_DIAGNOSIS R1).
  * PHYSFORCE (10b)      -> physics was the SOLE forecaster via a 7-param global ODE bottleneck
                            -> capacity collapse, much WORSE (pinball 0.348 vs 0.210 free).

E10 takes the principled middle ground. Physics is STRUCTURAL in the forward path AND interpretable,
but it does not bottleneck forecast capacity:

  (1) Physics decoder in the forward path (fixes R1): a recurrent box-ODE rolls NO, NO2, O3 in ppb
      over h=1..24, driven by FUTURE met incl. a sunlight proxy J (solar-backed photochemical index).
      Decoding NO/NO2/O3 (not just NOx/O3) lets the real chemistry be written (fixes R4).
  (2) Well-conditioned, positive, stable integration (fixes the PHYSFORCE collapse / R3): a
      semi-implicit update  C_{s+1} = (C_s + sources)/(1 + sinks)  -- unconditionally >=0, no relu/clamp
      hacks, no blow-up. Native ppb units (fixes R5). Per-station emission scale handles the 5-vs-100 ppb
      heterogeneity a single global emission rate could not.
  (3) Real chemistry (fixes R4): NO2 + hv -> NO + O3 (photolysis ~ J) and O3 + NO -> NO2 (titration ~
      k(T)). The fast cycle conserves Ox = O3 + NO2 BY CONSTRUCTION (structural Ox quasi-conservation);
      an explicit Leighton photostationary residual  jph*NO2 ~= ktit*NO*O3  is added as a soft constraint.
  (4) Hybrid forecast keeps capacity (fixes the PHYSFORCE bottleneck): the forecast median is the
      physics rollout PLUS a bounded learned residual,  H_med = clamp(H_phys + s*tanh(resid(z)), 0,1).
      When physics is right the residual ->0 (interpretable physics forecast); when physics is
      incomplete the residual absorbs the rest. A concentration data-fit pins the ODE to REAL dynamics
      regardless of the residual, so physics cannot become a free-floating decoration.

Decisive, honest test (PHYSICS_DIAGNOSIS point 4/5): same temporal split (train 2014-15 / test 2016),
rank index H, seeds {0,1}. Three variants:
  FREE      -- no ODE, free MLP median (the ~0.205 data-only baseline).
  HYBRID    -- physics decoder + bounded NN residual + chemistry losses (E10).
  PHYS-ONLY -- residual disabled (H_med = H_phys): what physics ALONE forecasts.
Verdict: HYBRID validated if pinball/CRPS <= FREE (no accuracy penalty) AND learned rates physical.
If neutral -> reported honestly as physics-guided/interpretable QR (NOT a physics-accuracy claim).

Outputs: results_e10.json, figs/e10_physics.png, append RESULTS_LOG.md.  Observed H only; no synthetic data.
"""
import json, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from datetime import datetime

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
ART = ROOT / "artefacts"; FIGS = ROOT / "figs"; FIGS.mkdir(exist_ok=True)

L, HZ = 48, 24
TAUS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95], dtype=np.float32)
nT = len(TAUS); MED = int(np.where(np.isclose(TAUS, 0.5))[0][0])
EPOCHS, BATCH, LR = 10, 512, 1e-3
LAM_C, LAM_LEIGH, ANNEAL = 0.3, 0.05, 4   # conc data-fit, Leighton residual, physics-loss anneal epochs

# ---- scalers (train-only, from meta) ----
meta = json.load(open(ART / "qrpinn_meta.json"))
rs = meta["robust_scaler"]
medNO, iqrNO = rs["NO_ppb"]["median"], rs["NO_ppb"]["iqr"]
medN2, iqrN2 = rs["NO2_ppb"]["median"], rs["NO2_ppb"]["iqr"]
medN,  iqrN  = rs["NOx_ppb"]["median"], rs["NOx_ppb"]["iqr"]
medO,  iqrO  = rs["O3_ppb"]["median"], rs["O3_ppb"]["iqr"]
ss = meta["standard_scaler"]
# PHYS col order: [ventilation_coefficient, boundary_layer_height_m, photochemical_activitiy_index, precip_mm, temp_C]
VCm, VCs   = ss["ventilation_coefficient"]["mean"], ss["ventilation_coefficient"]["std"]
BLm, BLs   = ss["boundary_layer_height_m"]["mean"], ss["boundary_layer_height_m"]["std"]
Jm,  Js_   = ss["photochemical_activitiy_index"]["mean"], ss["photochemical_activitiy_index"]["std"]
Pm,  Ps_   = ss["precip_mm"]["mean"], ss["precip_mm"]["std"]
Tm,  Ts_   = ss["temp_C"]["mean"], ss["temp_C"]["std"]

# ---- data ----
d = np.load(ART / "qrpinn_data.npz", allow_pickle=True)
X = d["X"].astype(np.float32); PHYS = d["PHYS"].astype(np.float32)
COBS = d["COBS"].astype(np.float32); CMASK = d["CMASK"].astype(np.float32)   # [.,.,0]=NOx, [.,.,1]=O3 (robust-scaled)
Hrob = d["H"].astype(np.float32); year = d["year"].astype(np.int32); S, T, F = X.shape

# ---- rank index H + train empirical-CDF grids (train-only) ----
both = (CMASK[:, :, 0] == 1) & (CMASK[:, :, 1] == 1) & (~np.isnan(Hrob))
yr2d = np.broadcast_to(year[None, :], (S, T)); tr_both = both & (yr2d <= 2015)
sN = np.sort(COBS[:, :, 0][tr_both]); sO = np.sort(COBS[:, :, 1][tr_both])   # scaled NOx, O3
def Fcdf(s, v): return np.searchsorted(s, v, side="right") / len(s)
H = np.full((S, T), np.nan, np.float32)
H[both] = (0.5 * Fcdf(sN, COBS[:, :, 0][both]) + 0.5 * Fcdf(sO, COBS[:, :, 1][both])).astype(np.float32)
thr = float(np.quantile(H[tr_both], 0.30))
glev = np.linspace(0, 1, 256).astype(np.float32)
gzN = torch.tensor(np.quantile(sN, glev).astype(np.float32))   # scaled-NOx knots
gzO = torch.tensor(np.quantile(sO, glev).astype(np.float32))   # scaled-O3 knots
GLEV = torch.tensor(glev)
print(f"rank thr={thr:.4f}; CDF grids ready (S={S} T={T} F={F})", flush=True)

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
print(f"train={len(idx_tr)} test={len(idx_te)}", flush=True)

class DS(torch.utils.data.Dataset):
    def __init__(self, idx): self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        si, t = int(self.idx[i, 0]), int(self.idx[i, 1])
        fut = H[si, t + 1:t + 1 + HZ].copy(); tm = (~np.isnan(fut)).astype(np.float32); fut[np.isnan(fut)] = 0.0
        return (torch.from_numpy(X[si, t - L + 1:t + 1, :]), torch.tensor(si),
                torch.from_numpy(PHYS[si, t + 1:t + 1 + HZ, :]),    # FUTURE drivers
                torch.from_numpy(COBS[si, t + 1:t + 1 + HZ, :]),    # FUTURE observed NOx,O3 (scaled)
                torch.from_numpy(CMASK[si, t + 1:t + 1 + HZ, :]),
                torch.from_numpy(fut), torch.from_numpy(tm))

def interp1d(x, xp, fp):
    xc = x.clamp(float(xp[0]), float(xp[-1]))
    idx = torch.searchsorted(xp, xc).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]; y0, y1 = fp[idx - 1], fp[idx]
    return (y0 + (xc - x0) / (x1 - x0 + 1e-9) * (y1 - y0)).clamp(0, 1)

class E10(nn.Module):
    """mode in {'hybrid','physonly','free'}. hybrid/physonly share the physics decoder."""
    def __init__(self, F, n_station, hidden=64, emb=8, mode="hybrid"):
        super().__init__(); self.mode = mode
        self.emb = nn.Embedding(n_station, emb); self.lstm = nn.LSTM(F, hidden, batch_first=True)
        din = hidden + emb
        self.inc = nn.Linear(din, HZ * (nT - 1))          # quantile spread (monotone increments)
        self.spread = nn.Parameter(torch.tensor(-2.0))    # band scale (softplus)
        if mode == "free":
            self.hmed = nn.Linear(din, HZ)                # free median forecast of H
            return
        # ---- physics decoder ----
        self.init_state = nn.Linear(din, 3)               # -> (NO0, NO20, O30) ppb via softplus
        # interpretable GLOBAL base rates (softplus>0): photolysis, titration, arrhenius,
        # dep_no, dep_no2, dep_o3, dilution(VC), wet, Ox-production
        self.rate_raw = nn.Parameter(torch.tensor(
            [-1.0, -3.0, -2.0, -2.5, -2.5, -2.5, -2.5, -3.0, -1.5], dtype=torch.float32))
        self.emis_raw = nn.Parameter(torch.tensor(-1.0))           # global emission base (ppb/h)
        self.emis_station = nn.Embedding(n_station, 1)             # per-station emission log-scale
        self.emis_station.weight.data.zero_()
        if mode == "hybrid":
            self.resid = nn.Linear(din, HZ)               # bounded NN correction on H
            self.resid.weight.data.mul_(0.1); self.resid.bias.data.zero_()
            self.res_scale = nn.Parameter(torch.tensor(-1.5))      # softplus -> ~0.18 cap, learnable

    def forward(self, x, si, pf):
        out, (hn, cn) = self.lstm(x); z = torch.cat([hn[-1], self.emb(si)], 1)
        inc = torch.nn.functional.softplus(self.inc(z)).view(-1, HZ, nT - 1)
        cum = torch.cat([torch.zeros_like(inc[:, :, :1]), torch.cumsum(inc, -1)], -1)   # [B,HZ,nT]
        off = (cum - cum[:, :, MED:MED + 1]) * torch.nn.functional.softplus(self.spread)
        if self.mode == "free":
            Hmed = torch.sigmoid(self.hmed(z))
            Q = (Hmed.unsqueeze(-1) + off).clamp(0, 1)
            return Q, Hmed, None, None

        # ---- physics rollout (ppb, semi-implicit positive) ----
        s0 = torch.nn.functional.softplus(self.init_state(z))
        no  = s0[:, 0] * iqrNO + medNO          # NO0  ppb >=0
        no2 = s0[:, 1] * iqrN2 + medN2          # NO2_0 ppb
        o   = s0[:, 2] * iqrO  + medO           # O3_0  ppb
        r = torch.nn.functional.softplus(self.rate_raw)
        kj, ktit0, Ea, dep_no, dep_no2, dep_o3, dil, wet, P_o3 = [r[k] for k in range(9)]
        E_nox = torch.nn.functional.softplus(self.emis_raw) * torch.exp(self.emis_station(si).squeeze(-1))  # [B]
        Hphys, leigh = [], []
        nox_tr, o_tr = [], []
        for s in range(HZ):
            J  = torch.relu(pf[:, s, 2] * Js_ + Jm) / Jm                      # sunlight proxy ~O(1)
            vc = torch.relu(pf[:, s, 0] * VCs + VCm) / VCm                    # ventilation ~O(1)
            bl = torch.relu(pf[:, s, 1] * BLs + BLm) + 50.0                   # BLH (m), floored
            pr = torch.relu(pf[:, s, 3] * Ps_ + Pm)                          # precip (mm) >=0
            tC = pf[:, s, 4] * Ts_ + Tm                                       # temp (C)
            jph = kj * J                                                      # photolysis 1/h
            tit = ktit0 * torch.exp(Ea * (tC - 25.0) / 25.0)                  # titration 1/(ppb h), Arrhenius
            sinkb = dil * vc + wet * pr                                       # common dilution+scavenging 1/h
            emisNO = E_nox / (1.0 + bl / BLm)                                 # emission into mixing layer ppb/h
            prodO3 = P_o3 * J                                                 # slow Ox photo-production ppb/h
            # semi-implicit, unconditionally positive update (fast cycle conserves Ox = O3+NO2)
            no2_n = (no2 + tit * no * o)              / (1.0 + jph + dep_no2 + sinkb)
            no_n  = (no  + jph * no2 + emisNO)        / (1.0 + tit * o + dep_no + sinkb)
            o_n   = (o   + jph * no2 + prodO3)        / (1.0 + tit * no + dep_o3 + sinkb)
            no, no2, o = no_n, no2_n, o_n
            nox = no + no2
            rN = interp1d((nox - medN) / iqrN, gzN, GLEV)                     # F_NOx
            rO = interp1d((o   - medO) / iqrO, gzO, GLEV)                     # F_O3
            Hphys.append(0.5 * (rN + rO))
            leigh.append((jph * no2 - tit * no * o) / (no2 + 1.0))           # Leighton residual (normalized)
            nox_tr.append((nox - medN) / iqrN); o_tr.append((o - medO) / iqrO)
        Hphys = torch.stack(Hphys, 1)                                        # [B,HZ]
        leigh = torch.stack(leigh, 1)
        chat = torch.stack([torch.stack(nox_tr, 1), torch.stack(o_tr, 1)], -1)   # [B,HZ,2] scaled NOx,O3

        if self.mode == "physonly":
            Hmed = Hphys
        else:  # hybrid
            delta = torch.nn.functional.softplus(self.res_scale) * torch.tanh(self.resid(z))
            Hmed = (Hphys + delta).clamp(0, 1)
        Q = (Hmed.unsqueeze(-1) + off).clamp(0, 1)
        return Q, Hmed, chat, leigh

def pinball_loss(q, y, tm):
    taus = torch.tensor(TAUS).view(1, 1, nT); err = y.unsqueeze(-1) - q
    return (torch.maximum(taus * err, (taus - 1) * err).mean(-1) * tm).sum() / tm.sum().clamp(min=1)

def train(mode, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = E10(F, S, mode=mode); opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl = torch.utils.data.DataLoader(DS(idx_tr), batch_size=BATCH, shuffle=True, num_workers=0)
    for ep in range(EPOCHS):
        model.train(); a = min(1.0, (ep + 1) / ANNEAL); t0 = time.time(); tot = nb = 0
        for x, si, pf, cobs, cmask, fut, tm in dl:
            opt.zero_grad(); Q, Hmed, chat, leigh = model(x, si, pf)
            loss = pinball_loss(Q, fut, tm)
            if chat is not None:
                loss = loss + a * LAM_C * (((chat - cobs) ** 2 * cmask).sum() / cmask.sum().clamp(min=1))
                loss = loss + a * LAM_LEIGH * (leigh ** 2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += loss.item(); nb += 1
        print(f"  [{mode:8s}] ep {ep+1}/{EPOCHS} loss={tot/nb:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return model

def predict(model):
    model.eval(); dl = torch.utils.data.DataLoader(DS(idx_te), batch_size=1024, shuffle=False, num_workers=0)
    Q, Y, M, HP = [], [], [], []
    with torch.no_grad():
        for x, si, pf, cobs, cmask, fut, tm in dl:
            q, hmed, _, _ = model(x, si, pf)
            Q.append(q.numpy()); Y.append(fut.numpy()); M.append(tm.numpy()); HP.append(hmed.numpy())
    return np.concatenate(Q), np.concatenate(Y), np.concatenate(M), np.concatenate(HP)

def cdf_at_thr(Q, thr):
    N, m = Q.shape; k = (Q < thr).sum(1); lo = np.clip(k - 1, 0, m - 1); hi = np.clip(k, 0, m - 1); ar = np.arange(N)
    qlo, qhi, tlo, thi = Q[ar, lo], Q[ar, hi], TAUS[lo], TAUS[hi]
    gap = qhi - qlo; den = np.where(gap > 1e-9, gap, 1.0); frac = np.where(gap > 1e-9, (thr - qlo) / den, 0.0)
    cdf = tlo + frac * (thi - tlo); cdf = np.where(k == 0, TAUS[0], cdf); cdf = np.where(k == m, TAUS[-1], cdf)
    return np.clip(cdf, 0, 1)

def metrics(Q, Y, M):
    taus = TAUS.reshape(1, 1, nT); err = Y[..., None] - Q; pb = np.maximum(taus * err, (taus - 1) * err); m3 = M[..., None]
    pin = float((pb * m3).sum() / m3.sum()); crps = float(2 * ((pb * m3).sum((0, 1)) / m3.sum((0, 1))).mean())
    def cov(a, b):
        lo, hi = Q[:, :, a], Q[:, :, b]; return float((((Y >= lo) & (Y <= hi)).astype(float) * M).sum() / M.sum())
    Qf, Yf, Mf = Q.reshape(-1, nT), Y.reshape(-1), M.reshape(-1)
    pexc = 1.0 - cdf_at_thr(Qf, thr); ind = (Yf >= thr).astype(float); sel = Mf > 0
    return {"pinball": pin, "crps": crps, "picp80": cov(1, 7), "picp90": cov(0, 8),
            "brier_exceed": float((((pexc - ind) ** 2)[sel]).mean())}

def learned_rates(model):
    r = torch.nn.functional.softplus(model.rate_raw).detach().numpy()
    names = ["photolysis_kj", "titration_ktit0", "arrhenius_Ea", "dep_NO", "dep_NO2", "dep_O3",
             "dilution_VC", "wet_scav", "Ox_production"]
    out = {n: float(v) for n, v in zip(names, r)}
    out["emission_base"] = float(torch.nn.functional.softplus(model.emis_raw).item())
    es = torch.exp(model.emis_station.weight.detach().squeeze(-1)).numpy()
    out["emission_station_scale"] = {meta["stations"][i]: float(es[i]) for i in range(S)}
    if model.mode == "hybrid":
        out["resid_scale_cap"] = float(torch.nn.functional.softplus(model.res_scale).item())
    return out

# ===================== RUN =====================
results, preds = {}, {}
for mode in ("free", "physonly", "hybrid"):
    print(f"\n== {mode.upper()} ==", flush=True)
    m = train(mode, seed=0); Q, Y, M, HP = predict(m)
    results[mode] = metrics(Q, Y, M); preds[mode] = (Q, Y, M, HP)
    print(f"   {mode}: {results[mode]}", flush=True)
    if mode == "hybrid":
        hyb_model = m
        # quantify physics' role: corr between hybrid median and the physics rollout
        Qh, Yh, Mh, HPh = preds["hybrid"]; Qp, _, _, HPp = preds["physonly"]
        sel = Mh.reshape(-1) > 0
        med_h = Qh[:, :, MED].reshape(-1)[sel]; med_phys = HPp.reshape(-1)[sel]
        results["hybrid"]["corr_median_vs_physics"] = float(np.corrcoef(med_h, med_phys)[0, 1])

# seed stability for hybrid + free (the decisive pair)
print("\n== seed 1 (free, hybrid) ==", flush=True)
seed1 = {}
for mode in ("free", "hybrid"):
    m = train(mode, seed=1); Q, Y, M, HP = predict(m); seed1[mode] = metrics(Q, Y, M)
    print(f"   seed1 {mode}: {seed1[mode]}", flush=True)

theta = learned_rates(hyb_model)
dvf = {k: results["hybrid"][k] - results["free"][k] for k in ("pinball", "crps", "brier_exceed")}
dpf = {k: results["physonly"][k] - results["free"][k] for k in ("pinball", "crps", "brier_exceed")}

R = {"config": {"epochs": EPOCHS, "thr": thr, "lam_c": LAM_C, "lam_leighton": LAM_LEIGH,
                "taus": TAUS.tolist(), "sunlight_proxy": "photochemical_activitiy_index",
                "integration": "semi-implicit positive box-ODE (NO/NO2/O3, ppb)"},
     "seed0": results, "seed1": seed1, "learned_rates": theta,
     "delta_hybrid_minus_free": dvf, "delta_physonly_minus_free": dpf}
json.dump(R, open(ROOT / "results_e10.json", "w"), indent=2)

# ---- figure ----
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = ["pinball", "crps", "brier_exceed"]; x = np.arange(len(names)); w = 0.26
    plt.figure(figsize=(7.5, 4.2))
    plt.bar(x - w, [results["free"][k] for k in names], w, label="FREE (no physics)", color="tab:gray")
    plt.bar(x,     [results["hybrid"][k] for k in names], w, label="HYBRID (E10)", color="tab:green")
    plt.bar(x + w, [results["physonly"][k] for k in names], w, label="PHYS-ONLY", color="tab:orange")
    plt.xticks(x, names); plt.ylabel("test 2016 (rank index)"); plt.legend()
    plt.title("E10: physics-guided hybrid vs free vs physics-only"); plt.tight_layout()
    plt.savefig(FIGS / "e10_physics.png", dpi=140); plt.close(); print("saved figs/e10_physics.png", flush=True)
except Exception as e:
    print("figure skipped:", e, flush=True)

# ---- console summary ----
print("\n=============== E10 (test 2016, rank index, seed 0) ===============")
for mode in ("free", "hybrid", "physonly"):
    v = results[mode]
    print(f"{mode:9s}: pinball={v['pinball']:.4f} crps={v['crps']:.4f} picp80={v['picp80']:.3f} "
          f"picp90={v['picp90']:.3f} brier={v['brier_exceed']:.4f}")
print(f"delta (hybrid-free): pinball={dvf['pinball']:+.4f} crps={dvf['crps']:+.4f} brier={dvf['brier_exceed']:+.4f}")
print(f"delta (physonly-free): pinball={dpf['pinball']:+.4f} crps={dpf['crps']:+.4f} brier={dpf['brier_exceed']:+.4f}")
print("learned global rates:", {k: round(v, 4) for k, v in theta.items() if k != "emission_station_scale"})

# ---- verdict + log ----
verdict = ("physics HELPS" if (dvf['pinball'] <= -0.003 or dvf['crps'] <= -0.0015)
           else ("physics NEUTRAL" if abs(dvf['pinball']) < 0.003 else "physics WORSE"))
run_id = "E10-" + datetime.now().strftime("%Y%m%d-%H%M%S")
r0, r1 = results, seed1
lines = [f"\n## Run {run_id}  (E10: physics-GUIDED HYBRID — NN-residual forecast + interpretable NO/NO2/O3 box-ODE)",
         f"- date: {datetime.now().isoformat(timespec='seconds')} | code: pipeline/11_qrpinn_e10.py | rank index; epochs={EPOCHS}; seed=0(+1)",
         f"- design: H_med = H_phys(box-ODE rollout, ppb, semi-implicit) + bounded NN residual; chemistry NO2+hv->NO+O3, O3+NO->NO2; Leighton residual + structural Ox=O3+NO2 conservation; lam_c={LAM_C}, lam_leighton={LAM_LEIGH}",
         "", "**Test-2016 (seed 0)**", "| variant | pinball | CRPS | PICP80 | PICP90 | Brier(exc) |", "|---|---|---|---|---|---|",
         f"| FREE (no physics) | {r0['free']['pinball']:.4f} | {r0['free']['crps']:.4f} | {r0['free']['picp80']:.3f} | {r0['free']['picp90']:.3f} | {r0['free']['brier_exceed']:.4f} |",
         f"| HYBRID (E10) | {r0['hybrid']['pinball']:.4f} | {r0['hybrid']['crps']:.4f} | {r0['hybrid']['picp80']:.3f} | {r0['hybrid']['picp90']:.3f} | {r0['hybrid']['brier_exceed']:.4f} |",
         f"| PHYS-ONLY | {r0['physonly']['pinball']:.4f} | {r0['physonly']['crps']:.4f} | {r0['physonly']['picp80']:.3f} | {r0['physonly']['picp90']:.3f} | {r0['physonly']['brier_exceed']:.4f} |",
         "", "**Seed stability (seed 1)**", "| variant | pinball | CRPS | PICP90 | Brier(exc) |", "|---|---|---|---|---|",
         f"| FREE | {r1['free']['pinball']:.4f} | {r1['free']['crps']:.4f} | {r1['free']['picp90']:.3f} | {r1['free']['brier_exceed']:.4f} |",
         f"| HYBRID | {r1['hybrid']['pinball']:.4f} | {r1['hybrid']['crps']:.4f} | {r1['hybrid']['picp90']:.3f} | {r1['hybrid']['brier_exceed']:.4f} |",
         "", f"- delta (HYBRID - FREE): pinball {dvf['pinball']:+.4f}, CRPS {dvf['crps']:+.4f}, Brier {dvf['brier_exceed']:+.4f}",
         f"- delta (PHYS-ONLY - FREE): pinball {dpf['pinball']:+.4f}, CRPS {dpf['crps']:+.4f}, Brier {dpf['brier_exceed']:+.4f}",
         f"- corr(hybrid median, physics rollout) = {r0['hybrid'].get('corr_median_vs_physics', float('nan')):.3f} (how much the forecast follows physics)",
         f"- learned global rates (softplus>0): " + ", ".join(f"{k}={v:.4f}" for k, v in theta.items() if k != "emission_station_scale"),
         f"- per-station emission scale: " + ", ".join(f"{k}={v:.2f}" for k, v in theta["emission_station_scale"].items()),
         f"- **VERDICT: {verdict}** (threshold ~0.003 pinball / 0.0015 CRPS vs E8 noise)",
         f"- fig figs/e10_physics.png ; NOTE rank-index [0,1] metrics; future met used as forcing (forecast-met assumption)."]
with open(ROOT / "RESULTS_LOG.md", "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nWrote results_e10.json and appended {run_id} to RESULTS_LOG.md ({verdict})", flush=True)
