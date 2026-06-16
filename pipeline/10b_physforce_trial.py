# -*- coding: utf-8 -*-
"""
10_qrpinn_e9.py  (E9)  -- Physics-FORCED QR-PINN: physics structured into the forecast.

Fix for the E8/PHYSICS_DIAGNOSIS finding that the old soft-penalty physics was neutral (R1,R3,R4).
Here the forecast MEDIAN of the combined index H is produced by a recurrent, sunlight-driven
NOx-O3 box-ODE decoder rolled over h=1..24 (physics is in the forward path, not a side loss):

  O3_{s+1}  = relu( O3_s + th_prod*J_s                         # sunlight-driven O3 photo-production
                          - th_titr*(NOx_s*O3_s)/100           # NO titration of O3 (coupling)
                          - th_depO*O3_s - th_dil*VC_s*O3_s - th_wet*P_s*O3_s )
  NOx_{s+1} = relu( NOx_s + th_emis/(1+J_s)                    # emission proxy (higher at night/low sun)
                           - th_conv*J_s*NOx_s                  # sunlight-driven NOx processing
                           - th_dil*VC_s*NOx_s - th_wet*P_s*NOx_s )
J_s = sunlight proxy (photochemical_activitiy_index, backed by solar) at the FUTURE hour t+s.
Decoded (NOx,O3) in ppb -> train empirical CDF -> H_phys_{t+h}; a monotone spread head adds the
quantile band around H_phys (so the predictive distribution is anchored on the physics rollout).

Compared against a matched NO-PHYSICS ablation (free MLP forecasts H median from the same encoder).
Losses: pinball(H) [+ for physics: lam_c * MSE(decoded NOx,O3 vs FUTURE observed, scaled, masked)].
Rank index H = 0.5 F_NOx(NOx)+0.5 F_O3(O3); train 2014-15 / test 2016; seed 0. Observed H only.
Outputs: results_e9.json, figs/e9_physics.png, append RESULTS_LOG.md.
"""
import json, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from datetime import datetime

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
ART = ROOT / "artefacts"; FIGS = ROOT / "figs"; FIGS.mkdir(exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

L, HZ = 48, 24
TAUS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95], dtype=np.float32)
nT = len(TAUS); MED = int(np.where(np.isclose(TAUS, 0.5))[0][0])
EPOCHS, BATCH, LR, LAM_C = 12, 512, 1e-3, 0.3

meta = json.load(open(ART / "qrpinn_meta.json"))
medN, iqrN = meta["robust_scaler"]["NOx_ppb"]["median"], meta["robust_scaler"]["NOx_ppb"]["iqr"]
medO, iqrO = meta["robust_scaler"]["O3_ppb"]["median"], meta["robust_scaler"]["O3_ppb"]["iqr"]
ss = meta["standard_scaler"]
# PHYS column order: [ventilation_coefficient, boundary_layer_height_m, photochemical_activitiy_index, precip_mm, temp_C]
VCm, VCs = ss["ventilation_coefficient"]["mean"], ss["ventilation_coefficient"]["std"]
Jm, Js_ = ss["photochemical_activitiy_index"]["mean"], ss["photochemical_activitiy_index"]["std"]
Pm, Ps_ = ss["precip_mm"]["mean"], ss["precip_mm"]["std"]

d = np.load(ART / "qrpinn_data.npz", allow_pickle=True)
X = d["X"].astype(np.float32); PHYS = d["PHYS"].astype(np.float32)
COBS = d["COBS"].astype(np.float32); CMASK = d["CMASK"].astype(np.float32)
Hrob = d["H"].astype(np.float32); year = d["year"].astype(np.int32); S, T, F = X.shape

# ---- rank index H + train CDF grids ----
both = (CMASK[:, :, 0] == 1) & (CMASK[:, :, 1] == 1) & (~np.isnan(Hrob))
yr2d = np.broadcast_to(year[None, :], (S, T)); tr_both = both & (yr2d <= 2015)
sN = np.sort(COBS[:, :, 0][tr_both]); sO = np.sort(COBS[:, :, 1][tr_both])
def Fcdf(s, v): return np.searchsorted(s, v, side="right")/len(s)
H = np.full((S, T), np.nan, np.float32)
H[both] = (0.5*Fcdf(sN, COBS[:, :, 0][both]) + 0.5*Fcdf(sO, COBS[:, :, 1][both])).astype(np.float32)
thr = float(np.quantile(H[tr_both], 0.30))
# differentiable CDF grids (z-space): 256 quantile knots
glev = np.linspace(0, 1, 256).astype(np.float32)
gzN = torch.tensor(np.quantile(sN, glev).astype(np.float32)); gzO = torch.tensor(np.quantile(sO, glev).astype(np.float32))
GLEV = torch.tensor(glev)
print(f"rank thr={thr:.4f}; CDF grids ready", flush=True)

def build_index():
    tr, te = [], []
    for si in range(S):
        Hs = H[si]
        for t in range(L-1, T-HZ):
            if np.isnan(Hs[t]): continue
            if not np.any(~np.isnan(Hs[t+1:t+1+HZ])): continue
            if year[t+HZ] <= 2015: tr.append((si, t))
            elif year[t] == 2016: te.append((si, t))
    return np.array(tr, np.int32), np.array(te, np.int32)
idx_tr, idx_te = build_index()
print(f"train={len(idx_tr)} test={len(idx_te)}", flush=True)

class DS(torch.utils.data.Dataset):
    def __init__(self, idx): self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        si, t = int(self.idx[i, 0]), int(self.idx[i, 1])
        fut = H[si, t+1:t+1+HZ].copy(); tm = (~np.isnan(fut)).astype(np.float32); fut[np.isnan(fut)] = 0.0
        return (torch.from_numpy(X[si, t-L+1:t+1, :]), torch.tensor(si),
                torch.from_numpy(PHYS[si, t+1:t+1+HZ, :]),       # FUTURE drivers
                torch.from_numpy(COBS[si, t+1:t+1+HZ, :]),       # FUTURE observed NOx,O3 (scaled)
                torch.from_numpy(CMASK[si, t+1:t+1+HZ, :]),
                torch.from_numpy(fut), torch.from_numpy(tm))

def interp1d(x, xp, fp):
    xc = x.clamp(float(xp[0]), float(xp[-1]))
    idx = torch.searchsorted(xp, xc).clamp(1, xp.numel()-1)
    x0, x1 = xp[idx-1], xp[idx]; y0, y1 = fp[idx-1], fp[idx]
    return (y0 + (xc-x0)/(x1-x0+1e-9)*(y1-y0)).clamp(0, 1)

class E9(nn.Module):
    def __init__(self, F, n_station, hidden=64, emb=8, physics=True):
        super().__init__(); self.physics = physics
        self.emb = nn.Embedding(n_station, emb); self.lstm = nn.LSTM(F, hidden, batch_first=True)
        din = hidden+emb
        self.inc = nn.Linear(din, HZ*(nT-1))             # quantile spread (monotone)
        self.spread = nn.Parameter(torch.tensor(-2.0))   # band scale (softplus)
        if physics:
            self.init_state = nn.Linear(din, 2)          # -> (NOx0, O30) ppb via softplus
            self.th = nn.Parameter(torch.zeros(7))       # prod,titr,depO,dil,wet,emis,conv
        else:
            self.hmed = nn.Linear(din, HZ)               # free median forecast of H
    def forward(self, x, si, pf):
        out, (hn, cn) = self.lstm(x); z = torch.cat([hn[-1], self.emb(si)], 1)
        if self.physics:
            n = torch.nn.functional.softplus(self.init_state(z)[:, 0]) * iqrN + medN   # NOx0 ppb>=0
            o = torch.nn.functional.softplus(self.init_state(z)[:, 1]) * iqrO + medO   # O30 ppb>=0
            th = torch.nn.functional.softplus(self.th)
            prod, titr, depO, dil, wet, emis, conv = [th[k] for k in range(7)]
            Hphys = []; n_tr = []; o_tr = []
            for s in range(HZ):
                photo_phys = pf[:, s, 2]*Js_ + Jm; J = torch.relu(photo_phys)/Jm           # sunlight proxy ~O(1)
                vc = torch.relu(pf[:, s, 0]*VCs + VCm)/VCm
                pr = torch.relu(pf[:, s, 3]*Ps_ + Pm)
                o = torch.relu(o + prod*J - titr*(n*o)/100.0 - depO*o - dil*vc*o - wet*pr*o).clamp(max=400)
                n = torch.relu(n + emis/(1.0+J) - conv*J*n - dil*vc*n - wet*pr*n).clamp(max=600)
                rN = interp1d((n-medN)/iqrN, gzN, GLEV); rO = interp1d((o-medO)/iqrO, gzO, GLEV)
                Hphys.append(0.5*(rN+rO)); n_tr.append((n-medN)/iqrN); o_tr.append((o-medO)/iqrO)
            Hmed = torch.stack(Hphys, 1)                                   # [B,HZ]
            chat = torch.stack([torch.stack(n_tr, 1), torch.stack(o_tr, 1)], -1)  # [B,HZ,2] scaled
        else:
            Hmed = torch.sigmoid(self.hmed(z)); chat = None
        inc = torch.nn.functional.softplus(self.inc(z)).view(-1, HZ, nT-1)
        cum = torch.cat([torch.zeros_like(inc[:, :, :1]), torch.cumsum(inc, -1)], -1)  # [B,HZ,nT]
        off = (cum - cum[:, :, MED:MED+1]) * torch.nn.functional.softplus(self.spread)
        Q = (Hmed.unsqueeze(-1) + off).clamp(0, 1)
        return Q, Hmed, chat

def pinball_loss(q, y, tm):
    taus = torch.tensor(TAUS).view(1, 1, nT); err = y.unsqueeze(-1)-q
    return (torch.maximum(taus*err, (taus-1)*err).mean(-1)*tm).sum()/tm.sum().clamp(min=1)

def train(physics, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = E9(F, S, physics=physics); opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl = torch.utils.data.DataLoader(DS(idx_tr), batch_size=BATCH, shuffle=True, num_workers=0)
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = nb = 0
        for x, si, pf, cobs, cmask, fut, tm in dl:
            opt.zero_grad(); Q, Hmed, chat = model(x, si, pf)
            loss = pinball_loss(Q, fut, tm)
            if physics:
                loss = loss + LAM_C*(((chat-cobs)**2*cmask).sum()/cmask.sum().clamp(min=1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += loss.item(); nb += 1
        print(f"  [{'PHYS' if physics else 'FREE'}] ep {ep+1}/{EPOCHS} loss={tot/nb:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return model
def predict(model):
    model.eval(); dl = torch.utils.data.DataLoader(DS(idx_te), batch_size=1024, shuffle=False, num_workers=0)
    Q, Y, M = [], [], []
    with torch.no_grad():
        for x, si, pf, cobs, cmask, fut, tm in dl:
            q, _, _ = model(x, si, pf); Q.append(q.numpy()); Y.append(fut.numpy()); M.append(tm.numpy())
    return np.concatenate(Q), np.concatenate(Y), np.concatenate(M)
def cdf_at_thr(Q, thr):
    N, m = Q.shape; k = (Q < thr).sum(1); lo = np.clip(k-1, 0, m-1); hi = np.clip(k, 0, m-1); ar = np.arange(N)
    qlo, qhi, tlo, thi = Q[ar, lo], Q[ar, hi], TAUS[lo], TAUS[hi]
    gap = qhi-qlo; den = np.where(gap > 1e-9, gap, 1.0); frac = np.where(gap > 1e-9, (thr-qlo)/den, 0.0)
    cdf = tlo+frac*(thi-tlo); cdf = np.where(k == 0, TAUS[0], cdf); cdf = np.where(k == m, TAUS[-1], cdf)
    return np.clip(cdf, 0, 1)
def metrics(Q, Y, M):
    taus = TAUS.reshape(1, 1, nT); err = Y[..., None]-Q; pb = np.maximum(taus*err, (taus-1)*err); m3 = M[..., None]
    pin = float((pb*m3).sum()/m3.sum()); crps = float(2*((pb*m3).sum((0, 1))/m3.sum((0, 1))).mean())
    def cov(a, b):
        lo, hi = Q[:, :, a], Q[:, :, b]; return float((((Y >= lo)&(Y <= hi)).astype(float)*M).sum()/M.sum())
    Qf, Yf, Mf = Q.reshape(-1, nT), Y.reshape(-1), M.reshape(-1)
    pexc = 1.0-cdf_at_thr(Qf, thr); ind = (Yf >= thr).astype(float); sel = Mf > 0
    return {"pinball": pin, "crps": crps, "picp80": cov(1, 7), "picp90": cov(0, 8),
            "brier_exceed": float((((pexc-ind)**2)[sel]).mean())}

# ---- run ----
print("\n== PHYSICS-FORCED (ODE decoder, sunlight-driven) ==", flush=True)
mP = train(True); QP, YP, MP = predict(mP); rP = metrics(QP, YP, MP)
theta = {k: float(torch.nn.functional.softplus(mP.th[i]).item())
         for i, k in enumerate(["prod", "titr", "depO", "dil", "wet", "emis", "conv"])}
print("\n== NO-PHYSICS ablation (free MLP median) ==", flush=True)
mF = train(False); QF, YF, MF = predict(mF); rF = metrics(QF, YF, MF)

R = {"config": {"epochs": EPOCHS, "thr": thr, "lam_c": LAM_C, "sunlight_proxy": "photochemical_activitiy_index"},
     "physics": rP, "free": rF, "learned_theta": theta,
     "delta_physics_minus_free": {k: rP[k]-rF[k] for k in ("pinball", "crps", "brier_exceed")}}
json.dump(R, open(ROOT / "results_e9.json", "w"), indent=2)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = ["pinball", "crps", "brier_exceed"]; x = np.arange(len(names))
    plt.figure(figsize=(7, 4))
    plt.bar(x-0.2, [rP[k] for k in names], 0.4, label="physics-forced", color="tab:green")
    plt.bar(x+0.2, [rF[k] for k in names], 0.4, label="no-physics (free)", color="tab:gray")
    plt.xticks(x, names); plt.ylabel("test 2016 (rank index)"); plt.legend()
    plt.title("E9: physics-structured forecast vs free forecast"); plt.tight_layout()
    plt.savefig(FIGS / "e9_physics.png", dpi=130); plt.close(); print("saved figs/e9_physics.png", flush=True)
except Exception as e:
    print("figure skipped:", e, flush=True)

print("\n=============== E9 (test 2016, rank index) ===============")
print(f"physics : pinball={rP['pinball']:.4f} crps={rP['crps']:.4f} picp80={rP['picp80']:.3f} picp90={rP['picp90']:.3f} brier={rP['brier_exceed']:.4f}")
print(f"free    : pinball={rF['pinball']:.4f} crps={rF['crps']:.4f} picp80={rF['picp80']:.3f} picp90={rF['picp90']:.3f} brier={rF['brier_exceed']:.4f}")
print(f"delta (phys-free): pinball={rP['pinball']-rF['pinball']:+.4f} crps={rP['crps']-rF['crps']:+.4f} brier={rP['brier_exceed']-rF['brier_exceed']:+.4f}")
print("learned theta (softplus>0):", {k: round(v, 3) for k, v in theta.items()})

run_id = "E9-" + datetime.now().strftime("%Y%m%d-%H%M%S")
verdict = ("physics HELPS" if (rP['pinball'] <= rF['pinball']-0.003 or rP['crps'] <= rF['crps']-0.0015)
           else ("physics NEUTRAL" if abs(rP['pinball']-rF['pinball']) < 0.003 else "physics WORSE"))
lines = [f"\n## Run {run_id}  (E9: physics-FORCED forecasting — sunlight-driven NOx-O3 ODE decoder)",
         f"- date: {datetime.now().isoformat(timespec='seconds')} | code: pipeline/10_qrpinn_e9.py | rank index; epochs={EPOCHS}; seed=0",
         f"- design: forecast median = sunlight-driven box-ODE rollout (J=photochemical index); spread head for quantiles; lam_c={LAM_C}",
         "", "| variant | pinball | CRPS | PICP80 | PICP90 | Brier(exc) |", "|---|---|---|---|---|---|",
         f"| physics-forced | {rP['pinball']:.4f} | {rP['crps']:.4f} | {rP['picp80']:.3f} | {rP['picp90']:.3f} | {rP['brier_exceed']:.4f} |",
         f"| no-physics (free) | {rF['pinball']:.4f} | {rF['crps']:.4f} | {rF['picp80']:.3f} | {rF['picp90']:.3f} | {rF['brier_exceed']:.4f} |",
         "", f"- delta (physics − free): pinball {rP['pinball']-rF['pinball']:+.4f}, CRPS {rP['crps']-rF['crps']:+.4f}, Brier {rP['brier_exceed']-rF['brier_exceed']:+.4f}",
         f"- learned rates (softplus>0): " + ", ".join(f"{k}={v:.3f}" for k, v in theta.items()),
         f"- **VERDICT: {verdict}** (threshold ~0.003 pinball / 0.0015 CRPS vs E8 noise)",
         f"- fig figs/e9_physics.png ; NOTE rank-index [0,1] metrics; future met used as forcing (forecast-met assumption)."]
with open(ROOT / "RESULTS_LOG.md", "a", encoding="utf-8") as f:
    f.write("\n".join(lines)+"\n")
print(f"\nWrote results_e9.json and appended {run_id} to RESULTS_LOG.md ({verdict})", flush=True)
