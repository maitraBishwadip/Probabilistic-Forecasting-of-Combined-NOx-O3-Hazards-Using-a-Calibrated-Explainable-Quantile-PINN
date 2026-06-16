# -*- coding: utf-8 -*-
"""
04_qrpinn_e3.py  (E3)
Physics ablation + combined-extreme predictive distribution + pollutant-dominance diagnostic.

Variants (identical seed/config) to isolate the physics term:
  QRNN      : pinball only (no aux, no physics)
  DATA-ONLY : pinball + aux NOx/O3 concentration fit (lam_phys = 0)
  FULL      : DATA-ONLY + coupled NOx-O3 box-model physics residual
  => (FULL - DATA-ONLY) = pure value of the physics residual.

Also: persists the FULL model's predictive distribution of the combined index H (the
"both-pollutants-together" extreme distribution), and a dominance diagnostic that states
whether one pollutant overpowers the combined index / the prediction.

Outputs: results_e3.json, artefacts/e3_full_test_quantiles.npz, figs/e3_*.png,
         appends a block to RESULTS_LOG.md.   (No synthetic data; observed targets only.)
"""
import json, time, numpy as np, torch, torch.nn as nn, pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
ART = ROOT / "artefacts"; FIGS = ROOT / "figs"; FIGS.mkdir(exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

# ---------------- config (matches 03) ----------------
L, HZ = 48, 24
TAUS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95], dtype=np.float32)
nT = len(TAUS); TAU05 = int(np.where(np.isclose(TAUS, 0.5))[0][0])
EPOCHS, BATCH, LR = 12, 512, 1e-3
LAM_DATA, LAM_PHYS_MAX, ANNEAL = 0.3, 0.2, 6

d = np.load(ART / "qrpinn_data.npz", allow_pickle=True)
X = d["X"].astype(np.float32); PHYS = d["PHYS"].astype(np.float32)
COBS = d["COBS"].astype(np.float32); CMASK = d["CMASK"].astype(np.float32)
H = d["H"].astype(np.float32); year = d["year"].astype(np.int32); thr = float(d["thr"][0])
stations = [str(s) for s in d["stations"]]; S, T, F = X.shape
print(f"Loaded X{X.shape} F={F} thr={thr:.4f}")

# ---------------- anchors (no 2015->2016 target bleed) ----------------
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
print(f"anchors train={len(idx_tr)} test={len(idx_te)}")

class DS(torch.utils.data.Dataset):
    def __init__(self, idx): self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        si, t = int(self.idx[i, 0]), int(self.idx[i, 1])
        fut = H[si, t+1:t+1+HZ].copy(); tmask = (~np.isnan(fut)).astype(np.float32); fut[np.isnan(fut)] = 0.0
        return (torch.from_numpy(X[si, t-L+1:t+1, :]), torch.tensor(si),
                torch.from_numpy(PHYS[si, t-L+1:t+1, :]), torch.from_numpy(COBS[si, t-L+1:t+1, :]),
                torch.from_numpy(CMASK[si, t-L+1:t+1, :]), torch.from_numpy(fut), torch.from_numpy(tmask))

class QRPINN(nn.Module):
    def __init__(self, F, n_station, hidden=64, emb=8, physics=True, aux=True):
        super().__init__(); self.physics, self.aux = physics, aux
        self.emb = nn.Embedding(n_station, emb); self.lstm = nn.LSTM(F, hidden, batch_first=True)
        self.base = nn.Linear(hidden+emb, HZ); self.inc = nn.Linear(hidden+emb, HZ*(nT-1))
        if aux: self.chead = nn.Linear(hidden, 2)
        self.praw = nn.Parameter(torch.zeros(7))
    def forward(self, x, si):
        out, (hn, cn) = self.lstm(x); z = hn[-1]; trunk = torch.cat([z, self.emb(si)], 1)
        base = self.base(trunk); inc = torch.nn.functional.softplus(self.inc(trunk)).view(-1, HZ, nT-1)
        q = torch.cat([base.unsqueeze(-1), base.unsqueeze(-1)+torch.cumsum(inc, -1)], -1)
        return q, (self.chead(out) if self.aux else None)

def pos(p): return torch.nn.functional.softplus(p)
def physics_residual(chat, pw, praw):
    n, o = chat[:, :, 0], chat[:, :, 1]
    gv, gp, gr = torch.sigmoid(pw[:, :, 0]), torch.sigmoid(pw[:, :, 2]), torch.sigmoid(pw[:, :, 3])
    P_o, L_titr, D_o, W_o, E_n, D_n, W_n = [pos(praw[i]) for i in range(7)]
    np_, op_ = n[:, :-1], o[:, :-1]; gv, gp, gr = gv[:, :-1], gp[:, :-1], gr[:, :-1]
    o_hat = op_ + (P_o*gp - L_titr*torch.relu(np_)*torch.sigmoid(op_) - D_o*gv*op_ - W_o*gr*op_)
    n_hat = np_ + (E_n - D_n*gv*np_ - W_n*gr*np_)
    return ((o[:, 1:]-o_hat)**2).mean() + ((n[:, 1:]-n_hat)**2).mean()
def pinball_loss(q, y, tmask):
    taus = torch.tensor(TAUS).view(1, 1, nT); err = y.unsqueeze(-1)-q
    l = torch.maximum(taus*err, (taus-1)*err).mean(-1)*tmask
    return l.sum()/tmask.sum().clamp(min=1)

def train_model(name, physics, aux):
    torch.manual_seed(0); model = QRPINN(F, S, physics=physics, aux=aux)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl = torch.utils.data.DataLoader(DS(idx_tr), batch_size=BATCH, shuffle=True, num_workers=0)
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = nb = 0
        lam = (LAM_PHYS_MAX*min(1.0, (ep+1)/ANNEAL)) if physics else 0.0
        for x, si, pw, cobs, cmask, fut, tmask in dl:
            opt.zero_grad(); q, chat = model(x, si); loss = pinball_loss(q, fut, tmask)
            if aux: loss = loss + LAM_DATA*(((chat-cobs)**2*cmask).sum()/cmask.sum().clamp(min=1))
            if physics: loss = loss + lam*physics_residual(chat, pw, model.praw)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [{name}] ep {ep+1}/{EPOCHS} loss={tot/nb:.4f} lam={lam:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return model

def cdf_at(qs, thr):
    N, m = qs.shape; k = np.sum(qs < thr, 1); lo = np.clip(k-1, 0, m-1); hi = np.clip(k, 0, m-1); ar = np.arange(N)
    qlo, qhi, tlo, thi = qs[ar, lo], qs[ar, hi], TAUS[lo], TAUS[hi]
    gap = qhi-qlo; den = np.where(gap > 1e-9, gap, 1.0); frac = np.where(gap > 1e-9, (thr-qlo)/den, 0.0)
    cdf = tlo+frac*(thi-tlo); cdf = np.where(k == 0, TAUS[0], cdf); cdf = np.where(k == m, TAUS[-1], cdf)
    return np.clip(cdf, 0, 1)
def collect_preds(model):
    model.eval(); dl = torch.utils.data.DataLoader(DS(idx_te), batch_size=1024, shuffle=False, num_workers=0)
    Q, Y, M = [], [], []
    with torch.no_grad():
        for x, si, pw, cobs, cmask, fut, tmask in dl:
            q, _ = model(x, si); Q.append(q.numpy()); Y.append(fut.numpy()); M.append(tmask.numpy())
    return np.concatenate(Q), np.concatenate(Y), np.concatenate(M)
def metrics(Q, Y, M):
    taus = TAUS.reshape(1, 1, nT); err = Y[..., None]-Q
    pb = np.maximum(taus*err, (taus-1)*err); m3 = M[..., None]
    pin = float((pb*m3).sum()/m3.sum()); pin_tau = (pb*m3).sum((0, 1))/m3.sum((0, 1))
    def cov(a, b):
        lo, hi = Q[:, :, a], Q[:, :, b]; ins = ((Y >= lo)&(Y <= hi)).astype(float)*M
        return float(ins.sum()/M.sum()), float(((hi-lo)*M).sum()/M.sum())
    p80, w80 = cov(1, 7); p90, w90 = cov(0, 8)
    perh = ((pb*m3).sum((0, 2))/(M.sum(0)*nT)).tolist()
    Qf, Yf, Mf = Q.reshape(-1, nT), Y.reshape(-1), M.reshape(-1)
    pexc = 1.0-cdf_at(Qf, thr); ind = (Yf >= thr).astype(float); sel = Mf > 0
    brier = float((((pexc-ind)**2)[sel]).mean())
    Q1, Y1, M1 = Q[:, 0, :], Y[:, 0], M[:, 0] > 0
    brier1 = float((((1-cdf_at(Q1, thr)-(Y1 >= thr).astype(float))**2)[M1]).mean())
    return {"pinball": pin, "pinball_mean": pin/nT, "crps": float(2*pin_tau.mean()),
            "picp80": p80, "width80": w80, "picp90": p90, "width90": w90,
            "brier_exceed": brier, "brier_exceed_h1": brier1,
            "test_base_rate_extreme": float(ind[sel].mean()), "pinball_per_horizon": perh}

# ============ TRAIN 3 VARIANTS ============
print("\n== QRNN ==");      m_qrnn = train_model("QRNN", False, False)
print("\n== DATA-ONLY =="); m_data = train_model("DATA", False, True)
print("\n== FULL ==");      m_full = train_model("FULL", True, True)
R = {"config": {"L": L, "HZ": HZ, "epochs": EPOCHS, "lam_phys_max": LAM_PHYS_MAX, "thr": thr,
                "n_train": int(len(idx_tr)), "n_test": int(len(idx_te))}, "models": {}}
Qn, Yn, Mn = collect_preds(m_qrnn); R["models"]["QRNN"] = metrics(Qn, Yn, Mn)
Qd, Yd, Md = collect_preds(m_data); R["models"]["DATA-ONLY"] = metrics(Qd, Yd, Md)
Qp, Yp, Mp = collect_preds(m_full); R["models"]["FULL"] = metrics(Qp, Yp, Mp)
R["models"]["FULL"]["learned_physics_params"] = {k: float(pos(m_full.praw[i]).item())
    for i, k in enumerate(["P_o", "L_titr", "D_o", "W_o", "E_n", "D_n", "W_n"])}

# ============ COMBINED-EXTREME PREDICTIVE DISTRIBUTION (FULL) ============
pexc_full = 1.0 - cdf_at(Qp.reshape(-1, nT), thr)
np.savez_compressed(ART / "e3_full_test_quantiles.npz", Q=Qp, H_true=Yp, mask=Mp,
                    taus=TAUS, thr=thr, p_exceed=pexc_full.reshape(Qp.shape[0], HZ))
print("saved combined predictive distribution -> artefacts/e3_full_test_quantiles.npz")

# ============ POLLUTANT-DOMINANCE DIAGNOSTIC ============
# index-level: H = 0.5 zN + 0.5 zO  (zN,zO are robust-scaled NOx,O3 = COBS where both observed)
mb = (CMASK[:, :, 0] == 1) & (CMASK[:, :, 1] == 1) & (~np.isnan(H))
zN, zO, Hv = COBS[:, :, 0][mb], COBS[:, :, 1][mb], H[mb]
varH = np.var(Hv); shareN = np.var(0.5*zN)/varH; shareO = np.var(0.5*zO)/varH
shareCov = 1.0 - shareN - shareO
corrHN = float(np.corrcoef(Hv, zN)[0, 1]); corrHO = float(np.corrcoef(Hv, zO)[0, 1])
corrNO = float(np.corrcoef(zN, zO)[0, 1])
ext = Hv >= thr
frac_N_bigger = float(np.mean(zN[ext] > zO[ext]))
excessN = float(np.mean(0.5*zN[ext])); excessO = float(np.mean(0.5*zO[ext]))
# prediction-level: corr of predicted median H (h=1) with observed zN,zO at the target hour
predmed = Qp[:, 0, TAU05]; tgtN = np.full(len(idx_te), np.nan); tgtO = np.full(len(idx_te), np.nan)
for i in range(len(idx_te)):
    si, t = int(idx_te[i, 0]), int(idx_te[i, 1]); tt = t+1
    if CMASK[si, tt, 0] == 1 and CMASK[si, tt, 1] == 1:
        tgtN[i] = COBS[si, tt, 0]; tgtO[i] = COBS[si, tt, 1]
ok = (~np.isnan(tgtN)) & (Mp[:, 0] > 0)
pcorrN = float(np.corrcoef(predmed[ok], tgtN[ok])[0, 1]); pcorrO = float(np.corrcoef(predmed[ok], tgtO[ok])[0, 1])
dom = "O3" if shareO > shareN else "NOx"
R["dominance"] = {"std_zNOx": float(np.std(zN)), "std_zO3": float(np.std(zO)),
                  "var_share_NOx": float(shareN), "var_share_O3": float(shareO), "var_share_cov": float(shareCov),
                  "corr_H_NOx": corrHN, "corr_H_O3": corrHO, "corr_NOx_O3": corrNO,
                  "extreme_frac_NOx_larger": frac_N_bigger,
                  "extreme_mean_contrib_NOx": excessN, "extreme_mean_contrib_O3": excessO,
                  "pred_corr_NOx": pcorrN, "pred_corr_O3": pcorrO, "dominant_pollutant": dom}
json.dump(R, open(ROOT / "results_e3.json", "w"), indent=2)

# ============ FIGURES (best-effort) ============
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # (1) example combined predictive PDFs at h=1 (low / mid / high hazard)
    med = Qp[:, 0, TAU05]; order = np.argsort(med); pick = [order[len(order)//10], order[len(order)//2], order[-len(order)//10]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for idx_, lab in zip(pick, ["low", "mid", "high"]):
        q = Qp[idx_, 0, :]; ax[0].plot(q, TAUS, marker="o", label=f"{lab} (P(ext)={1-cdf_at(q[None,:],thr)[0]:.2f})")
    ax[0].axvline(thr, ls="--", c="k", lw=1); ax[0].set_title("Combined index H — predictive CDF (h=1)")
    ax[0].set_xlabel("H"); ax[0].set_ylabel("cumulative prob"); ax[0].legend(fontsize=8)
    ax[1].bar(["NOx", "O3", "cov"], [shareN, shareO, shareCov], color=["tab:blue", "tab:orange", "gray"])
    ax[1].set_title(f"Variance share of H  (dominant = {dom})"); ax[1].set_ylabel("share of Var(H)")
    plt.tight_layout(); plt.savefig(FIGS / "e3_combined_pdf_dominance.png", dpi=130); plt.close()
    print("saved figs/e3_combined_pdf_dominance.png")
except Exception as e:
    print("figure skipped:", e)

# ============ CONSOLE + LOG ============
print("\n=============== E3 TEST 2016 ===============")
print(f"{'model':10s} {'pinball':>8s} {'pin_mean':>8s} {'CRPS':>7s} {'PICP80':>7s} {'PICP90':>7s} {'Brier':>7s} {'Brier@1':>8s}")
for k, v in R["models"].items():
    print(f"{k:10s} {v['pinball']:8.4f} {v['pinball_mean']:8.4f} {v['crps']:7.4f} {v['picp80']:7.3f} {v['picp90']:7.3f} {v['brier_exceed']:7.4f} {v['brier_exceed_h1']:8.4f}")
dd = R["dominance"]
print("\n--- POLLUTANT DOMINANCE ---")
print(f"std(zNOx)={dd['std_zNOx']:.2f}  std(zO3)={dd['std_zO3']:.2f}")
print(f"Var(H) share: NOx={dd['var_share_NOx']*100:.1f}%  O3={dd['var_share_O3']*100:.1f}%  cov={dd['var_share_cov']*100:.1f}%")
print(f"corr(H,NOx)={dd['corr_H_NOx']:.3f}  corr(H,O3)={dd['corr_H_O3']:.3f}  corr(NOx,O3)={dd['corr_NOx_O3']:.3f}")
print(f"among extreme hours: NOx larger in {dd['extreme_frac_NOx_larger']*100:.1f}%; mean contrib NOx={dd['extreme_mean_contrib_NOx']:.3f} vs O3={dd['extreme_mean_contrib_O3']:.3f}")
print(f"prediction tracks: corr(predH,NOx)={dd['pred_corr_NOx']:.3f} vs corr(predH,O3)={dd['pred_corr_O3']:.3f}")
print(f">>> DOMINANT POLLUTANT (overpowers combined index): {dd['dominant_pollutant']}")

run_id = "E3-" + datetime.now().strftime("%Y%m%d-%H%M%S")
lines = [f"\n## Run {run_id}  (E3: physics ablation + dominance)",
         f"- date: {datetime.now().isoformat(timespec='seconds')} | code: pipeline/04_qrpinn_e3.py",
         f"- config: L={L}, HZ={HZ}, epochs={EPOCHS}, lam_phys_max={LAM_PHYS_MAX}, thr={thr:.4f}, seed=0",
         f"- split: train 2014-15 ({len(idx_tr)}) / test 2016 ({len(idx_te)})", "",
         "| model | pinball | pin_mean | CRPS | PICP80 | PICP90 | width80 | Brier | Brier@1 |",
         "|---|---|---|---|---|---|---|---|---|"]
for k, v in R["models"].items():
    lines.append(f"| {k} | {v['pinball']:.4f} | {v['pinball_mean']:.4f} | {v['crps']:.4f} | {v['picp80']:.3f} | {v['picp90']:.3f} | {v['width80']:.3f} | {v['brier_exceed']:.4f} | {v['brier_exceed_h1']:.4f} |")
lp = R["models"]["FULL"]["learned_physics_params"]
lines += ["", f"- physics value = FULL - DATA-ONLY: pinball {R['models']['DATA-ONLY']['pinball']-R['models']['FULL']['pinball']:+.4f}, "
          f"CRPS {R['models']['DATA-ONLY']['crps']-R['models']['FULL']['crps']:+.4f}",
          f"- FULL learned physics params: " + ", ".join(f"{k}={v:.3f}" for k, v in lp.items()),
          "", f"- DOMINANCE: Var(H) share NOx={dd['var_share_NOx']*100:.1f}% / O3={dd['var_share_O3']*100:.1f}% / cov={dd['var_share_cov']*100:.1f}%; "
          f"corr(H,NOx)={dd['corr_H_NOx']:.3f} corr(H,O3)={dd['corr_H_O3']:.3f}; "
          f"pred corr NOx={dd['pred_corr_NOx']:.3f}/O3={dd['pred_corr_O3']:.3f}; "
          f"**dominant = {dd['dominant_pollutant']}** (overpowers the combined index).",
          f"- combined predictive distribution saved: artefacts/e3_full_test_quantiles.npz; fig: figs/e3_combined_pdf_dominance.png"]
with open(ROOT / "RESULTS_LOG.md", "a", encoding="utf-8") as f:
    f.write("\n".join(lines)+"\n")
print(f"\nWrote results_e3.json and appended {run_id} to RESULTS_LOG.md")
