# -*- coding: utf-8 -*-
"""
08_qrpinn_e7.py  (E7)  -- Spatial transfer (Leave-One-Station-Out) + bootstrap CIs.
Method & citations: METHODS_transfer.md.

* STATION-AGNOSTIC QR-PINN (no station embedding) so it can predict at an unseen station.
* In-distribution reference: train all 9 stations 2014-15 -> test all 9 in 2016.
* LOSO: for each station k, train on the other 8 (2014-15) -> evaluate on station k in 2016.
* Leakage control: rank-index F_NOx/F_O3 and thr=Q30 fit on TRAINING stations' 2014-15 only.
* Uncertainty: moving-block bootstrap 95% CIs (block=24h) on pinball & CRPS (Politis & Romano 1994).

Rank index H = 0.5 F_NOx(NOx) + 0.5 F_O3(O3) (adopted, E5). Observed H only; no synthetic data.
Outputs: results_e7.json, figs/e7_loso.png, append RESULTS_LOG.md.
"""
import json, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from datetime import datetime

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
ART = ROOT / "artefacts"; FIGS = ROOT / "figs"; FIGS.mkdir(exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

L, HZ = 48, 24
TAUS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95], dtype=np.float32)
nT = len(TAUS)
EPOCHS, BATCH, LR = 8, 512, 1e-3
LAM_DATA, LAM_PHYS_MAX, ANNEAL = 0.3, 0.2, 5

d = np.load(ART / "qrpinn_data.npz", allow_pickle=True)
X = d["X"].astype(np.float32); PHYS = d["PHYS"].astype(np.float32)
COBS = d["COBS"].astype(np.float32); CMASK = d["CMASK"].astype(np.float32)
Hrob = d["H"].astype(np.float32); year = d["year"].astype(np.int32)
stations = [str(s) for s in d["stations"]]; S, T, F = X.shape
both = (CMASK[:, :, 0] == 1) & (CMASK[:, :, 1] == 1) & (~np.isnan(Hrob))
yr2d = np.broadcast_to(year[None, :], (S, T))
def Fcdf(s, v): return np.searchsorted(s, v, side="right") / len(s)

H_cur = np.full((S, T), np.nan, np.float32)   # set per fold
def fit_rank(train_st):
    rm = np.zeros(S, bool); rm[train_st] = True
    cell = both & (yr2d <= 2015) & rm[:, None]
    sN = np.sort(COBS[:, :, 0][cell]); sO = np.sort(COBS[:, :, 1][cell])
    H = np.full((S, T), np.nan, np.float32)
    H[both] = (0.5*Fcdf(sN, COBS[:, :, 0][both]) + 0.5*Fcdf(sO, COBS[:, :, 1][both])).astype(np.float32)
    return H, float(np.quantile(H[cell], 0.30))

def build_anchors(train_st, eval_st):
    tr, te = [], []
    trset, evset = set(train_st), set(eval_st)
    for si in range(S):
        Hs = H_cur[si]
        if si not in trset and si not in evset: continue
        for t in range(L-1, T-HZ):
            if np.isnan(Hs[t]): continue
            if not np.any(~np.isnan(Hs[t+1:t+1+HZ])): continue
            if si in trset and year[t+HZ] <= 2015: tr.append((si, t))
            if si in evset and year[t] == 2016: te.append((si, t))
    return np.array(tr, np.int32), np.array(te, np.int32)

class DS(torch.utils.data.Dataset):
    def __init__(self, idx): self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        si, t = int(self.idx[i, 0]), int(self.idx[i, 1])
        fut = H_cur[si, t+1:t+1+HZ].copy(); tm = (~np.isnan(fut)).astype(np.float32); fut[np.isnan(fut)] = 0.0
        return (torch.from_numpy(X[si, t-L+1:t+1, :]), torch.from_numpy(PHYS[si, t-L+1:t+1, :]),
                torch.from_numpy(COBS[si, t-L+1:t+1, :]), torch.from_numpy(CMASK[si, t-L+1:t+1, :]),
                torch.from_numpy(fut), torch.from_numpy(tm))

class QRPINN(nn.Module):           # STATION-AGNOSTIC (no embedding)
    def __init__(self, F, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(F, hidden, batch_first=True)
        self.base = nn.Linear(hidden, HZ); self.inc = nn.Linear(hidden, HZ*(nT-1))
        self.chead = nn.Linear(hidden, 2); self.praw = nn.Parameter(torch.zeros(7))
    def forward(self, x):
        out, (hn, cn) = self.lstm(x); z = hn[-1]
        base = self.base(z); inc = torch.nn.functional.softplus(self.inc(z)).view(-1, HZ, nT-1)
        q = torch.cat([base.unsqueeze(-1), base.unsqueeze(-1)+torch.cumsum(inc, -1)], -1)
        return q, self.chead(out)
def pos(p): return torch.nn.functional.softplus(p)
def physics_residual(chat, pw, praw):
    n, o = chat[:, :, 0], chat[:, :, 1]
    gv, gp, gr = torch.sigmoid(pw[:, :, 0]), torch.sigmoid(pw[:, :, 2]), torch.sigmoid(pw[:, :, 3])
    P_o, L_titr, D_o, W_o, E_n, D_n, W_n = [pos(praw[i]) for i in range(7)]
    np_, op_ = n[:, :-1], o[:, :-1]; gv, gp, gr = gv[:, :-1], gp[:, :-1], gr[:, :-1]
    o_hat = op_ + (P_o*gp - L_titr*torch.relu(np_)*torch.sigmoid(op_) - D_o*gv*op_ - W_o*gr*op_)
    n_hat = np_ + (E_n - D_n*gv*np_ - W_n*gr*np_)
    return ((o[:, 1:]-o_hat)**2).mean() + ((n[:, 1:]-n_hat)**2).mean()
def pinball_loss(q, y, tm):
    taus = torch.tensor(TAUS).view(1, 1, nT); err = y.unsqueeze(-1)-q
    return (torch.maximum(taus*err, (taus-1)*err).mean(-1)*tm).sum()/tm.sum().clamp(min=1)

def train(idx_tr):
    torch.manual_seed(0); model = QRPINN(F); opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl = torch.utils.data.DataLoader(DS(idx_tr), batch_size=BATCH, shuffle=True, num_workers=0)
    for ep in range(EPOCHS):
        model.train(); lam = LAM_PHYS_MAX*min(1.0, (ep+1)/ANNEAL)
        for x, pw, cobs, cmask, fut, tm in dl:
            opt.zero_grad(); q, chat = model(x)
            loss = pinball_loss(q, fut, tm) + LAM_DATA*(((chat-cobs)**2*cmask).sum()/cmask.sum().clamp(min=1))
            loss = loss + lam*physics_residual(chat, pw, model.praw)
            loss.backward(); opt.step()
    return model
def predict(model, idx):
    model.eval(); dl = torch.utils.data.DataLoader(DS(idx), batch_size=1024, shuffle=False, num_workers=0)
    Q, Y, M = [], [], []
    with torch.no_grad():
        for x, pw, cobs, cmask, fut, tm in dl:
            q, _ = model(x); Q.append(q.numpy()); Y.append(fut.numpy()); M.append(tm.numpy())
    return np.concatenate(Q), np.concatenate(Y), np.concatenate(M)

def cdf_at_thr(Q, thr):
    N, m = Q.shape; k = (Q < thr).sum(1); lo = np.clip(k-1, 0, m-1); hi = np.clip(k, 0, m-1); ar = np.arange(N)
    qlo, qhi, tlo, thi = Q[ar, lo], Q[ar, hi], TAUS[lo], TAUS[hi]
    gap = qhi-qlo; den = np.where(gap > 1e-9, gap, 1.0); frac = np.where(gap > 1e-9, (thr-qlo)/den, 0.0)
    cdf = tlo+frac*(thi-tlo); cdf = np.where(k == 0, TAUS[0], cdf); cdf = np.where(k == m, TAUS[-1], cdf)
    return np.clip(cdf, 0, 1)
def metrics(Q, Y, M, thr):
    taus = TAUS.reshape(1, 1, nT); err = Y[..., None]-Q; pb = np.maximum(taus*err, (taus-1)*err); m3 = M[..., None]
    pin = float((pb*m3).sum()/m3.sum()); crps = float(2*((pb*m3).sum((0, 1))/m3.sum((0, 1))).mean())
    def cov(a, b):
        lo, hi = Q[:, :, a], Q[:, :, b]; return float((((Y >= lo)&(Y <= hi)).astype(float)*M).sum()/M.sum())
    Qf, Yf, Mf = Q.reshape(-1, nT), Y.reshape(-1), M.reshape(-1)
    pexc = 1.0-cdf_at_thr(Qf, thr); ind = (Yf >= thr).astype(float); sel = Mf > 0
    brier = float((((pexc-ind)**2)[sel]).mean())
    return {"pinball": pin, "crps": crps, "picp80": cov(1, 7), "picp90": cov(0, 8),
            "brier_exceed": brier, "n": int(M.sum())}
def anchor_scores(Q, Y, M):
    taus = TAUS.reshape(1, 1, nT); err = Y[..., None]-Q; pb = np.maximum(taus*err, (taus-1)*err)
    mh = M.sum(1); ok = mh > 0
    # per-anchor scores on the SAME scale as the pooled metrics() (pinball summed over tau, CRPS averaged)
    pin_i = (pb*M[..., None]).sum((1, 2))[ok]/mh[ok]
    crps_i = 2*((pb*M[..., None]).sum(1)[ok]/mh[ok][:, None]).mean(1)
    return pin_i, crps_i
def block_ci(scores, block=24, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(scores); nb = int(np.ceil(n/block)); means = np.empty(B)
    maxstart = max(1, n-block)
    for b in range(B):
        starts = rng.integers(0, maxstart, size=nb)
        idxs = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        means[b] = scores[idxs].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

# ============ in-distribution reference (all 9, station-agnostic) ============
all_st = list(range(S))
print("== reference: all-9 station-agnostic ==", flush=True)
H_cur, thr0 = fit_rank(all_st)
itr, ite = build_anchors(all_st, all_st)
t0 = time.time(); ref_model = train(itr)
Qr, Yr, Mr = predict(ref_model, ite); ref = metrics(Qr, Yr, Mr, thr0)
pin_i, crps_i = anchor_scores(Qr, Yr, Mr)
ref["pinball_CI95"] = block_ci(pin_i); ref["crps_CI95"] = block_ci(crps_i)
print(f"  reference {ref} ({time.time()-t0:.0f}s)", flush=True)

# ============ LOSO ============
loso = {}; loso_pin_all, loso_crps_all = [], []
for k in range(S):
    tr_st = [s for s in range(S) if s != k]
    H_cur, thrk = fit_rank(tr_st)
    itr, ite = build_anchors(tr_st, [k])
    t0 = time.time(); m = train(itr); Q, Y, M = predict(m, ite); met = metrics(Q, Y, M, thrk)
    pi, ci = anchor_scores(Q, Y, M); loso_pin_all.append(pi); loso_crps_all.append(ci)
    loso[stations[k]] = met
    print(f"  LOSO holdout={stations[k]:12s} pinball={met['pinball']:.4f} crps={met['crps']:.4f} "
          f"picp90={met['picp90']:.3f} brier={met['brier_exceed']:.4f} n={met['n']} ({time.time()-t0:.0f}s)", flush=True)

vals = lambda key: np.array([loso[s][key] for s in stations])
loso_summary = {key: {"mean": float(vals(key).mean()), "std": float(vals(key).std())}
                for key in ["pinball", "crps", "picp80", "picp90", "brier_exceed"]}
pin_pool = np.concatenate(loso_pin_all); crps_pool = np.concatenate(loso_crps_all)
loso_summary["pinball_pooled_CI95"] = block_ci(pin_pool); loso_summary["crps_pooled_CI95"] = block_ci(crps_pool)

R = {"config": {"epochs": EPOCHS, "taus": TAUS.tolist(), "station_agnostic": True, "thr_ref": thr0},
     "reference_in_distribution": ref, "loso_per_station": loso, "loso_summary": loso_summary}
json.dump(R, open(ROOT / "results_e7.json", "w"), indent=2)

# ============ figure ============
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    pv = vals("pinball"); order = np.argsort(pv)
    ax[0].bar(range(S), pv[order], color="tab:purple")
    ax[0].axhline(ref["pinball"], ls="--", c="k", label=f"in-distribution ref={ref['pinball']:.3f}")
    ax[0].set_xticks(range(S)); ax[0].set_xticklabels([stations[i] for i in order], rotation=60, ha="right", fontsize=7)
    ax[0].set_ylabel("pinball (held-out 2016)"); ax[0].set_title("LOSO transfer skill by station"); ax[0].legend(fontsize=8)
    cv = vals("picp90")
    ax[1].bar(range(S), cv[order], color="tab:green"); ax[1].axhline(0.90, ls="--", c="k", label="nominal 0.90")
    ax[1].set_xticks(range(S)); ax[1].set_xticklabels([stations[i] for i in order], rotation=60, ha="right", fontsize=7)
    ax[1].set_ylabel("PICP90"); ax[1].set_title("LOSO 90% coverage by station"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIGS / "e7_loso.png", dpi=130); plt.close(); print("saved figs/e7_loso.png", flush=True)
except Exception as e:
    print("figure skipped:", e, flush=True)

# ============ console + log ============
print("\n=============== E7 SPATIAL TRANSFER ===============")
print(f"in-distribution ref : pinball={ref['pinball']:.4f} {ref['pinball_CI95']}  crps={ref['crps']:.4f} {ref['crps_CI95']}  picp90={ref['picp90']:.3f}")
print(f"LOSO mean+/-std     : pinball={loso_summary['pinball']['mean']:.4f}+/-{loso_summary['pinball']['std']:.4f}  "
      f"crps={loso_summary['crps']['mean']:.4f}+/-{loso_summary['crps']['std']:.4f}  picp90={loso_summary['picp90']['mean']:.3f}")
print(f"LOSO pooled CI95    : pinball {loso_summary['pinball_pooled_CI95']}  crps {loso_summary['crps_pooled_CI95']}")
print("per-station pinball  :", {s: round(loso[s]['pinball'], 4) for s in stations})

run_id = "E7-" + datetime.now().strftime("%Y%m%d-%H%M%S")
lines = [f"\n## Run {run_id}  (E7: LOSO spatial transfer + block-bootstrap CIs)",
         f"- date: {datetime.now().isoformat(timespec='seconds')} | code: pipeline/08_qrpinn_e7.py | method: METHODS_transfer.md",
         f"- model: STATION-AGNOSTIC rank-index QR-PINN (physics on); epochs={EPOCHS}; seed=0; rank index H",
         f"- bootstrap: moving-block (block=24h, B=1000); 95% CI", "",
         "| setting | pinball [95% CI] | CRPS [95% CI] | PICP80 | PICP90 | Brier(exc) |",
         "|---|---|---|---|---|---|",
         f"| in-distribution (all-9) | {ref['pinball']:.4f} [{ref['pinball_CI95'][0]:.4f},{ref['pinball_CI95'][1]:.4f}] | "
         f"{ref['crps']:.4f} [{ref['crps_CI95'][0]:.4f},{ref['crps_CI95'][1]:.4f}] | {ref['picp80']:.3f} | {ref['picp90']:.3f} | {ref['brier_exceed']:.4f} |",
         f"| LOSO mean±std (9 folds) | {loso_summary['pinball']['mean']:.4f} ± {loso_summary['pinball']['std']:.4f} | "
         f"{loso_summary['crps']['mean']:.4f} ± {loso_summary['crps']['std']:.4f} | {loso_summary['picp80']['mean']:.3f} | "
         f"{loso_summary['picp90']['mean']:.3f} | {loso_summary['brier_exceed']['mean']:.4f} |",
         f"| LOSO pooled (CI) | [{loso_summary['pinball_pooled_CI95'][0]:.4f},{loso_summary['pinball_pooled_CI95'][1]:.4f}] | "
         f"[{loso_summary['crps_pooled_CI95'][0]:.4f},{loso_summary['crps_pooled_CI95'][1]:.4f}] | | | |",
         "", "| held-out station | pinball | CRPS | PICP90 | Brier(exc) | n |",
         "|---|---|---|---|---|---|"]
for s in stations:
    m = loso[s]
    lines.append(f"| {s} | {m['pinball']:.4f} | {m['crps']:.4f} | {m['picp90']:.3f} | {m['brier_exceed']:.4f} | {m['n']} |")
lines += ["", f"- transfer gap (LOSO mean − in-dist) pinball = {loso_summary['pinball']['mean']-ref['pinball']:+.4f}",
          f"- fig figs/e7_loso.png ; NOTE rank-index metrics on [0,1], not comparable across index defs; "
          f"spatial-CV caveats in METHODS_transfer.md sec.5."]
with open(ROOT / "RESULTS_LOG.md", "a", encoding="utf-8") as f:
    f.write("\n".join(lines)+"\n")
print(f"\nWrote results_e7.json and appended {run_id} to RESULTS_LOG.md", flush=True)
