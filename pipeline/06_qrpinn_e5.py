# -*- coding: utf-8 -*-
"""
06_qrpinn_e5.py  (E5)
Rebalance the combined index so neither pollutant overpowers it.

E3 found O3 overpowers the robust index H=0.5 rs(NOx)+0.5 rs(O3):
  Var(H) share O3 73% / NOx 44%; prediction corr O3 0.90 / NOx 0.12.

Fix: RANK / quantile-uniform combined index
  H_rank = 0.5 * F_NOx(NOx) + 0.5 * F_O3(O3)
  where F_* is the TRAIN empirical CDF (pooled), so each marginal is uniform[0,1]
  and cannot dominate by scale or tail shape. (Monotone transform => computable from the
  robust-scaled COBS already saved; no raw re-read, no synthetic data.)

Steps: compare index-level dominance (robust vs rank) -> retrain FULL on H_rank ->
       report metrics + prediction-level dominance (does the model now use NOx?).
Outputs: results_e5.json, artefacts/qrpinn_full_rank.pt, figs/e5_dominance.png, append RESULTS_LOG.md
"""
import json, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from datetime import datetime

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
ART = ROOT / "artefacts"; FIGS = ROOT / "figs"; FIGS.mkdir(exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

L, HZ = 48, 24
TAUS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95], dtype=np.float32)
nT = len(TAUS); TAU05 = int(np.where(np.isclose(TAUS, 0.5))[0][0])
EPOCHS, BATCH, LR = 12, 512, 1e-3
LAM_DATA, LAM_PHYS_MAX, ANNEAL = 0.3, 0.2, 6

d = np.load(ART / "qrpinn_data.npz", allow_pickle=True)
X = d["X"].astype(np.float32); PHYS = d["PHYS"].astype(np.float32)
COBS = d["COBS"].astype(np.float32); CMASK = d["CMASK"].astype(np.float32)
Hrob = d["H"].astype(np.float32); year = d["year"].astype(np.int32)
S, T, F = X.shape

# ---------- both-present cells (where a combined index is defined) ----------
both = (CMASK[:, :, 0] == 1) & (CMASK[:, :, 1] == 1) & (~np.isnan(Hrob))
yr2d = np.broadcast_to(year[None, :], (S, T))
tr_both = both & (yr2d <= 2015)

zN_all, zO_all = COBS[:, :, 0], COBS[:, :, 1]          # robust-scaled NOx,O3 (monotone in raw)

# ---------- RANK index (train empirical CDF, pooled) ----------
sN = np.sort(zN_all[tr_both]); sO = np.sort(zO_all[tr_both]); nTr = len(sN)
def Fcdf(sorted_arr, v): return np.searchsorted(sorted_arr, v, side="right") / len(sorted_arr)
Hrank = np.full((S, T), np.nan, dtype=np.float32)
rN_all = Fcdf(sN, zN_all[both]); rO_all = Fcdf(sO, zO_all[both])
Hrank[both] = (0.5*rN_all + 0.5*rO_all).astype(np.float32)
thr_rank = float(np.quantile(Hrank[tr_both], 0.30))
print(f"thr_rank(Q30 train)={thr_rank:.4f}  train frac extreme={(Hrank[tr_both]>=thr_rank).mean():.3f}", flush=True)

# ---------- index-level dominance helper ----------
def index_dominance(Harr, zN, zO, partN, partO, mask):
    Hv, a, b = Harr[mask], partN[mask], partO[mask]
    varH = np.var(Hv); shN = np.var(0.5*a)/varH; shO = np.var(0.5*b)/varH
    return {"var_share_NOx": float(shN), "var_share_O3": float(shO), "var_share_cov": float(1-shN-shO),
            "corr_H_NOx": float(np.corrcoef(Hv, a)[0, 1]), "corr_H_O3": float(np.corrcoef(Hv, b)[0, 1])}
# robust index parts are zN,zO ; rank index parts are ranks
rN_full = np.full((S, T), np.nan, np.float32); rO_full = np.full((S, T), np.nan, np.float32)
rN_full[both] = rN_all; rO_full[both] = rO_all
dom_robust = index_dominance(Hrob, zN_all, zO_all, zN_all, zO_all, both)
dom_rank = index_dominance(Hrank, zN_all, zO_all, rN_full, rO_full, both)
print("robust index dominance:", {k: round(v, 3) for k, v in dom_robust.items()}, flush=True)
print("rank   index dominance:", {k: round(v, 3) for k, v in dom_rank.items()}, flush=True)

# ---------- use rank target ----------
H = Hrank; thr = thr_rank

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
print(f"anchors train={len(idx_tr)} test={len(idx_te)}", flush=True)

class DS(torch.utils.data.Dataset):
    def __init__(self, idx): self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        si, t = int(self.idx[i, 0]), int(self.idx[i, 1])
        fut = H[si, t+1:t+1+HZ].copy(); tm = (~np.isnan(fut)).astype(np.float32); fut[np.isnan(fut)] = 0.0
        return (torch.from_numpy(X[si, t-L+1:t+1, :]), torch.tensor(si),
                torch.from_numpy(PHYS[si, t-L+1:t+1, :]), torch.from_numpy(COBS[si, t-L+1:t+1, :]),
                torch.from_numpy(CMASK[si, t-L+1:t+1, :]), torch.from_numpy(fut), torch.from_numpy(tm))

class QRPINN(nn.Module):
    def __init__(self, F, n_station, hidden=64, emb=8):
        super().__init__()
        self.emb = nn.Embedding(n_station, emb); self.lstm = nn.LSTM(F, hidden, batch_first=True)
        self.base = nn.Linear(hidden+emb, HZ); self.inc = nn.Linear(hidden+emb, HZ*(nT-1))
        self.chead = nn.Linear(hidden, 2); self.praw = nn.Parameter(torch.zeros(7))
    def forward(self, x, si):
        out, (hn, cn) = self.lstm(x); trunk = torch.cat([hn[-1], self.emb(si)], 1)
        base = self.base(trunk); inc = torch.nn.functional.softplus(self.inc(trunk)).view(-1, HZ, nT-1)
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

def train_full():
    model = QRPINN(F, S); opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl = torch.utils.data.DataLoader(DS(idx_tr), batch_size=BATCH, shuffle=True, num_workers=0)
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = nb = 0; lam = LAM_PHYS_MAX*min(1.0, (ep+1)/ANNEAL)
        for x, si, pw, cobs, cmask, fut, tm in dl:
            opt.zero_grad(); q, chat = model(x, si)
            loss = pinball_loss(q, fut, tm) + LAM_DATA*(((chat-cobs)**2*cmask).sum()/cmask.sum().clamp(min=1))
            loss = loss + lam*physics_residual(chat, pw, model.praw)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [FULL-rank] ep {ep+1}/{EPOCHS} loss={tot/nb:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return model
def predict(model, idx):
    model.eval(); dl = torch.utils.data.DataLoader(DS(idx), batch_size=1024, shuffle=False, num_workers=0)
    Q, Y, M = [], [], []
    with torch.no_grad():
        for x, si, pw, cobs, cmask, fut, tm in dl:
            q, _ = model(x, si); Q.append(q.numpy()); Y.append(fut.numpy()); M.append(tm.numpy())
    return np.concatenate(Q), np.concatenate(Y), np.concatenate(M)
def cdf_at_thr(Q, thr):
    N, m = Q.shape; k = (Q < thr).sum(1); lo = np.clip(k-1, 0, m-1); hi = np.clip(k, 0, m-1); ar = np.arange(N)
    qlo, qhi, tlo, thi = Q[ar, lo], Q[ar, hi], TAUS[lo], TAUS[hi]
    gap = qhi-qlo; den = np.where(gap > 1e-9, gap, 1.0); frac = np.where(gap > 1e-9, (thr-qlo)/den, 0.0)
    cdf = tlo+frac*(thi-tlo); cdf = np.where(k == 0, TAUS[0], cdf); cdf = np.where(k == m, TAUS[-1], cdf)
    return np.clip(cdf, 0, 1)
def metrics(Q, Y, M):
    taus = TAUS.reshape(1, 1, nT); err = Y[..., None]-Q
    pb = np.maximum(taus*err, (taus-1)*err); m3 = M[..., None]
    pin = float((pb*m3).sum()/m3.sum())
    def cov(a, b):
        lo, hi = Q[:, :, a], Q[:, :, b]; ins = ((Y >= lo)&(Y <= hi)).astype(float)*M
        return float(ins.sum()/M.sum()), float(((hi-lo)*M).sum()/M.sum())
    p80, w80 = cov(1, 7); p90, w90 = cov(0, 8)
    Qf, Yf, Mf = Q.reshape(-1, nT), Y.reshape(-1), M.reshape(-1)
    pexc = 1.0-cdf_at_thr(Qf, thr); ind = (Yf >= thr).astype(float); sel = Mf > 0
    brier = float((((pexc-ind)**2)[sel]).mean())
    return {"pinball": pin, "crps": float(2*((pb*m3).sum((0, 1))/m3.sum((0, 1))).mean()),
            "picp80": p80, "width80": w80, "picp90": p90, "width90": w90,
            "brier_exceed": brier, "test_base_rate_extreme": float(ind[sel].mean())}

print("\n== train FULL on rank index ==", flush=True); model = train_full()
torch.save(model.state_dict(), ART / "qrpinn_full_rank.pt"); print("saved checkpoint", flush=True)
Qp, Yp, Mp = predict(model, idx_te); met = metrics(Qp, Yp, Mp)

# ---------- prediction-level dominance on rank target ----------
predmed = Qp[:, 0, TAU05]; tgtN = np.full(len(idx_te), np.nan); tgtO = np.full(len(idx_te), np.nan)
for i in range(len(idx_te)):
    si, t = int(idx_te[i, 0]), int(idx_te[i, 1]); tt = t+1
    if CMASK[si, tt, 0] == 1 and CMASK[si, tt, 1] == 1:
        tgtN[i] = Fcdf(sN, COBS[si, tt, 0]); tgtO[i] = Fcdf(sO, COBS[si, tt, 1])
ok = (~np.isnan(tgtN)) & (Mp[:, 0] > 0)
pcorrN = float(np.corrcoef(predmed[ok], tgtN[ok])[0, 1]); pcorrO = float(np.corrcoef(predmed[ok], tgtO[ok])[0, 1])

R = {"index_robust_dominance": dom_robust, "index_rank_dominance": dom_rank,
     "thr_rank": thr_rank, "metrics_rank_target": met,
     "prediction_dominance_rank": {"pred_corr_NOx": pcorrN, "pred_corr_O3": pcorrO,
                                   "dominant": "O3" if pcorrO > pcorrN else "NOx"},
     "prediction_dominance_robust_E3": {"pred_corr_NOx": 0.116, "pred_corr_O3": 0.898, "dominant": "O3"}}
json.dump(R, open(ROOT / "results_e5.json", "w"), indent=2)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar([0, 1], [dom_robust["var_share_NOx"], dom_robust["var_share_O3"]], width=0.35, label="robust", color="tab:gray")
    ax[0].bar([0.4, 1.4], [dom_rank["var_share_NOx"], dom_rank["var_share_O3"]], width=0.35, label="rank", color="tab:green")
    ax[0].set_xticks([0.2, 1.2]); ax[0].set_xticklabels(["NOx", "O3"]); ax[0].set_ylabel("share of Var(H)")
    ax[0].set_title("Index-level variance share"); ax[0].legend()
    ax[1].bar([0, 1], [0.116, 0.898], width=0.35, label="robust (E3)", color="tab:gray")
    ax[1].bar([0.4, 1.4], [pcorrN, pcorrO], width=0.35, label="rank (E5)", color="tab:green")
    ax[1].set_xticks([0.2, 1.2]); ax[1].set_xticklabels(["NOx", "O3"]); ax[1].set_ylabel("corr(pred H, pollutant)")
    ax[1].set_title("Prediction-level dominance"); ax[1].legend()
    plt.tight_layout(); plt.savefig(FIGS / "e5_dominance.png", dpi=130); plt.close(); print("saved figs/e5_dominance.png", flush=True)
except Exception as e:
    print("figure skipped:", e, flush=True)

print("\n=============== E5 (rank index, test 2016) ===============")
print("metrics:", {k: round(v, 4) for k, v in met.items()})
print(f"index var-share  robust: NOx {dom_robust['var_share_NOx']*100:.1f}% / O3 {dom_robust['var_share_O3']*100:.1f}%")
print(f"index var-share  rank  : NOx {dom_rank['var_share_NOx']*100:.1f}% / O3 {dom_rank['var_share_O3']*100:.1f}%")
print(f"prediction corr  robust(E3): NOx 0.116 / O3 0.898")
print(f"prediction corr  rank (E5): NOx {pcorrN:.3f} / O3 {pcorrO:.3f}")

run_id = "E5-" + datetime.now().strftime("%Y%m%d-%H%M%S")
lines = [f"\n## Run {run_id}  (E5: rank/quantile-uniform combined index)",
         f"- date: {datetime.now().isoformat(timespec='seconds')} | code: pipeline/06_qrpinn_e5.py",
         f"- new target: H_rank = 0.5 F_NOx(NOx) + 0.5 F_O3(O3) (train empirical CDF); thr=Q30={thr_rank:.4f}",
         f"- split: train 2014-15 ({len(idx_tr)}) / test 2016 ({len(idx_te)}); seed=0", "",
         "| index | Var-share NOx | Var-share O3 | corr(H,NOx) | corr(H,O3) |",
         "|---|---|---|---|---|",
         f"| robust (old) | {dom_robust['var_share_NOx']*100:.1f}% | {dom_robust['var_share_O3']*100:.1f}% | {dom_robust['corr_H_NOx']:.3f} | {dom_robust['corr_H_O3']:.3f} |",
         f"| rank (new) | {dom_rank['var_share_NOx']*100:.1f}% | {dom_rank['var_share_O3']*100:.1f}% | {dom_rank['corr_H_NOx']:.3f} | {dom_rank['corr_H_O3']:.3f} |",
         "", "| prediction-level | corr(pred H, NOx) | corr(pred H, O3) |",
         "|---|---|---|",
         f"| robust (E3) | 0.116 | 0.898 |",
         f"| rank (E5) | {pcorrN:.3f} | {pcorrO:.3f} |",
         "", f"- FULL-rank metrics (rank target, NOT comparable to robust-target runs): " +
         ", ".join(f"{k}={round(v,4)}" for k, v in met.items()),
         f"- checkpoint artefacts/qrpinn_full_rank.pt ; fig figs/e5_dominance.png"]
with open(ROOT / "RESULTS_LOG.md", "a", encoding="utf-8") as f:
    f.write("\n".join(lines)+"\n")
print(f"\nWrote results_e5.json and appended {run_id} to RESULTS_LOG.md", flush=True)
