# -*- coding: utf-8 -*-
"""
03_qrpinn_model.py  (E1 baselines + E2 QR-PINN)
Quantile-Regressive PINN for the full predictive distribution of the combined
NOx-O3 hazard index H, hourly multi-step (h=1..24), temporal split (train 2014-15 / test 2016).

Models compared:
  * CLIMATOLOGY : empirical H quantiles by (station, season, hour-of-day) from TRAIN.
  * QRNN        : LSTM encoder -> monotone non-crossing quantile head (pinball only). No physics.
  * QR-PINN     : QRNN + auxiliary NOx/O3 concentration head + coupled box-model physics residual.

Outputs: results_qrpinn.json  and appends a run block to RESULTS_LOG.md
No synthetic data: targets/metrics use observed H only; missing inputs were mask+zero in E0.
"""
import json, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from datetime import datetime

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
ART = ROOT / "artefacts"
torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"

# ---------------- config ----------------
L = 48            # lookback hours
HZ = 24           # forecast horizons (1..24 h)
TAUS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95], dtype=np.float32)
nT = len(TAUS)
EPOCHS = 12
BATCH = 512
LR = 1e-3
LAM_DATA = 0.3
LAM_PHYS_MAX = 0.2
ANNEAL = 6        # epochs to ramp physics weight 0 -> max

# ---------------- load E0 tensors ----------------
d = np.load(ART / "qrpinn_data.npz", allow_pickle=True)
X = d["X"].astype(np.float32)          # [S,T,F]
PHYS = d["PHYS"].astype(np.float32)    # [S,T,5] scaled: vent, blh, photo, precip, temp
COBS = d["COBS"].astype(np.float32)    # [S,T,2] scaled NOx,O3
CMASK = d["CMASK"].astype(np.float32)  # [S,T,2]
H = d["H"].astype(np.float32)          # [S,T]
year = d["year"].astype(np.int32)      # [T]
thr = float(d["thr"][0])
stations = [str(s) for s in d["stations"]]
S, T, F = X.shape
month = np.array([( (np.arange(T) ) )])  # placeholder; compute season/hour from a date range
import pandas as pd
times = pd.date_range("2014-01-01 00:00", periods=T, freq="h")
hour = times.hour.values
season = np.select([np.isin(times.month,[12,1,2]), np.isin(times.month,[3,4,5]),
                    np.isin(times.month,[6,7,8,9])], [0,1,2], default=3)
print(f"Loaded X{X.shape} F={F} thr={thr:.4f} stations={len(stations)}")

# ---------------- build anchor index lists (no 2015->2016 target bleed) ----------------
def build_index():
    tr, te = [], []
    for si in range(S):
        Hs = H[si]
        for t in range(L-1, T-HZ):
            if np.isnan(Hs[t]):            # require known current state
                continue
            fut = Hs[t+1:t+1+HZ]
            if not np.any(~np.isnan(fut)): # need >=1 valid future target
                continue
            if year[t+HZ] <= 2015:
                tr.append((si, t))
            elif year[t] == 2016:
                te.append((si, t))
    return np.array(tr, dtype=np.int32), np.array(te, dtype=np.int32)
idx_tr, idx_te = build_index()
print(f"anchors: train={len(idx_tr)}  test={len(idx_te)}")

# ---------------- climatology baseline ----------------
# train H observations -> quantiles per (station, season, hour), with fallbacks
def climatology():
    cells = {}
    for si in range(S):
        for t in range(T):
            if year[t] <= 2015 and not np.isnan(H[si, t]):
                cells.setdefault((si, season[t], hour[t]), []).append(H[si, t])
    glob = []
    for si in range(S):
        glob += [H[si, t] for t in range(T) if year[t] <= 2015 and not np.isnan(H[si, t])]
    glob_q = np.quantile(np.array(glob), TAUS).astype(np.float32)
    q = {k: np.quantile(np.array(v), TAUS).astype(np.float32) for k, v in cells.items() if len(v) >= 20}
    return q, glob_q
clim_q, clim_glob = climatology()
def clim_lookup(si, tt):
    return clim_q.get((si, season[tt], hour[tt]), clim_glob)

# ---------------- torch dataset ----------------
class DS(torch.utils.data.Dataset):
    def __init__(self, idx): self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        si, t = int(self.idx[i, 0]), int(self.idx[i, 1])
        x = X[si, t-L+1:t+1, :]                       # [L,F]
        pw = PHYS[si, t-L+1:t+1, :]                   # [L,5]
        cobs = COBS[si, t-L+1:t+1, :]                 # [L,2]
        cmask = CMASK[si, t-L+1:t+1, :]               # [L,2]
        fut = H[si, t+1:t+1+HZ].copy()                # [HZ]
        tmask = (~np.isnan(fut)).astype(np.float32)
        fut[np.isnan(fut)] = 0.0
        return (torch.from_numpy(x), torch.tensor(si), torch.from_numpy(pw),
                torch.from_numpy(cobs), torch.from_numpy(cmask),
                torch.from_numpy(fut), torch.from_numpy(tmask))

# ---------------- model ----------------
class QRPINN(nn.Module):
    def __init__(self, F, n_station, hidden=64, emb=8, physics=True, aux=True):
        super().__init__()
        self.physics = physics; self.aux = aux
        self.emb = nn.Embedding(n_station, emb)
        self.lstm = nn.LSTM(F, hidden, batch_first=True)
        trunk_in = hidden + emb
        self.base = nn.Linear(trunk_in, HZ)
        self.inc = nn.Linear(trunk_in, HZ*(nT-1))
        if aux:
            self.chead = nn.Linear(hidden, 2)         # per-timestep NOx,O3 reconstruction
        # learnable POSITIVE physics params (softplus): P_o,L_titr,D_o,W_o,E_n,D_n,W_n
        self.praw = nn.Parameter(torch.zeros(7))
    def forward(self, x, si):
        out, (hn, cn) = self.lstm(x)                  # out[B,L,h], hn[1,B,h]
        z = hn[-1]                                     # [B,h]
        trunk = torch.cat([z, self.emb(si)], dim=1)
        base = self.base(trunk)                        # [B,HZ]
        inc = torch.nn.functional.softplus(self.inc(trunk)).view(-1, HZ, nT-1)
        q = torch.cat([base.unsqueeze(-1), base.unsqueeze(-1) + torch.cumsum(inc, dim=-1)], dim=-1)
        chat = self.chead(out) if self.aux else None   # [B,L,2]
        return q, chat

def pos(p): return torch.nn.functional.softplus(p)

def physics_residual(chat, pw, praw):
    # chat[B,L,2] scaled (NOx,O3); pw[B,L,5] scaled vent,blh,photo,precip,temp
    n = chat[:, :, 0]; o = chat[:, :, 1]
    g_vent = torch.sigmoid(pw[:, :, 0]); g_photo = torch.sigmoid(pw[:, :, 2]); g_pr = torch.sigmoid(pw[:, :, 3])
    P_o, L_titr, D_o, W_o, E_n, D_n, W_n = [pos(praw[i]) for i in range(7)]
    n_prev, o_prev = n[:, :-1], o[:, :-1]
    gv, gp, gr = g_vent[:, :-1], g_photo[:, :-1], g_pr[:, :-1]
    # coupled box step (O3 produced by photochem, lost by NOx titration + ventilation + wet;
    #                    NOx emitted, lost by ventilation + wet) -- the NOx-O3 coupling is L_titr term
    o_hat = o_prev + (P_o*gp - L_titr*torch.relu(n_prev)*torch.sigmoid(o_prev)
                      - D_o*gv*o_prev - W_o*gr*o_prev)
    n_hat = n_prev + (E_n - D_n*gv*n_prev - W_n*gr*n_prev)
    return ((o[:, 1:]-o_hat)**2).mean() + ((n[:, 1:]-n_hat)**2).mean()

def pinball_loss(q, y, tmask):
    # q[B,HZ,nT], y[B,HZ], tmask[B,HZ]
    taus = torch.tensor(TAUS, device=q.device).view(1, 1, nT)
    err = y.unsqueeze(-1) - q
    l = torch.maximum(taus*err, (taus-1)*err)          # [B,HZ,nT]
    l = l.mean(-1) * tmask                              # avg over tau, mask horizons
    return l.sum() / tmask.sum().clamp(min=1)

def train_model(name, physics, aux):
    torch.manual_seed(0)
    model = QRPINN(F, S, physics=physics, aux=aux).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl = torch.utils.data.DataLoader(DS(idx_tr), batch_size=BATCH, shuffle=True, num_workers=0)
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = 0; nb = 0
        lam_phys = (LAM_PHYS_MAX * min(1.0, (ep+1)/ANNEAL)) if physics else 0.0
        for x, si, pw, cobs, cmask, fut, tmask in dl:
            opt.zero_grad()
            q, chat = model(x, si)
            loss = pinball_loss(q, fut, tmask)
            if aux:
                dm = ((chat - cobs)**2 * cmask).sum() / cmask.sum().clamp(min=1)
                loss = loss + LAM_DATA*dm
            if physics:
                loss = loss + lam_phys*physics_residual(chat, pw, model.praw)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  [{name}] epoch {ep+1}/{EPOCHS} loss={tot/nb:.4f} lam_phys={lam_phys:.3f} ({time.time()-t0:.0f}s)")
    return model

# ---------------- evaluation ----------------
def cdf_at(qs, thr):
    N, m = qs.shape
    k = np.sum(qs < thr, axis=1)
    lo = np.clip(k-1, 0, m-1); hi = np.clip(k, 0, m-1)
    ar = np.arange(N)
    qlo, qhi = qs[ar, lo], qs[ar, hi]; tlo, thi = TAUS[lo], TAUS[hi]
    gap = qhi-qlo; den = np.where(gap > 1e-9, gap, 1.0)
    frac = np.where(gap > 1e-9, (thr-qlo)/den, 0.0)
    cdf = tlo + frac*(thi-tlo)
    cdf = np.where(k == 0, TAUS[0], cdf); cdf = np.where(k == m, TAUS[-1], cdf)
    return np.clip(cdf, 0, 1)

def collect_preds(model):
    model.eval()
    dl = torch.utils.data.DataLoader(DS(idx_te), batch_size=1024, shuffle=False, num_workers=0)
    Q, Y, M, HZi = [], [], [], []
    with torch.no_grad():
        for x, si, pw, cobs, cmask, fut, tmask in dl:
            q, _ = model(x, si)
            Q.append(q.numpy()); Y.append(fut.numpy()); M.append(tmask.numpy())
    Q = np.concatenate(Q); Y = np.concatenate(Y); M = np.concatenate(M)  # [N,HZ,nT],[N,HZ]
    return Q, Y, M

def collect_clim():
    Q = np.zeros((len(idx_te), HZ, nT), dtype=np.float32)
    Y = np.zeros((len(idx_te), HZ), dtype=np.float32)
    M = np.zeros((len(idx_te), HZ), dtype=np.float32)
    for i in range(len(idx_te)):
        si, t = int(idx_te[i, 0]), int(idx_te[i, 1])
        for h in range(HZ):
            tt = t+1+h
            Q[i, h] = clim_lookup(si, tt)
            v = H[si, tt]
            if not np.isnan(v): Y[i, h] = v; M[i, h] = 1.0
    return Q, Y, M

def metrics(Q, Y, M):
    taus = TAUS.reshape(1, 1, nT)
    err = Y[..., None] - Q
    pb = np.maximum(taus*err, (taus-1)*err)            # [N,HZ,nT]
    m3 = M[..., None]
    pin = (pb*m3).sum()/m3.sum()/1.0
    pin_tau = (pb*m3).sum(axis=(0,1))/m3.sum(axis=(0,1))   # per tau
    crps = 2*pin_tau.mean()                              # quantile-based CRPS estimator
    # coverage / width for 80% (tau .1-.9) and 90% (.05-.95)
    def cov(lo_i, hi_i):
        lo = Q[:, :, lo_i]; hi = Q[:, :, hi_i]
        inside = ((Y >= lo) & (Y <= hi)).astype(float)*M
        picp = inside.sum()/M.sum()
        width = ((hi-lo)*M).sum()/M.sum()
        return float(picp), float(width)
    p80, w80 = cov(1, 7); p90, w90 = cov(0, 8)
    # per-horizon pinball
    perh = ((pb*m3).sum(axis=(0,2))/ (M.sum(axis=0)*nT)).tolist()
    # Brier on exceedance P(H>=thr)
    Qf = Q.reshape(-1, nT); Yf = Y.reshape(-1); Mf = M.reshape(-1)
    cdf = cdf_at(Qf, thr); pexc = 1.0 - cdf
    ind = (Yf >= thr).astype(float)
    sel = Mf > 0
    brier = float((((pexc-ind)**2)[sel]).mean())
    base_rate = float(ind[sel].mean())
    # horizon-1 brier
    Q1 = Q[:, 0, :]; Y1 = Y[:, 0]; M1 = M[:, 0] > 0
    p1 = 1.0 - cdf_at(Q1, thr); i1 = (Y1 >= thr).astype(float)
    brier1 = float((((p1-i1)**2)[M1]).mean())
    return {"pinball": float(pin), "crps": float(crps),
            "picp80": p80, "width80": w80, "picp90": p90, "width90": w90,
            "brier_exceed": brier, "brier_exceed_h1": brier1, "test_base_rate_extreme": base_rate,
            "pinball_per_horizon": perh}

# ---------------- run ----------------
results = {"config": {"L": L, "HZ": HZ, "taus": TAUS.tolist(), "epochs": EPOCHS,
                      "batch": BATCH, "lr": LR, "lam_data": LAM_DATA, "lam_phys_max": LAM_PHYS_MAX,
                      "thr_Q30": thr, "n_train_anchors": int(len(idx_tr)),
                      "n_test_anchors": int(len(idx_te))},
           "models": {}}

print("\n== CLIMATOLOGY ==")
Qc, Yc, Mc = collect_clim(); results["models"]["climatology"] = metrics(Qc, Yc, Mc)

print("\n== QRNN (no physics, no aux) ==")
m_qrnn = train_model("QRNN", physics=False, aux=False)
Qq, Yq, Mq = collect_preds(m_qrnn); results["models"]["QRNN"] = metrics(Qq, Yq, Mq)

print("\n== QR-PINN (aux + physics) ==")
m_pinn = train_model("QR-PINN", physics=True, aux=True)
Qp, Yp, Mp = collect_preds(m_pinn); results["models"]["QR-PINN"] = metrics(Qp, Yp, Mp)
results["models"]["QR-PINN"]["learned_physics_params"] = {
    k: float(torch.nn.functional.softplus(m_pinn.praw[i]).item())
    for i, k in enumerate(["P_o","L_titr","D_o","W_o","E_n","D_n","W_n"])}

json.dump(results, open(ROOT / "results_qrpinn.json", "w"), indent=2)

# ---------------- console summary ----------------
print("\n================= TEST 2016 RESULTS =================")
hdr = f"{'model':12s} {'pinball':>8s} {'CRPS':>7s} {'PICP80':>7s} {'PICP90':>7s} {'Brier':>7s} {'Brier@1':>8s}"
print(hdr)
for k, v in results["models"].items():
    print(f"{k:12s} {v['pinball']:8.4f} {v['crps']:7.4f} {v['picp80']:7.3f} {v['picp90']:7.3f} "
          f"{v['brier_exceed']:7.4f} {v['brier_exceed_h1']:8.4f}")
print(f"(test base-rate extreme H>=thr = {results['models']['climatology']['test_base_rate_extreme']:.3f}; "
      f"nominal coverage 0.80 / 0.90)")

# ---------------- append to RESULTS_LOG.md ----------------
run_id = "QRPINN-" + datetime.now().strftime("%Y%m%d-%H%M%S")
lines = [f"\n## Run {run_id}", f"- date: {datetime.now().isoformat(timespec='seconds')}",
         f"- code: pipeline/02_qrpinn_dataprep.py + pipeline/03_qrpinn_model.py",
         f"- config: L={L}, HZ={HZ}, taus={TAUS.tolist()}, epochs={EPOCHS}, batch={BATCH}, lr={LR}, "
         f"lam_data={LAM_DATA}, lam_phys_max={LAM_PHYS_MAX}, thr_Q30={thr:.4f}",
         f"- split: train 2014-15 ({len(idx_tr)} anchors) / test 2016 ({len(idx_te)} anchors); seed=0",
         f"- target: combined index H=0.5 rs(NOx)+0.5 rs(O3); extreme = H>=Q30(train) (~70% class)",
         "", "| model | pinball | CRPS | PICP80 | PICP90 | width80 | Brier(exceed) | Brier@h1 |",
         "|---|---|---|---|---|---|---|---|"]
for k, v in results["models"].items():
    lines.append(f"| {k} | {v['pinball']:.4f} | {v['crps']:.4f} | {v['picp80']:.3f} | {v['picp90']:.3f} "
                 f"| {v['width80']:.3f} | {v['brier_exceed']:.4f} | {v['brier_exceed_h1']:.4f} |")
lp = results["models"]["QR-PINN"]["learned_physics_params"]
lines += ["", f"- QR-PINN learned physics params (softplus>0): " +
          ", ".join(f"{k}={v:.3f}" for k, v in lp.items()),
          f"- nominal interval coverage targets: PICP80=0.80, PICP90=0.90",
          f"- NOTE: first-version box model (coupled NOx-O3, within-window, leakage-free); "
          f"full Leighton/Ox residual + LOSO are later runs."]
logp = ROOT / "RESULTS_LOG.md"
header = "" if logp.exists() else "# RESULTS LOG — source of truth for the paper (see CLAUDE.md)\n"
with open(logp, "a", encoding="utf-8") as f:
    if header: f.write(header)
    f.write("\n".join(lines) + "\n")
print(f"\nAppended {run_id} to RESULTS_LOG.md; wrote results_qrpinn.json")
