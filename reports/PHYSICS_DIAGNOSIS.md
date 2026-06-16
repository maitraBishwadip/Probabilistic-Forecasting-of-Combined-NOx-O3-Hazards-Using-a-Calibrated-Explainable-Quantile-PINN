# PHYSICS_DIAGNOSIS.md — Why the QR-PINN physics term was ineffective (and the fix)

**Scope.** Diagnosis of why the physics-informed component of the QR-PINN did not improve forecasts
(E3/E8), with code-level evidence, followed by the corrective design (E9). Findings reference real
code lines (`pipeline/09_qrpinn_e8.py`, identical physics in `03`–`07`) and `RESULTS_LOG.md` only.

---

## 0. Verdict
The original physics residual is a **near-neutral regularizer, not a working physics constraint** —
a **formulation problem, not a bug**. The term trains (parameters move off zero) and results are
seed-stable, so the neutrality is by design.

## 1. Evidence (from `RESULTS_LOG.md`)
- **E8 physics-weight sweep (adopted rank index):** λ=0 → pinball **0.2054** / Brier 0.1023;
  λ=0.2 → **0.2063** / 0.1014; λ=0.5 → **0.2188** / 0.1052. Physics does not help and *hurts* when
  up-weighted.
- **E3 (robust index):** FULL − DATA-ONLY = **−0.0135 pinball** (~1.3%) — a tiny effect that did
  **not** reproduce on the rank index (E8).
- **Not a fluke:** E8 seeds 0/1 nearly identical (0.2063 vs 0.2060). Learned rates small but non-zero
  (P_o=0.436, L_titr=0.175, D_o=0.169, others ≈0.16) — the term is active, just inconsequential.

## 2. Root causes (ranked, with code evidence)

**R1 — Architectural disconnect: physics never touches the forecast (primary cause).**
In `forward()` the forecast quantiles `q` are built from `hn[-1]` (the final hidden state), while the
physics head `chat = self.chead(out)` reconstructs the **past input window**. The physics residual is
computed on `chat` (past), and the training loss links physics to the forecast **only through the
shared LSTM encoder**. The residual therefore *never* constrains `H_{t+1..t+24}` — at best it nudges
the encoder's representation. The "PI" acts on history, not on the prediction it should govern.

**R2 — The auxiliary task is redundant (low information).**
`chat` is trained to reproduce the observed scaled NOx/O3 over the window, but those same series are
already **inputs** in `X`. Reconstructing inputs the encoder can already read is trivial autoencoding,
so neither the data-fit nor physics-on-that-reconstruction injects new signal.

**R3 — The box model collapses to persistence in scaled space.**
The residual operates on **robust-scaled** concentrations (sign-ambiguous, ≈N(0,·)) with learnable
softplus rates. The update `o_hat = o_prev + (P_o·g − L_titr·… − D_o·… )` is minimized trivially by
**small rates ⇒ o_hat ≈ o_prev**, i.e. it reduces to `MSE(o[1:], o[:-1])` — a weak temporal-smoothness
prior, not physics. The learned rates (≈0.16) are exactly this near-persistence regime; over-weighting
(λ=0.5) over-smooths and hurts.

**R4 — The actual chemistry is absent.**
`EXPERIMENT_DESIGN.md §3` specifies the **Leighton photostationary relation** and **Oₓ = O₃ + NO₂
conservation**. Neither is implemented: the aux head outputs only NOx and O3 (not NO and NO₂), so
NO₂ + hν → NO + O₃ and O₃ + NO → NO₂ cannot be expressed and Oₓ is never formed. The only coupling is
a heuristic `L_titr·relu(n)·σ(o)` term — the physically meaningful, sunlight-driven, *conserved*
constraint is missing.

**R5 — Space mismatch between physics and target.**
The forecast target is the **rank-uniform** `H = ½F_NOx+½F_O3 ∈ [0,1]`; the physics acts on
robust-scaled concentrations. Two different nonlinear transforms, coupled only via the encoder, so a
good concentration-space constraint is diluted before it reaches `H`.

**R6 — Predictability structure leaves physics nothing to add.**
Skill in `H` is O3-dominated (pred-corr O3 0.63–0.90; E3/E5), and O3's diurnal/photochemical signal is
already learned directly from met features. The physics coupling mostly concerns NOx (noisy,
traffic-driven) — the component contributing least to the score — so it cannot move the metric even if
correct.

## 3. What is ruled out
Not a gradient/wiring bug (parameters train; gradients flow), not seed variance (E8), not mere
under-weighting (λ=0.5 is *worse* — more weight ≠ more help, consistent with R3).

## 4. The fix (E9 — implemented as `pipeline/10_qrpinn_e9.py`)
Make physics **structural in the forward path** and **sunlight-driven**:
1. **Physics decoder (fixes R1):** a recurrent box-ODE rolls NOx and O3 forward over h=1..24; the
   forecast **median is the rollout**, so physics directly determines `H_{t+h}`.
2. **Sunlight-driven conversion (fixes R4, per directive):** photochemical O3 production and NOx
   processing are driven by the **sunlight proxy** `J` (`photochemical_activitiy_index`, backed by
   `solar_rad_Wm2`); NO-titration couples the two species.
3. **Physical units (fixes R3):** the ODE runs in ppb (un-scaled), and a future-concentration
   data-fit forces the decoder to learn real dynamics — rates can no longer degenerate to persistence
   because persistence would forecast poorly.
4. **Decisive test:** physics decoder vs a matched free-MLP forecast (no ODE, no J). Physics is
   validated only if it forecasts `H` at least as well **and** the learned rates are physical.
5. **Honesty fallback:** if still neutral, reframe as physics-*guided*/interpretable QR (report
   learned rates), not a physics-accuracy claim (`CLAUDE.md` integrity rules).

## 5. Outcome (what was actually run — `RESULTS_LOG.md`)

The first attempt at the §4 fix — the **physics-FORCED** trial (`pipeline/10b_physforce_trial.py`,
logged as PHYSFORCE) — over-corrected: it made the box-ODE the *sole* forecaster through **7 global
scalar rates**, which collapsed forecast capacity and made physics **much worse** (pinball 0.348 vs
0.210 free). That is a root cause R1 did not anticipate: *physics as the only forecaster is a capacity
bottleneck.*

The corrected fix is **E10** (`pipeline/11_qrpinn_e10.py`), a physics-**guided hybrid** that keeps full
forecast capacity: a stable, semi-implicit NO/NO₂/O₃ box-ODE (real Leighton photolysis + titration,
structural Oₓ=O₃+NO₂ conservation, **per-station** emission) runs in the forward path and sets the
forecast *base*, while a bounded NN residual corrects it. This fixed every **implementation** failure —
the ODE no longer collapses, and HYBRID (pinball **0.252–0.260**) and PHYS-ONLY (**0.317**) both far
outperform PHYSFORCE (0.348).

**But the scientific verdict (R6) is unchanged and is reported as fact:** the embedded chemistry does
**not** improve probabilistic accuracy of `H`. The data-only model is still best (FREE **0.195–0.210**,
stable across seeds 0/1), even though the physics variants are *handed future met* as forcing. So the
honesty fallback (point 5) applies: the QR-PINN's physics is **interpretable / physics-guided** — learned
rates are physically ordered (photolysis ≫ titration; warm-biased Arrhenius; small, ordered deposition)
and per-station emission scales recover real city-to-city intensity differences (NARAYANGANJ 3.3, GAZIPUR
1.1 high; BARISHAL 0.28, AGRABAD 0.37 low), and the hybrid forecast genuinely follows the rollout
(corr ≈ 0.75) — **not** an accuracy mechanism.
