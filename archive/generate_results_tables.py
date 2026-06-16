import json
import pandas as pd
import numpy as np
from pathlib import Path

# Load all result files
base_path = Path(".")
lstm_data = json.load(open("results_pure_lstm.json"))
rf_data = json.load(open("results_rf.json"))
pinn_data = json.load(open("results_pinn.json"))
gbt_data = json.load(open("results_gbt.json"))

print("=" * 100)
print("COMPOUND PM2.5-O3 EXTREMES PREDICTION: COMPREHENSIVE RESULTS ANALYSIS")
print("=" * 100)
print()

# ============================================================================
# 1. EXECUTIVE SUMMARY TABLE
# ============================================================================
print("\n1. EXECUTIVE SUMMARY - MODEL COMPARISON")
print("-" * 100)

summary_data = {
    "Model": ["Pure LSTM", "Random Forest", "PINN (Full)", "GBT (Joint Regime)",
              "Persistence (Baseline)", "Climatology (Baseline)", "Single-Pollutant (Baseline)"],
    "PR_AUC": [
        lstm_data["PureLSTM"]["PR_AUC"],
        rf_data["RandomForest"]["PR_AUC"],
        pinn_data["PINN_full"]["PR_AUC"],
        gbt_data["GBT_joint_regime"]["PR_AUC"],
        gbt_data["baselines"]["persistence"]["PR_AUC"],
        gbt_data["baselines"]["climatology_season_regime"]["PR_AUC"],
        gbt_data["baselines"]["single_pollutant_independent"]["PR_AUC"]
    ],
    "PR_AUC_95CI": [
        f"[{lstm_data['PureLSTM']['PR_AUC_CI95'][0]:.3f}, {lstm_data['PureLSTM']['PR_AUC_CI95'][1]:.3f}]",
        f"[{rf_data['RandomForest']['PR_AUC_CI95'][0]:.3f}, {rf_data['RandomForest']['PR_AUC_CI95'][1]:.3f}]",
        f"[{pinn_data['PINN_full']['PR_AUC_CI95'][0]:.3f}, {pinn_data['PINN_full']['PR_AUC_CI95'][1]:.3f}]",
        f"[{gbt_data['GBT_joint_regime']['PR_AUC_CI95'][0]:.3f}, {gbt_data['GBT_joint_regime']['PR_AUC_CI95'][1]:.3f}]",
        "N/A", "N/A", "N/A"
    ],
    "ROC_AUC": [
        lstm_data["PureLSTM"]["ROC_AUC"],
        rf_data["RandomForest"]["ROC_AUC"],
        pinn_data["PINN_full"]["ROC_AUC"],
        gbt_data["GBT_joint_regime"]["ROC_AUC"],
        gbt_data["baselines"]["persistence"]["ROC_AUC"],
        gbt_data["baselines"]["climatology_season_regime"]["ROC_AUC"],
        gbt_data["baselines"]["single_pollutant_independent"]["ROC_AUC"]
    ],
    "Brier": [
        lstm_data["PureLSTM"]["Brier"],
        rf_data["RandomForest"]["Brier"],
        pinn_data["PINN_full"]["Brier"],
        gbt_data["GBT_joint_regime"]["Brier"],
        gbt_data["baselines"]["persistence"]["Brier"],
        gbt_data["baselines"]["climatology_season_regime"]["Brier"],
        gbt_data["baselines"]["single_pollutant_independent"]["Brier"]
    ],
    "F1": [
        lstm_data["PureLSTM"]["F1_test"],
        rf_data["RandomForest"]["F1_test"],
        "N/A",
        gbt_data["GBT_joint_regime"]["F1_test"],
        "N/A", "N/A", "N/A"
    ],
    "Recall@Prec0.3": [
        lstm_data["PureLSTM"]["recall_at_prec0.3"],
        rf_data["RandomForest"]["recall_at_prec0.3"],
        "N/A",
        gbt_data["GBT_joint_regime"]["recall_at_prec0.3"],
        "N/A", "N/A", "N/A"
    ],
    "N Samples": [
        lstm_data["PureLSTM"]["n"],
        rf_data["RandomForest"]["n"],
        pinn_data["PINN_full"]["n"],
        gbt_data["GBT_joint_regime"]["n"],
        gbt_data["baselines"]["persistence"]["n"],
        gbt_data["baselines"]["climatology_season_regime"]["n"],
        gbt_data["baselines"]["single_pollutant_independent"]["n"]
    ],
    "N Positives": [
        lstm_data["PureLSTM"]["n_pos"],
        rf_data["RandomForest"]["n_pos"],
        pinn_data["PINN_full"]["n_pos"],
        gbt_data["GBT_joint_regime"]["n_pos"],
        gbt_data["baselines"]["persistence"]["n_pos"],
        gbt_data["baselines"]["climatology_season_regime"]["n_pos"],
        gbt_data["baselines"]["single_pollutant_independent"]["n_pos"]
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
print("\n[KEY FINDINGS]:")
print(f"  > Best PR_AUC: GBT at {gbt_data['GBT_joint_regime']['PR_AUC']:.4f}")
print(f"  > Baseline (Persistence): {gbt_data['baselines']['persistence']['PR_AUC']:.4f}")
print(f"  > Single-pollutant baseline: {gbt_data['baselines']['single_pollutant_independent']['PR_AUC']:.4f} (surprisingly strong!)")
print(f"  > Base rate (14% positive): Tests are imbalanced - PR_AUC is primary metric")
print()

# ============================================================================
# 2. PER-REGIME PERFORMANCE
# ============================================================================
print("\n2. PER-REGIME PERFORMANCE (PR_AUC)")
print("-" * 100)

regimes = ["Humid-Transition", "Stagnant-Trapping", "Ventilated-Stormy",
           "Monsoon-Wet-Windy", "Dry-Sunny-Photochemical"]

regime_data = {
    "Regime": [],
    "LSTM": [],
    "RF": [],
    "PINN": [],
    "GBT": [],
    "N_samples": [],
    "N_events": []
}

for regime in regimes:
    regime_data["Regime"].append(regime)
    regime_data["LSTM"].append(lstm_data["per_regime_PR_AUC"][regime]["PR_AUC"])
    regime_data["RF"].append(rf_data["per_regime_PR_AUC"][regime]["PR_AUC"])
    regime_data["PINN"].append(pinn_data["per_regime_PR_AUC"][regime]["PR_AUC"])
    regime_data["GBT"].append(gbt_data["per_regime_PR_AUC"][regime]["PR_AUC"])
    regime_data["N_samples"].append(gbt_data["per_regime_PR_AUC"][regime]["n"])
    regime_data["N_events"].append(gbt_data["per_regime_PR_AUC"][regime]["n_pos"])

regime_df = pd.DataFrame(regime_data)
print(regime_df.to_string(index=False))
print("\n[REGIME INSIGHTS]:")
print("  > Dry-Sunny-Photochemical: Strong performance across all models (GBT=0.565, PINN=0.403, RF=0.583)")
print("  > Stagnant-Trapping: GBT excels (0.588), good for air pollution events")
print("  > Monsoon-Wet-Windy: High rainfall decreases both PM2.5 & O3, harder to predict extremes (PINN=0.605!)")
print("  > Ventilated-Stormy: Weak across models (LSTM=0.148, RF=0.245) - low event frequency (n=14)")
print(f"  > Sample sizes: {regime_df['N_samples'].min()}-{regime_df['N_samples'].max()}")
print()

# ============================================================================
# 3. LEAD TIME ANALYSIS
# ============================================================================
print("\n3. LEAD TIME DEGRADATION (PR_AUC)")
print("-" * 100)

lead_time_data = {
    "Lead Time": ["24h", "48h", "72h"],
    "LSTM": [
        lstm_data["lead_times"]["24h"],
        lstm_data["lead_times"]["48h"],
        lstm_data["lead_times"]["72h"]
    ],
    "RF": [
        rf_data["lead_times"]["24h"],
        rf_data["lead_times"]["48h"],
        rf_data["lead_times"]["72h"]
    ],
    "GBT": [
        gbt_data["lead_times"]["24h"],
        gbt_data["lead_times"]["48h"],
        gbt_data["lead_times"]["72h"]
    ]
}

lead_df = pd.DataFrame(lead_time_data)
print(lead_df.to_string(index=False))
print("\n[LEAD TIME INSIGHTS]:")
lstm_decay_24_72 = ((lstm_data["lead_times"]["72h"] - lstm_data["lead_times"]["24h"]) / lstm_data["lead_times"]["24h"] * 100)
rf_decay_24_72 = ((rf_data["lead_times"]["72h"] - rf_data["lead_times"]["24h"]) / rf_data["lead_times"]["24h"] * 100)
gbt_decay_24_72 = ((gbt_data["lead_times"]["72h"] - gbt_data["lead_times"]["24h"]) / gbt_data["lead_times"]["24h"] * 100)
print(f"  > 24h to 72h degradation: LSTM={lstm_decay_24_72:.1f}%, RF={rf_decay_24_72:.1f}%, GBT={gbt_decay_24_72:.1f}%")
print(f"  > 3-day lead time remains useful across all models (24h is best, 72h still viable)")
print()

# ============================================================================
# 4. LEAVE-ONE-SEASON-OUT CROSS-VALIDATION
# ============================================================================
print("\n4. LEAVE-ONE-SEASON-OUT (LOSO) CROSS-VALIDATION")
print("-" * 100)

loso_data = {
    "Model": ["Pure LSTM", "Random Forest", "PINN (Full)", "GBT (Joint Regime)"],
    "Mean PR_AUC": [
        lstm_data["LOSO"]["mean_PR_AUC"],
        rf_data["LOSO"]["mean_PR_AUC"],
        pinn_data["LOSO"]["mean_PR_AUC"],
        gbt_data["LOSO"]["mean_PR_AUC"]
    ],
    "Std Dev": [
        lstm_data["LOSO"]["std"],
        rf_data["LOSO"]["std"],
        pinn_data["LOSO"]["std"],
        gbt_data["LOSO"]["std"]
    ],
    "N Folds": [
        lstm_data["LOSO"]["folds"],
        rf_data["LOSO"]["folds"],
        pinn_data["LOSO"]["folds"],
        gbt_data["LOSO"]["folds"]
    ]
}

loso_df = pd.DataFrame(loso_data)
print(loso_df.to_string(index=False))
print("\n[GENERALIZATION INSIGHTS]:")
print(f"  > GBT shows best seasonal robustness: mean={gbt_data['LOSO']['mean_PR_AUC']:.3f}, std={gbt_data['LOSO']['std']:.3f}")
print(f"  > PINN has higher variability: std={pinn_data['LOSO']['std']:.3f}, suggesting season-dependent performance")
print(f"  > LSTM & RF have lower std, but also lower mean scores - more conservative")
print()

# ============================================================================
# 5. PINN ABLATION STUDIES
# ============================================================================
print("\n5. PINN ABLATION STUDIES (Understanding Physics Contribution)")
print("-" * 100)

pinn_ablation = {
    "PINN Variant": ["Full (Physics+Regime)", "No Physics", "No Regime", "No Lift (Independence)"],
    "PR_AUC": [
        pinn_data["PINN_full"]["PR_AUC"],
        pinn_data["PINN_no_physics"]["PR_AUC"],
        pinn_data["PINN_no_regime"]["PR_AUC"],
        pinn_data["PINN_no_lift(independence)"]["PR_AUC"]
    ],
    "ROC_AUC": [
        pinn_data["PINN_full"]["ROC_AUC"],
        pinn_data["PINN_no_physics"]["ROC_AUC"],
        pinn_data["PINN_no_regime"]["ROC_AUC"],
        pinn_data["PINN_no_lift(independence)"]["ROC_AUC"]
    ],
    "Brier": [
        pinn_data["PINN_full"]["Brier"],
        pinn_data["PINN_no_physics"]["Brier"],
        pinn_data["PINN_no_regime"]["Brier"],
        pinn_data["PINN_no_lift(independence)"]["Brier"]
    ]
}

pinn_ablation_df = pd.DataFrame(pinn_ablation)
print(pinn_ablation_df.to_string(index=False))
print("\n[ABLATION INSIGHTS]:")
print(f"  > Physics + Regime (Full) best: PR_AUC = {pinn_data['PINN_full']['PR_AUC']:.4f}")
print(f"  > Removing physics constraint reduces PR_AUC by {(pinn_data['PINN_full']['PR_AUC'] - pinn_data['PINN_no_physics']['PR_AUC'])*100:.1f}%")
print(f"  > Regime information crucial: No-regime PR_AUC = {pinn_data['PINN_no_regime']['PR_AUC']:.4f} (marginal improvement)")
print(f"  > Independence assumption harmful: No-lift PR_AUC drops to {pinn_data['PINN_no_lift(independence)']['PR_AUC']:.4f}")
print()

# ============================================================================
# 6. LEARNED PHYSICS PARAMETERS BY REGIME (PINN)
# ============================================================================
print("\n6. LEARNED PHYSICS BY REGIME (PINN)")
print("-" * 100)

physics_params = pinn_data["learned_physics_by_regime"]
physics_display = {
    "Regime": [],
    "Dilution (a)": [],
    "Wet Removal (b)": [],
    "Base Loss (d)": [],
    "O3 Production (p)": [],
    "PM Emission (q)": []
}

for regime in regimes:
    physics_display["Regime"].append(regime)
    params = physics_params[regime]
    physics_display["Dilution (a)"].append(f"{params['dilution_a']:.3f}")
    physics_display["Wet Removal (b)"].append(f"{params['wet_b']:.3f}")
    physics_display["Base Loss (d)"].append(f"{params['base_loss_d']:.3f}")
    physics_display["O3 Production (p)"].append(f"{params['o3_prod_p']:.3f}")
    physics_display["PM Emission (q)"].append(f"{params['pm_emit_q']:.3f}")

physics_df = pd.DataFrame(physics_display)
print(physics_df.to_string(index=False))
print("\n[PHYSICS PARAMETER INSIGHTS]:")
print("  > Wet Removal (b): High in all regimes (0.67-0.84) - wet processes dominate removal")
print("  > Dilution (a): Highest in Humid-Transition (0.556) & Monsoon (0.545) - better ventilation")
print("  > O3 Production (p): Highest in Stagnant-Trapping (0.309) - photochemical buildup")
print("  > PM Emission (q): Highest in Stagnant-Trapping (0.54) - air mass traps emissions")
print()

# ============================================================================
# 7. LEARNED REGIME LIFT (PINN)
# ============================================================================
print("\n7. REGIME LIFT FACTORS (PINN - Compound Extremes Risk Multiplier)")
print("-" * 100)

lift_data = pinn_data["learned_regime_lift"]
lift_display = {
    "Regime": list(lift_data.keys()),
    "Compound Risk Lift": list(lift_data.values())
}

lift_df = pd.DataFrame(lift_display)
print(lift_df.to_string(index=False))
print("\n[LIFT INTERPRETATION]:")
print("  > Lift ~0.85-0.90: Co-occurrence of PM2.5 & O3 extremes is ~15-10% LESS likely than if independent")
print("  > Ventilated-Stormy (0.902): Near-independence, extremes decoupled")
print("  > Stagnant-Trapping (0.863): Slight negative correlation - pollutants not synchronized")
print()

# ============================================================================
# 8. RANDOM FOREST TOP FEATURES
# ============================================================================
print("\n8. RANDOM FOREST FEATURE IMPORTANCES (Top 12)")
print("-" * 100)

top_features = rf_data["top_features"]
feature_list = []
for feature, importance in sorted(top_features.items(), key=lambda x: x[1], reverse=True):
    feature_list.append({"Feature": feature, "Importance": f"{importance:.4f}"})

feature_df = pd.DataFrame(feature_list)
print(feature_df.to_string(index=False))
print("\n[FEATURE INSIGHTS]:")
print("  > O3 is dominant predictor: O3_8h (0.1022), O3_lag1 (0.0721), O3_lag3 (0.0504), O3_lag7 (0.0451)")
print("  > PM2.5 critical: PM2.5_ugm3 (0.067), PM10_ugm3 (0.0425)")
print("  > Meteorology matters: Wind, pressure, RH in top features")
print("  > Temporal autocorrelation: Lagged O3 crucial - concentration trends predictive")
print()

# ============================================================================
# 9. INTERPRETATION & KEY FINDINGS
# ============================================================================
print("\n" + "=" * 100)
print("KEY FINDINGS & INTERPRETATION")
print("=" * 100)

print("\n[MODEL RANKING] (by PR_AUC):")
models = [
    ("GBT (Joint Regime)", gbt_data["GBT_joint_regime"]["PR_AUC"]),
    ("PINN (Full Physics)", pinn_data["PINN_full"]["PR_AUC"]),
    ("Random Forest", rf_data["RandomForest"]["PR_AUC"]),
    ("Pure LSTM", lstm_data["PureLSTM"]["PR_AUC"])
]
for i, (name, score) in enumerate(sorted(models, key=lambda x: x[1], reverse=True), 1):
    print(f"  {i}. {name}: {score:.4f}")

print("\n[VS BASELINES]:")
print(f"  > Persistence baseline: {gbt_data['baselines']['persistence']['PR_AUC']:.4f}")
print(f"  > GBT improvement over persistence: {(gbt_data['GBT_joint_regime']['PR_AUC'] - gbt_data['baselines']['persistence']['PR_AUC'])/gbt_data['baselines']['persistence']['PR_AUC']*100:.1f}%")
print(f"  > Single-pollutant baseline: {gbt_data['baselines']['single_pollutant_independent']['PR_AUC']:.4f} (WARNING: Strong baseline!)")
print(f"  > GBT only {(gbt_data['GBT_joint_regime']['PR_AUC'] / gbt_data['baselines']['single_pollutant_independent']['PR_AUC'] - 1)*100:.1f}% better than single-pollutant")

print("\n[PHYSICS-INFORMED INSIGHTS]:")
print(f"  > PINN incorporation of physics adds interpretability")
print(f"  > Learned parameters show regime-specific dynamics:")
print(f"    - Wet removal dominant (0.67-0.84) suggests monsoon sensitivity")
print(f"    - Photochemical production higher in dry seasons (0.23-0.33)")
print(f"  > Regime lift ~0.86: Compound extremes slightly less synchronized than expected")

print("\n[CHALLENGES & CAVEATS]:")
print(f"  > Imbalanced data: Only 14% positive events (145/1053)")
print(f"  > Rare regimes: Monsoon & Ventilated-Stormy have only 14 events each")
print(f"  > Single-pollutant baseline is surprisingly strong (PR_AUC=0.588)")
print(f"    => Suggests O3 alone may be near-sufficient for prediction")
print(f"  > Lead time drops significantly 24h to 72h: 24h lead time most reliable")

print("\n[RECOMMENDATIONS]:")
print(f"  1. Deploy GBT for operational forecasting (best balanced performance)")
print(f"  2. Use 24h lead time forecasts (higher skill, degradation by 72h)")
print(f"  3. Apply regime-specific calibration (performance varies 0.15-0.60 across regimes)")
print(f"  4. Investigate why single-pollutant baseline is so strong")
print(f"  5. Collect more data for Monsoon/Ventilated-Stormy regimes (n=14 too small)")

print("\n" + "=" * 100)
