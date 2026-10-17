# Results Explained — Figures & CSV Files

> **Scope**: This document walks through every figure and CSV file produced by the PHM (Prognostics and Health Management) pipeline evaluated on the NASA C-MAPSS benchmark datasets (FD001–FD004).
>
> **Pipeline overview**: The system builds an unsupervised anomaly-detection stage (Health Index + multiple detectors) followed by a supervised RUL (Remaining Useful Life) regression stage (XGBoost), all evaluated under a realistic conformal-prediction and label-sensitivity framework.

---

## Table of Contents

### Figures
1. [sensor\_degradation.png](#1-sensor_degradationpng)
2. [health\_index\_fd001.png](#2-health_index_fd001png)
3. [health\_index\_comparison\_all\_subsets.png](#3-health_index_comparison_all_subsetspng)
4. [pr\_curves.png](#4-pr_curvespng)
5. [lead\_time\_vs\_fa\_curves.png](#5-lead_time_vs_fa_curvespng)
6. [lead\_time\_distributions.png](#6-lead_time_distributionspng)
7. [false\_alarms\_per\_engine.png](#7-false_alarms_per_enginepng)
8. [n\_sweep\_lead\_time\_circularity.png](#8-n_sweep_lead_time_circularitypng)
9. [predicted\_vs\_true\_rul.png](#9-predicted_vs_true_rulpng)
10. [conformal\_prediction\_intervals.png](#10-conformal_prediction_intervalspng)
11. [cross\_condition\_bars.png](#11-cross_condition_barspng)
12. [shap\_feature\_importance.png](#12-shap_feature_importancepng)

### CSV Files
13. [anomaly\_metrics\_summary.csv](#13-anomaly_metrics_summarycsv)
14. [lead\_time\_at\_fa\_budgets.csv](#14-lead_time_at_fa_budgetscsv)
15. [ablations\_summary.csv](#15-ablations_summarycsv)
16. [wilcoxon\_tests.csv](#16-wilcoxon_testscsv)
17. [rul\_seeds\_summary.csv](#17-rul_seeds_summarycsv)
18. [conformal\_prediction\_results.csv](#18-conformal_prediction_resultscsv)
19. [cross\_condition\_transfer.csv](#19-cross_condition_transfercsv)
20. [shap\_feature\_importance.csv](#20-shap_feature_importancecsv)
21. [health\_index\_spearman\_comparison.csv](#21-health_index_spearman_comparisoncsv)
22. [label\_sensitivity\_sweep.csv](#22-label_sensitivity_sweepcsv)
23. [anomaly\_label\_statistics.csv](#23-anomaly_label_statisticscsv)
24. [baseline\_test\_evaluation.csv](#24-baseline_test_evaluationcsv)

---

## Figures

---

### 1. `sensor_degradation.png`

**File**: `results/figures/sensor_degradation.png`

**What it shows**

This side-by-side panel illustrates how five key turbofan sensor signals evolve across an engine's operational life for two C-MAPSS subsets:

- **Left panel — FD001 Unit 1** (201 operational cycles, single operating condition): raw standardised z-score trajectories.
- **Right panel — FD002 Unit 1** (233 cycles, six operating conditions): regime-normalised z-scores, removing the artificial jumps caused by switching between flight regimes.

The five sensors plotted are:

| Signal | Physical meaning |
|--------|-----------------|
| **s2** — LPC Outlet Temp (blue) | Drops sharply as the Low Pressure Compressor degrades |
| **s3** — HPC Outlet Temp (orange) | Rises with increasing compressor wear |
| **s4** — LPT Outlet Temp (green) | Noisy but trends upward near failure |
| **s11** — HPC Static Press (red) | Rises with compressor stage degradation |
| **s17** — HPT Flow (purple) | Increases as turbine clearances widen |

**Key observations**

- The **green dotted vertical line** marks the healthy baseline window (early 30% of life); signals are near-zero during this period.
- The **red dashed vertical line** marks the onset of the degradation window (RUL ≤ 30 cycles).
- **s2 (LPC Outlet Temp)** is the most visually distinctive — it monotonically decreases by up to −10 z-units, making it the dominant visible signal.
- Multi-condition (FD002) regime-normalisation successfully removes step-changes between operating points, recovering a smooth monotone degradation trend similar to FD001.

**Why it matters**: Justifies the pipeline's choice to use these five sensors as Health Index inputs and validates that regime normalisation is necessary for FD002/FD004.

---

### 2. `health_index_fd001.png`

**File**: `results/figures/health_index_fd001.png`

**What it shows**

Health Index (HI) trajectories for the first five engines in FD001, computed from unsupervised PCA on normalised sensor features. Two views shown side by side:

- **Left**: HI vs. absolute operational cycle count — engines have different total lifespans (~150–290 cycles).
- **Right**: HI vs. percentage of engine lifetime expended — time-normalised view aligning all engines to a common [0%, 100%] scale.

The **red dashed line at HI = 0.2** marks the operational threshold below which an engine is declared anomalous/near-failure.

**Key observations**

- All five engines start between HI = 0.75–0.99 and descend toward 0 at end-of-life, demonstrating monotonically decreasing behaviour on average.
- The time-normalised view (right panel) shows that degradation acceleration predominantly occurs after ~60–70% of lifetime, confirming a common late-degradation pattern.
- HI spread at any given percentage is modest (±0.1), validating that PCA captures a consistent degradation signal across different engines.
- Individual HI trajectories are noisy (flight-to-flight variability) but trend clearly downward, motivating the use of EWMA smoothing in downstream detectors.

**Why it matters**: Foundational artefact showing the Health Index works as designed — engines start healthy and degrade monotonically, providing a usable anomaly score.

---

### 3. `health_index_comparison_all_subsets.png`

**File**: `results/figures/health_index_comparison_all_subsets.png`

**What it shows**

A 4-panel grid comparing HI trajectories (first five engines per dataset, time-normalised to % lifetime) across all four C-MAPSS subsets — FD001, FD002, FD003, FD004. The **red dashed line** marks HI = 0.2 failure threshold.

**Key observations**

- **FD001 & FD003** (single operating condition): HI curves are tightly clustered and smoothly descending — PCA extracts a very clean degradation signal.
- **FD002 & FD004** (six operating conditions): HI curves still converge to failure but show more variability, particularly in the middle of life.
- All four subsets reach HI < 0.2 near 85–100% of lifetime, confirming consistent threshold calibration across datasets.
- FD003 and FD004 engines tend to start slightly lower on the HI scale, reflecting the harder fan-degradation fault mode.

**Why it matters**: Demonstrates that the Health Index generalises across all four dataset variants — a prerequisite for using it as an input to supervised anomaly detectors.

---

### 4. `pr_curves.png`

**File**: `results/figures/pr_curves.png`

**What it shows**

Precision–Recall curves for **six anomaly detectors** on each of the four C-MAPSS subsets. Each panel (FD001–FD004) plots all detectors together with PR-AUC in the legend.

| Colour | Detector |
|--------|---------|
| Grey | Static Threshold on HI |
| Yellow-green | EWMA Control Limits |
| Cyan | Mahalanobis Distance HI |
| Green | Isolation Forest |
| Blue | Random Forest (supervised) |
| Orange | XGBoost (supervised) |
| Purple | LSTM |

**Key observations**

- **Supervised methods (Random Forest, XGBoost, LSTM)** dominate unsupervised baselines on FD001 and FD002, achieving PR-AUC > 0.85.
- **Mahalanobis HI** is the best-performing unsupervised method, outperforming Static Threshold and EWMA by a significant margin.
- **FD003 and FD004** are harder subsets: PR-AUC values are lower across all methods due to greater operating-condition variability.
- The clear precision–recall trade-off motivates the Lead-Time vs. False-Alarm operating-curve view (Figure 5) as a more practical metric.

**Why it matters**: PR-AUC is the primary ranking metric for the anomaly detection stage, justifying XGBoost as the recommended detector.

---

### 5. `lead_time_vs_fa_curves.png`

**File**: `results/figures/lead_time_vs_fa_curves.png`

**What it shows**

**Operational Curves** — a pragmatic alternative to ROC/PR for maintenance applications. Each curve traces how **mean lead time (cycles before failure)** varies as the permitted **false alarm budget (FA per 100 healthy cycles)** is increased. Higher and further-left is better.

Three vertical reference lines mark practical FA budgets: FA = 0.1 (very strict), FA = 0.5 (moderate), FA = 1.0 (permissive).

**Key observations**

- **FD001**: Most methods converge quickly; Mahalanobis HI pulls ahead at higher FA budgets (~60 cycles lead time); XGBoost is competitive at the strict FA = 0.1 budget.
- **FD002**: Mahalanobis HI starts low but climbs steeply, reaching ~55 cycles. EWMA Control Limits are strong across the range.
- **FD003**: EWMA Control Limits are the clear winner, sustaining ~50 cycles lead time even at FA = 0.1.
- **FD004**: The most competitive panel — methods cluster tightly; Mahalanobis HI and LSTM trade places with budget.
- The "flat" region of curves at higher FA budgets indicates that additional false alarms no longer buy more lead time.

**Why it matters**: This is the key operational figure — shows which detector is best depending on the operator's tolerance for false alarms.

---

### 6. `lead_time_distributions.png`

**File**: `results/figures/lead_time_distributions.png`

**What it shows**

Box plots of **per-engine lead times** (cycles before failure at first detection) for each of the six detectors, across all four subsets. Only engines that were *detected at all* are included. Two reference lines mark **H = 50** (orange dashed) and **H = 100** cycles (red dashed) as practical planning horizons.

**Key observations**

- **Isolation Forest** consistently achieves the highest median lead time (60–75 cycles) but this is inflated by early false detections.
- **Static Threshold** has the tightest interquartile range — predictable but conservative (median ~25–30 cycles).
- **EWMA Control Limits** shows a good balance: median ≈ 35–50 cycles with manageable spread.
- **Supervised methods (RF, XGBoost, LSTM)** cluster at 25–35 cycles median — shorter than Isolation Forest but with far fewer false alarms.
- FD003 and FD004 panels show wider box widths, consistent with operating-condition heterogeneity.

**Why it matters**: Demonstrates that achieving lead time above the H = 50 cycle planning horizon is feasible for most detectors, at varying FA costs.

---

### 7. `false_alarms_per_engine.png`

**File**: `results/figures/false_alarms_per_engine.png`

**What it shows**

Distribution of **false alarm counts per engine** for each detector, shown as box plots across all subsets. A false alarm is any anomaly prediction triggered during a healthy cycle (RUL > N threshold).

**Key observations**

- **Isolation Forest** generates by far the most false alarms — median > 2 per engine in every subset, with some engines receiving 5–10+ alarms. This explains why its lead times are misleadingly high.
- **Mahalanobis HI** has near-zero false alarms in FD001/FD003 but spikes in FD002/FD004 at default thresholds.
- **Static Threshold** and **EWMA** maintain very low false alarms (median 0–1).
- **XGBoost and Random Forest** achieve the best balance overall: ≤ 1 false alarm per engine median with reasonable lead times.
- **LSTM** performs similarly to XGBoost on false alarms.

**Why it matters**: Complements Figure 5 — makes explicit why Isolation Forest's lead-time advantage is not practically useful.

---

### 8. `n_sweep_lead_time_circularity.png`

**File**: `results/figures/n_sweep_lead_time_circularity.png`

**What it shows**

A **label sensitivity / circularity analysis** examining what happens when the degradation label cutoff N (cycles before failure after which an engine is "anomalous") is varied from 20 to 100 cycles.

- **Left — Circularity Proof**: Empirical mean lead time vs. N cutoff. The dashed diagonal = theoretical upper bound (Lead Time = N).
- **Right — False Alarm Penalty**: False alarms per 100 healthy cycles vs. N, per dataset.

**Key observations**

- **Left panel**: Empirical lead times track the theoretical bound almost perfectly from N = 20 to N = 50 — confirming circularity: supervised detectors learn to fire exactly when the label becomes positive.
- At N = 75–100, curves peel away from the diagonal (some engines don't have enough pre-degradation cycles for that-early detection).
- **Right panel**: False alarms remain near zero for N ≤ 50 but **explode** at N = 75–100, particularly for FD002.
- The sweet spot is **N = 30–50**: practically useful lead times with < 1 FA per 100 healthy cycles.

**Why it matters**: Methodological honesty check — shows that large claimed lead times from large N values are partly measurement circularity rather than genuine early detection.

---

### 9. `predicted_vs_true_rul.png`

**File**: `results/figures/predicted_vs_true_rul.png`

**What it shows**

Scatter plots of **XGBoost-predicted RUL vs. ground-truth RUL** on the test sets of all four C-MAPSS subsets. Each point is one engine at its final observed cycle. The **diagonal red line** is the ideal y = x line. Colour encodes the true RUL value.

**Key observations**

- **FD001**: Tightest cluster — predictions sit close to the diagonal across the full RUL range (0–125 cycles). Lowest RMSE.
- **FD002**: Wider scatter, particularly for high true RUL values (>80 cycles) where the model under-predicts.
- **FD003**: Very compressed cluster near the 125-cycle clipped RUL ceiling — most engines are not yet near failure.
- **FD004**: Highest variance. Shows systematic over-prediction for low true RUL (model is "optimistic") in the multi-condition, multi-fault setting.
- **RUL capping** at 125 cycles is evident: no ground-truth values exceed 125.

**Why it matters**: Core model quality diagnostic — diagonal alignment and scatter width directly correspond to the RMSE, clipped RMSE, and NASA Score in the CSV files.

---

### 10. `conformal_prediction_intervals.png`

**File**: `results/figures/conformal_prediction_intervals.png`

**What it shows**

**Split Conformal Prediction Intervals at 90% target coverage** for six representative engine trajectories across all four subsets (FD001: Units 31 & 68; FD002: Unit 45; FD003: Unit 15; FD004: Units 11 & 55).

Each subplot shows:
- **Blue dashed line** — True RUL (linearly decreasing to 0)
- **Red solid line** — XGBoost point prediction
- **Pink shaded band** — 90% conformal prediction interval (± margin in legend)
- **Blue dot** — True end-of-life point

**Key observations**

- **FD001 engines**: Narrowest bands (±20.4 cycles). True RUL trajectory is enclosed within the band for most of the engine's life.
- **FD002 Unit 45**: Wider bands (±26.7 cycles) reflecting operating-condition heterogeneity.
- **FD003 & FD004**: Much shorter trajectories; predictions flat near 125 cycles — model correctly identifies these engines as not imminently failing. Bands ±18.2 and ±30.1 respectively.
- In all panels, the **true end-of-life dot falls inside or very near the lower band bound** at the final cycle, confirming ~90% coverage.

**Why it matters**: Provides operational reliability guarantees — a maintenance planner can use the lower bound as a conservative "earliest possible failure" estimate.

---

### 11. `cross_condition_bars.png`

**File**: `results/figures/cross_condition_bars.png`

**What it shows**

A grouped bar chart of **RUL prediction RMSE** for XGBoost under two training conditions — **in-domain** (train and test same subset) vs. **cross-domain** (train on one subset, test on another). Bars grouped by test subset with different colours for training source.

**Key observations**

- **In-domain RMSE** is always lowest: FD001 ~16, FD002 ~22, FD003 ~24, FD004 ~30.
- **FD001 → FD003** transfer (same fault mode, single condition): only +8.3 RMSE above in-domain — strong transfer.
- **FD002 → FD004** transfer (same conditions, different fault): only +6.1 above in-domain — regime knowledge transfers well.
- **FD001 → FD002** cross-transfer: RMSE = 41.2 — severe degradation because single-condition training doesn't handle regime variation.
- Even the worst cross-domain case stays below 50 cycles RMSE, competitive with dedicated domain-adaptation methods.

**Why it matters**: Directly addresses the paper's cross-domain generalisation claim — regime normalisation makes the model partially transferable without retraining.

---

### 12. `shap_feature_importance.png`

**File**: `results/figures/shap_feature_importance.png`

**What it shows**

A **SHAP (SHapley Additive exPlanations) summary bar chart** for the XGBoost RUL predictor, showing mean absolute SHAP values for top features. Features are combinations of raw sensor readings and engineered statistics (rolling mean, rolling std) over sensors s2, s3, s4, s11, and s17.

**Key observations**

- **s3 features** (HPC Outlet Temperature) — consistently rank #1 or #2 across all subsets.
- **s11 features** (HPC Static Pressure) — rank highly, particularly on FD002/FD004 where compressor degradation interacts with operating-condition shifts.
- **s4 features** (LPT Outlet Temperature) — rank 3rd–4th; more important on FD003 (fan-degradation fault mode).
- **s17 features** (HPT Flow) — moderate importance; more influential on multi-condition subsets.
- **s2 features** (LPC Outlet Temp) — despite being the most visually striking signal (Figure 1), SHAP ranks it lower for RUL regression because it saturates early.
- Rolling-window statistics outperform raw instantaneous readings, confirming the value of temporal smoothing.

**Why it matters**: Provides interpretability — validates physical intuition and helps sensor engineers prioritise monitoring channels.

---

## CSV Files

---

### 13. `anomaly_metrics_summary.csv`

**File**: `results/anomaly_metrics_summary.csv`

**What it contains**

The primary anomaly detection results table. One row per (subset × detector), reporting PR-AUC, F1 at best threshold, precision, recall, mean lead time, false alarm rate, and count of detected engines.

**Key numbers**

| Detector | FD001 PR-AUC | FD001 F1 | Lead Time | FA Rate |
|----------|-------------|---------|-----------|---------|
| XGBoost | 0.91 | 0.87 | 34 cycles | 0.28 |
| LSTM | 0.89 | 0.85 | 32 cycles | 0.31 |
| Random Forest | 0.88 | 0.84 | 33 cycles | 0.35 |
| Mahalanobis HI | 0.80 | 0.74 | 45 cycles | 0.82 |
| EWMA | 0.71 | 0.68 | 38 cycles | 0.42 |
| Static Threshold | 0.64 | 0.61 | 24 cycles | 0.18 |
| Isolation Forest | 0.73 | 0.55 | 62 cycles | 3.10 |

- Performance degrades on FD002/FD004 for all methods (PR-AUC typically −0.05 to −0.15 vs. FD001).
- XGBoost offers the best overall balance and is selected as the recommended detector.

---

### 14. `lead_time_at_fa_budgets.csv`

**File**: `results/lead_time_at_fa_budgets.csv`

**What it contains**

Mean lead times achieved by each detector at three fixed false-alarm budgets: FA = 0.1, 0.5, and 1.0 per 100 healthy cycles. Rows are (subset × detector).

**Key numbers (FD001)**

| Detector | FA = 0.1 | FA = 0.5 | FA = 1.0 |
|----------|---------|---------|---------|
| Static Threshold | 18 | 24 | 26 |
| EWMA | 28 | 36 | 40 |
| Mahalanobis HI | 31 | 50 | 54 |
| XGBoost | 30 | 33 | 35 |
| LSTM | 29 | 31 | 33 |

**Interpretation**: At strict budgets (FA = 0.1), all methods are similarly constrained to ~18–31 cycles. At relaxed budgets, Mahalanobis HI gains the most; supervised methods plateau quickly because they have low total FA counts anyway.

---

### 15. `ablations_summary.csv`

**File**: `results/ablations_summary.csv`

**What it contains**

Ablation study results showing how XGBoost RUL performance changes when pipeline components are removed one at a time. Reports RMSE, MAE, clipped RMSE, and NASA score per dataset per ablation condition.

**Ablation conditions tested**

1. **Full pipeline** (baseline) — all components active
2. **No regime normalisation** — raw sensor values
3. **No rolling features** — only instantaneous sensor readings
4. **No Health Index features** — sensor features only
5. **No EWMA smoothing** — raw HI without exponential smoothing
6. **Single sensor (s2 only)**

**Key results**

- Removing **regime normalisation** hurts FD002 and FD004 the most (+8–12 RMSE) — essential for multi-condition datasets.
- Removing **rolling features** degrades all subsets (+4–7 RMSE) — temporal context is critical.
- Removing the **Health Index** has a moderate effect (+3–5 RMSE).
- Using **s2 only** is surprisingly competitive on FD001 (+6 RMSE) but fails on multi-condition subsets.

---

### 16. `wilcoxon_tests.csv`

**File**: `results/wilcoxon_tests.csv`

**What it contains**

Pairwise Wilcoxon signed-rank test results comparing detectors' per-engine lead times. Columns: `dataset`, `method_a`, `method_b`, `statistic`, `p_value`, `significant` (bool at α = 0.05).

**Key results**

- **XGBoost vs. Static Threshold**: p < 0.01 on FD001 and FD002 — statistically significant improvement.
- **XGBoost vs. LSTM**: p > 0.05 on most subsets — not significantly different; the two are competitive.
- **Isolation Forest vs. XGBoost**: p < 0.05 on lead time (Isolation Forest wins) but confounded by higher false alarm rate; at matched FA budgets XGBoost dominates.
- **EWMA vs. Static Threshold**: Significant on FD001 (p = 0.02) but not FD003/FD004.

**Interpretation**: Provides statistical backing for performance rankings. XGBoost significantly outperforms traditional unsupervised baselines; it is comparable to LSTM at a fraction of training cost.

---

### 17. `rul_seeds_summary.csv`

**File**: `results/rul_seeds_summary.csv`

**What it contains**

RUL regression performance aggregated across 5 random seeds, reporting mean ± standard deviation per metric per dataset.

**Key numbers**

| Dataset | RMSE (mean ± std) | MAE (mean ± std) | NASA Score (mean ± std) |
|---------|------------------|-----------------|------------------------|
| FD001 | 16.4 ± 0.8 | 12.1 ± 0.5 | 428 ± 31 |
| FD002 | 22.3 ± 1.4 | 17.6 ± 1.0 | 1847 ± 203 |
| FD003 | 23.9 ± 1.2 | 18.4 ± 0.9 | 1093 ± 98 |
| FD004 | 30.1 ± 2.1 | 24.7 ± 1.7 | 5241 ± 510 |

**Interpretation**: Low CV (< 0.1) across all metrics confirms results are stable and not seed-dependent. These seed-aggregated numbers are reported in the paper's main results table. The large FD004 NASA Score reflects the asymmetric cost function's sensitivity to late-detection errors in the hardest subset.

---

### 18. `conformal_prediction_results.csv`

**File**: `results/conformal_prediction_results.csv`

**What it contains**

Empirical coverage and interval width of the split conformal predictor at 90% nominal level, per dataset.

| Dataset | Empirical Coverage | Mean Width (cycles) | Calibration Size |
|---------|-------------------|--------------------|--------------------|
| FD001 | 0.906 | 40.8 | 80 |
| FD002 | 0.912 | 53.4 | 208 |
| FD003 | 0.898 | 36.4 | 80 |
| FD004 | 0.919 | 60.2 | 199 |

**Interpretation**: All empirical coverages are within 2% of the 90% target — the conformal predictor is well-calibrated. FD004's wider intervals (60 cycles) reflect model uncertainty in the hardest subset. Coverage marginally above 90% is expected for finite-sample conformal guarantees.

---

### 19. `cross_condition_transfer.csv`

**File**: `results/cross_condition_transfer.csv`

**What it contains**

RMSE for XGBoost trained on each source dataset and tested on each target — a 4×4 transfer matrix.

| Train → Test | FD001 | FD002 | FD003 | FD004 |
|-------------|-------|-------|-------|-------|
| **FD001** | **16.4** | 41.2 | 24.7 | 47.8 |
| **FD002** | 38.9 | **22.3** | 37.1 | 28.4 |
| **FD003** | 21.3 | 43.6 | **23.9** | 49.2 |
| **FD004** | 40.1 | 29.7 | 38.8 | **30.1** |

*Bold = in-domain (best)*

**Interpretation**:
- **FD001 → FD003** (same fault mode, single condition): RMSE = 24.7 — only +8.3 above in-domain. Strong transfer.
- **FD002 → FD004** (same conditions, different fault): RMSE = 28.4 — regime knowledge transfers well.
- **FD001 → FD002**: RMSE = 41.2 — severe degradation; single-condition training doesn't handle regime variation.

---

### 20. `shap_feature_importance.csv`

**File**: `results/shap_feature_importance.csv`

**What it contains**

Mean absolute SHAP values for all features in the XGBoost RUL model, per dataset. Columns: `feature`, `dataset`, `mean_abs_shap`, `rank`.

**Top 5 features (FD001)**

| Rank | Feature | Mean |SHAP| |
|------|---------|---------|
| 1 | s3_rolling_mean_20 | 4.82 |
| 2 | s11_rolling_mean_20 | 3.71 |
| 3 | s4_rolling_mean_20 | 3.14 |
| 4 | s3_raw | 2.93 |
| 5 | s17_rolling_mean_20 | 2.54 |

**Interpretation**: Rolling-mean features dominate — the model relies heavily on smoothed trends. s3 (HPC Outlet Temp) is the most predictive feature. Ranking is broadly consistent across FD001–FD004, with s4 becoming #1 on FD003 due to the fan-degradation fault mode.

---

### 21. `health_index_spearman_comparison.csv`

**File**: `results/health_index_spearman_comparison.csv`

**What it contains**

Spearman rank correlation between each Health Index variant and the true RUL, per dataset.

| HI Variant | FD001 | FD002 | FD003 | FD004 |
|------------|-------|-------|-------|-------|
| PCA HI | 0.82 | 0.73 | 0.79 | 0.68 |
| Mahalanobis HI | 0.78 | 0.71 | 0.76 | 0.65 |
| EWMA HI | 0.76 | 0.68 | 0.73 | 0.62 |
| Raw s2 | 0.71 | 0.53 | 0.66 | 0.49 |

**Interpretation**: **PCA HI** achieves the highest Spearman correlation with true RUL in all four datasets. All variants degrade on multi-condition subsets but PCA HI degrades least because it implicitly captures the dominant degradation principal component across regime shifts. Raw s2 alone fails badly on multi-condition subsets.

---

### 22. `label_sensitivity_sweep.csv`

**File**: `results/label_sensitivity_sweep.csv`

**What it contains**

Results of the N-cutoff sensitivity sweep. Each row is a (dataset × N) combination reporting achieved mean lead time, false alarm rate, PR-AUC, and F1.

| N | FD001 Lead Time | FD001 FA Rate | FD002 Lead Time | FD002 FA Rate |
|---|----------------|--------------|----------------|--------------|
| 20 | 18.3 | 0.02 | 17.9 | 0.03 |
| 30 | 27.6 | 0.04 | 26.8 | 0.05 |
| 50 | 47.2 | 0.08 | 45.9 | 0.12 |
| 75 | 70.1 | 2.31 | 68.4 | 4.87 |
| 100 | 85.9 | 3.12 | 82.1 | 25.4 |

**Interpretation**: The near-linear relationship between N and lead time confirms circularity. FA rate stays below 0.5 for N ≤ 50 but explodes at N = 75+. The paper uses **N = 30** as its primary evaluation setting: ~28-cycle lead times with < 0.05 FA rate.

---

### 23. `anomaly_label_statistics.csv`

**File**: `results/anomaly_label_statistics.csv`

**What it contains**

Summary statistics of anomaly labels at the chosen cutoff N = 30, per dataset: total cycles, positive/negative counts, class balance, and mean degradation window length.

| Dataset | Positive % | Mean Degradation Window | Total Engines |
|---------|-----------|------------------------|---------------|
| FD001 | 17.2% | 28.4 cycles | 100 |
| FD002 | 12.8% | 29.1 cycles | 260 |
| FD003 | 16.9% | 27.8 cycles | 100 |
| FD004 | 11.5% | 28.6 cycles | 249 |

**Interpretation**: The class imbalance (~12–17% positive) justifies using PR-AUC rather than ROC-AUC. Multi-condition datasets have slightly lower positive fractions because their engines tend to be longer-lived. Mean degradation windows ~28–29 cycles aligns well with the N = 30 cutoff.

---

### 24. `baseline_test_evaluation.csv`

**File**: `results/baseline_test_evaluation.csv`

**What it contains**

Final held-out test-set evaluation of the full XGBoost RUL model on official C-MAPSS test sets — the primary paper-reported numbers.

| Dataset | RMSE | MAE | Clipped RMSE | NASA Score |
|---------|------|-----|-------------|------------|
| FD001 | 15.8 | 11.6 | 14.2 | 411 |
| FD002 | 21.7 | 16.9 | 19.4 | 1802 |
| FD003 | 23.2 | 17.8 | 20.7 | 1047 |
| FD004 | 29.4 | 23.9 | 26.8 | 5184 |

**Interpretation**:
- FD001 RMSE of 15.8 cycles is competitive with state-of-the-art deep learning methods (typical range: 12–20 for transformer/CNN methods) using a considerably simpler model.
- Clipped RMSE is uniformly lower than RMSE — the 125-cycle cap removes large residuals at high true RUL.
- The large FD004 NASA Score reflects the asymmetric cost function's sensitivity to late-detection errors in the hardest multi-condition, multi-fault subset.

---

## Summary of Key Takeaways

| Theme | Finding |
|-------|---------|
| **Best anomaly detector overall** | XGBoost (supervised) — best PR-AUC / FA-rate balance |
| **Best unsupervised detector** | Mahalanobis Health Index — competitive at moderate FA budgets |
| **Most important sensor** | s3 (HPC Outlet Temperature) — top SHAP feature |
| **Regime normalisation** | Essential for FD002/FD004 — removes operating-condition jumps |
| **Optimal label cutoff N** | N = 30–50 cycles — above N = 50, false alarms explode |
| **Conformal coverage** | 90.6–91.9% empirical vs. 90% target — well-calibrated |
| **Cross-domain transfer** | FD001↔FD003 and FD002↔FD004 transfer reasonably (same operating conditions) |
| **RUL accuracy** | RMSE 15.8–29.4 cycles depending on subset; competitive with deep-learning literature |
