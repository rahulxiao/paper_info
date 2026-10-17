# Master Project Plan: Predictive Maintenance & Anomaly Detection on NASA C-MAPSS

**Document Type:** Master Research Project Plan & Execution Architecture  
**Project Title:** *Predictive Maintenance and Anomaly Detection for Industrial IoT Telemetry Using Machine Learning: A Rigorous Empirical Study on NASA C-MAPSS under Leakage-Free Protocols*  
**Corpus / Codebase Source:** `project/src/`, `project/configs/`, `project/results/`, `paper/ieee_paper.tex`, `project/thesis/`  
**Current Date / Version:** October 2026 / Release Candidate 1.0  
**Grounded In:** Verifiable source code, execution logs (`run.log`), 12 publication figures, LaTeX tables, and the 151-metric traceability manifest (`results/manifest.csv`).

---

## 1. Research Objectives and Core Research Questions

### 1.1 Primary Research Objectives
1. **Establish a Standardized, Leakage-Free Benchmark:** Construct an airtight empirical evaluation protocol on NASA C-MAPSS (`FD001`–`FD004`) ensuring zero data leakage across preprocessing, variance filtering, operating regime normalization, threshold tuning, and conformal calibration.
2. **Dual Diagnostic and Prognostic Evaluation:** Simultaneously evaluate window-level anomaly detection ($F_1$, PR-AUC, ROC-AUC) and Remaining Useful Life (RUL) regression (RMSE, $\mathrm{RMSE}_{\mathrm{clip}}$, MAE, NASA Score) across tree ensembles, deep neural networks, and statistical process control baselines.
3. **Characterize Lead-Time vs. False-Alarm Operating Trade-Offs:** Map 50-point operating characteristic curves under a standardized 3-cycle persistence filter to quantify achievable fleet warning times at controlled industrial false alarm budgets ($0.1, 0.5, 1.0$ FA / 100 cycles).
4. **Empirically Prove Supervised Lead-Time Circularity:** Systematically sweep ground-truth binary label thresholds $N \in \{20, 30, 50, 75, 100\}$ to prove that supervised classifier lead times trivially track the artificial cutoff $N$ rather than physical wear onset.
5. **Advanced Methodological Novelty:** Rigorously evaluate domain shift (cross-condition transferability), uncertainty quantification (split conformal prediction intervals), model interpretability (TreeSHAP vs. statistical control chart consensus), and feature/window preprocessing ablations.

### 1.2 Core Research Questions (RQs) & Empirical Resolutions

| RQ # | Research Question | Methodological Resolution & Findings | Empirical Status |
| :--- | :--- | :--- | :--- |
| **RQ1** | *How accurately can machine learning models identify incipient anomalies and forecast degradation from IIoT sensor telemetry?* | On official test engines, tree ensembles (XGBoost $F_1 = 0.753$--$0.858$) and deep recurrent networks (LSTM $F_1 = 0.690$--$0.827$) achieve high classification fidelity. In RUL regression, LSTM achieves the lowest RMSE on single-regime sets ($14.69$ on FD001, $13.98$ on FD003), while XGBoost matches or outperforms deep learning on complex multi-regime dual-fault sets ($29.60$ on FD004). Isolation Forest fails ($F_1 < 0.16$). | **ANSWERED & VERIFIED** (`results/manifest.csv`) |
| **RQ2** | *How does machine learning empirical performance compare against moving-average and static-threshold control chart baselines?* | While ML dominates window classification ($+0.14$ to $+0.42$ $F_1$ gain over baselines), **the tuned Static Threshold baseline warns earlier than ML detectors under minimal false alarms on multi-fault datasets** (mean lead time $35.8$ cycles on FD003 vs XGBoost $28.1$ cycles, Holm-corrected $p = 0.0264$). This reveals the **Baseline Warning Lead-Time Paradox**. | **ANSWERED & VERIFIED** (Wilcoxon tests in `wilcoxon_tests.csv`) |
| **RQ3** | *What maximum warning lead time is reliably achievable in advance of failure, and how does it depend on the label cutoff $N$?* | Operating curves demonstrate that at controlled budgets ($\mathrm{FA} \le 0.5$ / 100 cycles), lead times of $40$--$55$ cycles are achievable on FD001--FD003, but drop to $15$--$21$ cycles on FD004. The $N$-sweep proves that supervised lead time tracks $N$ almost perfectly ($LeadTime \approx N$). Setting $N > 50$ causes false alarms to surge to $>25$ / 100 cycles and premature alarms to exceed $53\%$. | **ANSWERED & VERIFIED** (`label_sensitivity_sweep.csv`) |
| **RQ4** | *Which data preprocessing, dimensionality reduction, and temporal window lengths achieve the best accuracy-interpretability trade-off?* | Ablations on FD001 show that raw flattened windows perform worst ($\mathrm{RMSE} = 15.67$). PCA (95% variance) achieves the lowest error ($\mathrm{RMSE} = 13.30$). However, **extracting 4 summary statistics (mean, std, slope, last)** yields near-identical accuracy ($\mathrm{RMSE} = 13.38$), superior PR-AUC ($0.935$), and retains full engineering interpretability. $W=30$ is optimal. | **ANSWERED & VERIFIED** (`ablations_summary.csv`) |

---

## 2. Problem Statement, Motivation & Literature Gaps

Industrial cyber-physical systems (gas turbines, jet engines, wind power generators) incur immense financial penalties and safety hazards when critical components undergo unscheduled run-to-failure breakdown. While Industrial Internet of Things (IIoT) telemetry provides continuous high-frequency sensor streams, translating these signals into actionable, dependable maintenance directives faces severe methodological barriers.

### Critical Gaps in Published C-MAPSS Literature:
1. **Pervasive Data Leakage:** Numerous studies fit scalers (`MinMaxScaler`, `StandardScaler`), PCA projections, and operating normalizers across concatenated training and test trajectories, corrupting benchmark validity.
2. **Cherry-Picked Single-Seed Evaluations:** A vast majority of papers report single-seed runs, masking high performance variance and capitalizing on favorable neural weight initializations.
3. **The Unacknowledged Circularity of Supervised Anomaly Detection:** Prior works claim "early warning" capability by reporting alarms triggered near the arbitrarily chosen binary label boundary $RUL \le N$ (typically $N=30$), failing to recognize that the model is merely learning the artificial label boundary rather than true thermodynamic wear onset.
4. **Absence of Strict Statistical Significance Testing:** Literature frequently claims algorithmic superiority based on marginal decimal differences in RMSE without testing against stochastic variance via Wilcoxon signed-rank tests or family-wise error rate corrections.
5. **Overlooked Operational Baselines:** Sophisticated deep learning architectures are rarely benchmarked against tuned Statistical Process Control (Shewhart, EWMA) baselines under identical false-alarm constraints.

---

## 3. Project Scope and Key Contributions

### 3.1 Boundaries of Scope
* **In Scope:**
  * All four canonical subsets of NASA C-MAPSS (`FD001`, `FD002`, `FD003`, `FD004`).
  * Six model families: Gradient-boosted decision trees (XGBoost), Random Forest, PyTorch Deep LSTM, PyTorch 1D-CNN, Isolation Forest, and Statistical Baselines (Static $k$-sigma, EWMA).
  * Unsupervised degradation indicators: PCA-PC1 and Mahalanobis Distance Health Indices.
  * Rigorous experimental design: Quarantined 5-fold `GroupKFold` cross-validation across 5 random seeds (0–4).
  * Advanced Stage 2 evaluations: Cross-condition transferability, split conformal prediction intervals (90% target coverage), TreeSHAP feature attributions, and preprocessing ablations.
  * Dissemination artifacts: 17-page IEEE Transactions manuscript (`paper/ieee_paper.tex`), 8-chapter thesis monograph (`project/thesis/`), 12 publication figures at 300 DPI, and a 151-metric traceability manifest.
* **Out of Scope:**
  * High-frequency acoustic and vibration telemetry (C-MAPSS records cycle-averaged thermodynamic parameters).
  * Edge hardware embedded deployment and real-time streaming ingestion latency benchmarking.
  * Bayesian neural networks / Monte Carlo dropout (split conformal prediction was prioritized for exact finite-sample validity).

### 3.2 Primary Expected & Achieved Contributions
1. **First Unified Leakage-Free Benchmark:** Simultaneous evaluation of diagnostic classification, prognostic regression, lead-time operating curves, conformal intervals, TreeSHAP attributions, and cross-condition transfer under a zero-leakage protocol across all four C-MAPSS subsets.
2. **The Baseline Warning Lead-Time Paradox:** Empirical demonstration and statistical proof that on multi-fault datasets, simple statistical control charts warn earlier than complex ML models under minimal false alarms due to the point-wise truncation of supervised classification loss functions.
3. **Empirical Proof of Supervised Lead-Time Circularity:** Definitive experimental evidence that supervised early warning lead time is an artifact of the ground-truth cutoff $N$, providing guidelines on the operational limits of binary degradation labeling.
4. **Quantified Limits of Cross-Condition Generalization:** Precise quantification showing that unadapted models experience a complete collapse in performance ($F_1$ drops by $\sim 0.70$, RMSE doubles) when deployed across distinct operational regimes.
5. **Rigorous Open-Source Benchmark Suite:** A fully reproducible codebase accompanied by 26 unit/integration tests, one-click shell/batch runners, and a 1:1 metric-to-code traceability manifest.

---

## 4. Work Breakdown by Phases, Milestones, and Evidence

### Phase 1: Dataset Preparation, Ingestion & Leakage-Free Preprocessing
* **Status:** **COMPLETE**
* **Milestone M1:** Ingestion, variance filtering, condition-aware regime normalization, and sliding window construction fully validated.
* **Tasks & Verifiable Evidence:**
  * Download and package all 12 raw C-MAPSS text files into `project/data/raw/` $\to$ *Verified (`Instraction.md`, `project/data/raw/`).*
  * Implement raw file parser with column assignments $\to$ *Verified (`project/src/data_loader.py :: load_raw()`).*
  * Implement train-only sensor variance filter ($\sigma^2 > 0.001$) $\to$ *Verified (`project/src/data_loader.py :: select_features()`).*
  * Design condition-aware $K$-means ($K=6$) operating regime normalizer $\to$ *Verified (`project/src/regime_norm.py`).*
  * Implement piecewise-linear RUL target ($R_{\mathrm{clip}} = 125$) $\to$ *Verified (`project/src/data_loader.py :: add_rul_train()`).*
  * Implement binary anomaly labeler ($N=30$) $\to$ *Verified (`project/src/labels.py :: add_anomaly_labels()`).*
  * Construct temporal sliding windows ($W=30, S=1$) and left-padded test windows $\to$ *Verified (`project/src/data_loader.py`).*
  * Implement 4-statistic window feature extraction (Mean, Std, Slope, Last) $\to$ *Verified (`project/src/features_extractor.py`).*
  * Build unsupervised PCA-PC1 and Mahalanobis Health Indices $\to$ *Verified (`project/src/health_indices.py`).*

### Phase 2: Model Architecture Development & Baseline Engineering
* **Status:** **COMPLETE**
* **Milestone M2:** All six model architectures implemented, tested, and containerized.
* **Tasks & Verifiable Evidence:**
  * Implement XGBoost RUL regressor and anomaly classifier with `scale_pos_weight` $\to$ *Verified (`project/src/models.py`, `ml_anomaly.py`).*
  * Implement Random Forest RUL regressor and classifier $\to$ *Verified (`project/src/models.py`, `ml_anomaly.py`).*
  * Implement PyTorch 2-layer stacked LSTM with early stopping and CosineAnnealing $\to$ *Verified (`project/src/models.py`, `trainer.py`).*
  * Implement PyTorch 1D-CNN temporal convolutional network $\to$ *Verified (`project/src/models.py`).*
  * Implement unsupervised Isolation Forest baseline $\to$ *Verified (`project/src/ml_anomaly.py`).*
  * Implement Static $k$-Sigma multi-sensor voting baseline with OOF grid search $\to$ *Verified (`project/src/baselines_extended.py`).*
  * Implement EWMA dynamic control limit baseline $\to$ *Verified (`project/src/baselines_extended.py`).*

### Phase 3: Stage 1 Benchmark Execution & Statistical Testing
* **Status:** **COMPLETE**
* **Milestone M3:** Quarantined 5-fold GroupKFold across 5 seeds executed; 50-point operating curves and significance tests generated.
* **Tasks & Verifiable Evidence:**
  * Execute 5-fold $\times$ 5-seed cross-validation across all 4 subsets $\to$ *Verified (`project/results/run.log`).*
  * Evaluate window-level classification on official test engines $\to$ *Verified (`project/results/test_anomaly_metrics.csv`).*
  * Evaluate fleet lead time and false alarms over 50 threshold points $\to$ *Verified (`project/results/lead_time_at_fa_budgets.csv`).*
  * Execute $N$-sweep circularity experiments ($N \in \{20, 30, 50, 75, 100\}$) $\to$ *Verified (`project/results/label_sensitivity_sweep.csv`).*
  * Execute paired Wilcoxon signed-rank tests with Holm-Bonferroni correction and 2,000 bootstrap CIs $\to$ *Verified (`project/results/wilcoxon_tests.csv`).*

### Phase 4: Stage 2 Novelty Experiments
* **Status:** **COMPLETE**
* **Milestone M4:** Transfer, conformal prediction, TreeSHAP, and ablation experiments executed and logged.
* **Tasks & Verifiable Evidence:**
  * Execute cross-condition transfer pairs (FD001 $\to$ FD002/004, FD003 $\to$ FD004) $\to$ *Verified (`project/results/cross_condition_transfer.csv`).*
  * Execute split conformal prediction at 90% target coverage $\to$ *Verified (`project/results/conformal_prediction_results.csv`).*
  * Compute TreeSHAP attributions and sensor overlap vs. control charts $\to$ *Verified (`project/results/shap_feature_importance.csv`, `shap_sensor_overlap.csv`).*
  * Execute preprocessing, smoothing, and window-length ablations on FD001 $\to$ *Verified (`project/results/ablations_summary.csv`).*

### Phase 5: Artifact Consolidation, Manifest & Visualization
* **Status:** **COMPLETE**
* **Milestone M5:** 12 publication figures at 300 DPI, LaTeX tables, and 151-metric manifest compiled.
* **Tasks & Verifiable Evidence:**
  * Generate 12 publication-grade figures at 300 DPI $\to$ *Verified (`project/results/figures/`, 12 PNG files).*
  * Export comprehensive LaTeX result tables (Tables I–XI) $\to$ *Verified (`project/results/tables_latex.tex`).*
  * Export Markdown result tables $\to$ *Verified (`project/results/tables_markdown.md`).*
  * Construct 151-metric traceability manifest mapping every paper number 1:1 to source CSVs $\to$ *Verified (`project/results/manifest.csv`).*
  * Write comprehensive narrative results summary $\to$ *Verified (`project/results/results_summary.md`).*

### Phase 6: Academic Manuscript & Thesis Monograph Drafting
* **Status:** **COMPLETE**
* **Milestone M6:** IEEE Transactions manuscript and 8-chapter thesis monograph fully drafted.
* **Tasks & Verifiable Evidence:**
  * Draft IEEE paper Section I (Introduction) $\to$ *Verified (`paper/ieee_paper.tex`, lines 67–242).*
  * Draft IEEE paper Section II (Background Studies) $\to$ *Verified (`paper/ieee_paper.tex`, lines 244–480).*
  * Draft IEEE paper Section III (Dataset & Preprocessing) $\to$ *Verified (`paper/ieee_paper.tex`, lines 482–551).*
  * Draft IEEE paper Section IV (Models, Baselines & Metrics) $\to$ *Verified (`paper/ieee_paper.tex`, lines 553–605).*
  * Draft IEEE paper Section V (Results & Empirical Analysis) $\to$ *Verified (`paper/ieee_paper.tex`, lines 607–1066).*
  * Draft IEEE paper Section VI (Discussion & Limitations) $\to$ *Verified (`paper/ieee_paper.tex`, lines 1068–1118).*
  * Draft IEEE paper Section VII (Conclusion) $\to$ *Verified (`paper/ieee_paper.tex`, lines 1120–1140).*
  * Integrate 30 verified references in IEEE bibliography format $\to$ *Verified (`paper/ieee_paper.tex`, lines 1146–1355).*
  * Draft Thesis Chapters 1–8 and Appendices $\to$ *Verified (`project/thesis/chapters/ch01_introduction.tex` to `ch08_conclusion.tex`).*

### Phase 7: Quality Audit, Consistency Check & Compilation
* **Status:** **IN PROGRESS (Next Actionable Phase)**
* **Milestone M7:** Full numeric verification against manifest, citation DOI audit, and clean PDF compilation under MiKTeX.

---

## 5. Work Classification: Completed, Remaining, and Recommended

### 5.1 Category 1: Completed Work (Verifiably Backed by Artifacts)
1. **Raw Telemetry Pipeline:** 100% complete across all four subsets; variance filtering and regime normalization code fully implemented and tested.
2. **Model Suite:** All 6 model architectures (XGBoost, Random Forest, LSTM, 1D-CNN, Isolation Forest, Static $k$-sigma, EWMA) and 2 unsupervised health indices implemented and benchmarked.
3. **Stage 1 & Stage 2 Experiments:** Full 5-fold $\times$ 5-seed runs, 50-point operating sweeps, $N$-sweep circularity proofs, Wilcoxon hypothesis testing, transfer matrices, conformal calibration, TreeSHAP attributions, and ablations completed.
4. **Visual & Tabular Artifacts:** 12 figures generated at 300 DPI, LaTeX Tables I–XI generated, 151-metric traceability manifest compiled.
5. **Manuscript Drafts:** Full 1,356-line IEEE conference/transactions paper (`paper/ieee_paper.tex`) with all 7 sections written; full 8-chapter thesis monograph drafted.
6. **Code Quality & Testing:** 26 passing pytest tests verifying leakage prevention, data loader shapes, feature extractors, and baseline latches.

### 5.2 Category 2: Remaining Work (Essential Tasks to Finalize Paper)
1. **Numeric Cross-Check Audit:** Systematically verify every scalar in `paper/ieee_paper.tex` against `results/manifest.csv` to ensure zero discrepancies between paper text and code logs.
2. **Figure Caption & Callout Alignment:** Ensure all 12 figures in `project/results/figures/` are accurately cited and described in `paper/ieee_paper.tex` with complete academic captions.
3. **Reference & DOI Audit:** Verify DOIs, conference titles, volume numbers, and URLs for all 30 references in `paper/ieee_paper.tex` and thesis bibliography.
4. **Notation Standardization:** Ensure mathematical notation is uniformly defined across Section III (Preprocessing), Section IV (Models), and Section V (Results).
5. **MiKTeX / PDFLaTeX Compilation:** Execute clean PDF compilation under MiKTeX (`pdflatex ieee_paper.tex && bibtex ieee_paper && pdflatex ieee_paper.tex`), verify Exit Code 0, and inspect output PDF layout, margins, and table formatting.

### 5.3 Category 3: Recommended Improvements (Future Scientific Enhancements)
1. **Non-Exchangeable Conformal Prediction:** The observed under-coverage gap ($73.7\%$--$84.0\%$ vs. $90\%$ target) is caused by test sequence truncation violating exchangeability. Incorporating weighted conformal prediction (Barber et al., 2023) or conformal PID control (Gibbs & Candès, 2021) is recommended as a theoretical follow-up.
2. **Engine-Level Error Breakdown:** Detailed failure mode breakdown comparing HPC wear vs. Fan wear errors on FD003 and FD004 to pinpoint aerodynamic interaction challenges.
3. **Threshold Sensitivity Look-Up Tables:** Tabulating exact operating thresholds $\tau$, lead times, and false alarm rates across discrete industrial operating budgets to aid plant engineers.
4. **Comparison Matrix with Published Deep Learning Literature:** A consolidated benchmark table contrasting this study's leakage-free RMSE against historically cited un-reproducible papers (Babu et al., Li et al., Zheng et al.).

---

## 6. Actionable Milestone & Deliverable Checklist

| Phase | Milestone / Task Description | Target Deliverable | Dependencies | Status |
| :---: | :--- | :--- | :--- | :---: |
| **P1** | Ingestion, Variance Filter & Regime Normalization | `data_loader.py`, `regime_norm.py` | Raw data in `project/data/raw/` | **COMPLETE** |
| **P2** | Implementation of 6 Model Families & 2 Baselines | `models.py`, `baselines_extended.py` | Preprocessing pipeline (P1) | **COMPLETE** |
| **P3** | Stage 1 Benchmarks (5-fold $\times$ 5-seed, OOF Sweeps) | `results/run.log`, `rul_seeds.csv` | Models & Data loader (P1, P2) | **COMPLETE** |
| **P4** | Stage 2 Novelty Experiments (Transfer, Conformal, SHAP) | `stage2_experiments.py`, CSV logs | Stage 1 models (P3) | **COMPLETE** |
| **P5** | Artifact Generation (12 Figs, Tables, 151-Metric Manifest) | `results/figures/`, `manifest.csv` | Experimental CSVs (P3, P4) | **COMPLETE** |
| **P6** | Academic Paper Drafting (Sections I–VII) | `paper/ieee_paper.tex` | Manifest & Figures (P5) | **COMPLETE** |
| **P7** | Master Documentation Package (`docs/*.md`) | 5 Master Documentation Files | Repository files & results | **COMPLETE** |
| **P8** | Numeric Cross-Check Audit (Paper vs. Manifest) | Quality Audit Matrix | `ieee_paper.tex`, `manifest.csv` | **IN PROGRESS** |
| **P9** | Final PDFLaTeX Compilation & Layout Verification | `paper/ieee_paper.pdf` | Quality Audit (P8) | **PENDING** |
| **P10** | Camera-Ready Submission & Archival Tagging | Git Release Tag v1.0 | Final PDF (P9) | **PENDING** |
