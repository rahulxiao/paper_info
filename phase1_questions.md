# Research Project: Questions and Answers for Supervisor Discussion

**Project:** Leakage-Conscious Benchmarking of Predictive Maintenance and Anomaly Detection on the NASA C-MAPSS Dataset

## 1. Data Preprocessing

### Q1. Should we remove constant sensors before or after normalization?

**Answer:** We should identify sensors with very low variance using only the training data, then remove them before model training. This avoids using information from validation or test data during feature selection.

### Q2. Why are we using the first 30% of engine cycles as the healthy baseline?

**Answer:** The project uses the early 30% of training-engine cycles to estimate normal operating behavior. However, we should verify whether these cycles are genuinely healthy across all engines, because degradation might begin earlier for some engines.

### Q3. Is our variance threshold of 0.001 appropriate for every sensor?

**Answer:** It is our current feature-selection rule. We should check whether it removes useful sensor information and whether the results change when we use alternative thresholds.

## 2. Model Training and Validation

### Q4. Is five-fold GroupKFold enough to evaluate our models fairly?

**Answer:** GroupKFold keeps each engine entirely within one fold, preventing the same engine from appearing in both training and validation data. However, it does not automatically prevent every type of leakage or eliminate bias from model selection.

### Q5. Should we use nested cross-validation for hyperparameter selection?

**Answer:** Nested cross-validation would provide a more rigorous estimate when model settings are selected using validation performance. Our current paper identifies the lack of nested outer cross-validation as a possible source of optimistic bias.

### Q6. Should we report results from every random seed or only the average?

**Answer:** We should report the average performance and its variation across seeds. This helps show whether the results are stable or depend heavily on random initialization or data splitting.

## 3. Anomaly Detection

### Q7. Could our anomaly detection model simply be learning the labels we created?

**Answer:** Yes, this is a potential limitation. Our anomaly labels are derived from the RUL cutoff, so a model might learn to recognize the chosen labeling rule rather than detect an independently verified physical fault. Testing multiple cutoff values can help investigate this issue.

### Q8. Why do we apply a three-cycle persistence filter?

**Answer:** The filter requires an alarm score to exceed the threshold for at least three consecutive cycles. This is intended to reduce false alarms caused by temporary fluctuations. We should also measure whether the filter delays genuine warnings.

### Q9. How should we select the anomaly detection threshold?

**Answer:** Our current approach tunes thresholds using out-of-fold training predictions. We should ensure that the official test data are not used for threshold selection and clearly document the selection procedure.

### Q10. What should we do if a control chart performs better than a machine-learning model?

**Answer:** We should report the result honestly. Control charts may be simpler and can sometimes provide earlier warnings or fewer false alarms. The best method depends on the evaluation metric and maintenance requirements.

## 4. RUL Prediction

### Q11. Why do we cap the RUL training labels at 125 cycles?

**Answer:** The current project uses a piecewise linear RUL target capped at 125 cycles. This limits the influence of long early-life periods and focuses learning on degradation closer to failure. We should verify whether this cutoff is appropriate for our experimental objectives.

### Q12. Which metric should we use to compare RUL models?

**Answer:** We should report multiple metrics, including RMSE, MAE, and the NASA scoring function. RMSE emphasizes large errors, MAE summarizes average absolute error, and the NASA score applies an asymmetric penalty to prediction errors.

### Q13. Should we evaluate each C-MAPSS subset separately?

**Answer:** Yes. FD001–FD004 differ in operating conditions and fault complexity. Reporting each subset separately helps identify where a model performs well and where it struggles.

## 5. Uncertainty and Conformal Prediction

### Q14. Why did our conformal prediction intervals fail to reach 90% coverage?

**Answer:** Our paper reports empirical coverage below the nominal 90% target on the official test engines. One possible explanation is a mismatch between calibration and test-engine distributions, but we should investigate other causes before drawing a firm conclusion.

### Q15. Can we claim that our conformal intervals guarantee 90% coverage?

**Answer:** Not based on our current empirical results. Standard split conformal prediction provides a marginal coverage guarantee under exchangeability, but that guarantee may not apply when calibration and test data have different distributions. We should report the observed coverage and explain the assumptions and limitations.

## 6. Results and Research Quality

### Q16. How can we prove that our early-warning results are meaningful?

**Answer:** We can examine the relationship between warning lead time and false-alarm rate, test different decision thresholds, and repeat the experiments with multiple anomaly-label cutoffs. We must also acknowledge that lead time can be influenced by how the labels are constructed.

### Q17. What is the most important limitation of our current study?

**Answer:** Several limitations deserve attention: possible hyperparameter-selection bias, anomaly labels derived from an RUL cutoff, under-coverage of conformal intervals, and limited evidence of generalization beyond the benchmark. We should prioritize them according to their impact on the conclusions.

### Q18. Which additional experiment would strengthen our results the most?

**Answer:** A useful next step would be to strengthen the evaluation protocol by separating model selection from final evaluation, then rerunning the key comparisons. We should also test whether the main findings remain consistent across subsets and random seeds.

### Q19. Can we claim that our model is suitable for real industrial deployment?

**Answer:** Not from C-MAPSS experiments alone. The dataset is simulated, so successful benchmark performance does not establish performance on real industrial equipment. Validation on independent real-world data would be needed.

### Q20. What should we verify before finalizing the paper?

**Answer:** We should check that every reported metric matches the experiment output, that the figures correspond to the correct experiments, that the data splits prevent engine-level overlap, and that limitations are described honestly. We should also verify the implementation of every method described in the paper.

---

## Important Points to Confirm with the Supervisor

- [ ] Whether the current validation protocol needs nested cross-validation.
- [ ] Whether the healthy-baseline assumption is appropriate.
- [ ] Whether the anomaly-label cutoff creates circularity in lead-time evaluation.
- [ ] How to investigate the conformal prediction coverage gap.
- [ ] Which additional experiments are essential before finalizing the results.
- [ ] How strongly the conclusions can be generalized beyond NASA C-MAPSS.

**Note:** These are suggested answers based on the current paper description. They are not substitutes for checking the actual source code, logs, and experimental outputs. Where the evidence is incomplete, the answer deliberately identifies what still needs to be verified.