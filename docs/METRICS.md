# Metrics Reference

This document summarizes the metrics used in the pipeline and points to their implementation locations.

> **Note**
> "Higher is better" or "Lower is better" is stated for each metric as used in the paper and in the evaluation code.

## Fidelity (Resemblance)

### Categorical Marginals
- **Jensen-Shannon (JS) distance**: lower is better.
  - Implementation: `src/data_evaluator/univariate_evaluator.py` (`UnivariateEvaluator.jensen_shanon_divergence_categorical`).

### Numerical Marginals
- **Absolute Cohen's d**: lower is better.
  - Implementation: `src/data_evaluator/univariate_evaluator.py` (`UnivariateEvaluator.cohen_s_d`).

### Multivariate Structure
- **Pearson correlation matrix (numeric-numeric)**:
  - Metric: Frobenius norm of the real-synthetic difference (lower is better), plus Spearman rank correlation of the upper triangle (higher is better).
  - Implementation: `src/data_evaluator/multivariate_evaluator.py` (`evaluate_pearson_correlation_matrix_diff`).
- **Cramer's V matrix (categorical-categorical)**:
  - Metric: Frobenius norm of the real-synthetic difference (lower is better), plus Spearman rank correlation (higher is better).
  - Implementation: `src/data_evaluator/multivariate_evaluator.py` (`evaluate_cramerV_correlation_matrix_diff`).
- **Correlation ratio η^2 (categorical-numeric)**:
  - Metric: Frobenius norm of the real-synthetic difference (lower is better), plus Spearman rank correlation (higher is better).
  - Implementation: `src/data_evaluator/multivariate_evaluator.py` (`evaluate_correlation_ratio_matrix_diff`).

### Mode Collapse (Categorical Support)
- **Missing categorical levels**: lower is better.
  - Implementation: `src/data_evaluator/univariate_evaluator.py` (`UnivariateEvaluator.get_mode_collapse`, `evaluate_mode_collapse_values`).
  - Used by: `FineTuningGenerationTask` for mode-patching (`src/data_synthesizer/pipeline/generation_task.py`).

## Utility
- **TSTR vs TRTR**: compare models trained on synthetic vs real data when tested on real data.
  - Implementation: `src/data_evaluator/utility_evaluator_new.py` (`train_synthetic_test_real`, `train_real_test_real`).
  - Metrics: accuracy, balanced accuracy, weighted F1, ROC AUC, log loss, and others (see `train_test`).
- **UtilityGap**: absolute gap between TRTR and TSTR (lower is better). This is reported in the paper and in Optuna studies.
  - Related code: `src/data_evaluator/utility_evaluator_new.py` and Optuna scripts under `experiments/scripts/optuna/`.

## Privacy

### Attribute Inference (AIA)
- **Classification**: accuracy, balanced accuracy, weighted F1, macro F1 (lower attacker performance is better for privacy).
- **Regression**: MSE / RMSE / R^2 (lower R^2 and lower accuracy are better for privacy).
  - Implementation: `src/data_evaluator/privacy_evaluation/privacy_evaluator.py` (`evaluate_attribute_synthetic_prediction`, `evaluate_attribute_real_prediction`).

### εANY Risk (HEOM-kNN)
- **εANY**: fraction of synthetic records within any real record's kNN radius (lower is better).
  - Implementation: `src/data_synthesizer/privacy_sampling.py` (`get_epsilon_any`, `get_epsilon_heom_any*`).
  - Used by: `SamplingAndRejectTask` with `dissimilarity_type="heom-knn-any"` (`src/data_synthesizer/pipeline/generation_task.py`).

### DCR / CAP (SDMetrics)
- **Distance to Closest Record (DCR)**: larger distances imply lower memorization risk (higher is better for privacy).
  - Implementation: `sdmetrics.single_table.DCROverfittingProtection` in scripts such as:
    - `experiments/scripts/credit/disclosure_protection/credit_ctgan_dcr.py`
- **Correct Attribution Probability (CAP)**: lower is better for privacy; reported as `cap_protection` in SDMetrics.
  - Implementation: `sdmetrics.single_table.DisclosureProtection` in scripts such as:
    - `experiments/scripts/adult/disclosure_protection/adult_ctgan.py`

### RPR (Relative Proximity Ratio)
- **RPR**: compares proximity of synthetic records to training vs test records (closer to 50% is better).
  - Computed in privacy-analysis notebooks under `experiments/pipeline_experiment/`.
  - Source notebooks include `epsilon_comparison_privacy_evaluation.ipynb`.

### Anonymeter Attacks
- **Singling-out and linkability**: lower attack success is better.
  - Implementation: `src/data_evaluator/privacy_evaluation/privacy_evaluator_anonymeter.py` and `PrivacyAnonymeterEvaluationTask` (`src/data_synthesizer/pipeline/evaluation_task.py`).
