Great questions—let me walk through each point carefully, then give a line‑by‑line explanation of the guard evaluators so you can connect the theory to the code you’re running.

---

## 1) Why don’t tree‑based models need scaling? (and why we skipped `SimpleImputer` by default)

**Scaling.** CART/GBDT/Random‑Forest style trees split on one feature at a time by **ordering** observations and choosing a threshold that maximizes an impurity reduction. Any **monotone transform** of a feature (min‑max, standardize, log, etc.) preserves the order, so the optimal split point (in terms of rank) does not change. That’s why trees are effectively **scale‑invariant** and “require little preprocessing” in practice (contrast with distance‑based or linear models). This is a well‑known property discussed in standard references on trees/boosting (e.g., CART and ESL).¹

**Missing values.** The specific estimators we used for guards:

* **`HistGradientBoostingClassifier/Regressor`** (HGB) from scikit‑learn are designed to **handle `NaN`s natively** (no imputation needed). The library’s docs and examples highlight HGB’s native missing‑value handling and its use for interval estimation via quantile loss.² ³

Because of those two properties, the “simple, robust default” is:

* **no scaling** for tree ensembles, and
* **no imputation** when using HGB (*unless* you swap in a model that needs it).

> If you later choose a model that is not scale‑invariant or cannot take `NaN`s (e.g., logistic regression, linear SVR), plug a `StandardScaler`/`SimpleImputer` into the pipeline just for that candidate.

**References (sampling the most load‑bearing ones):**

* Niculescu‑Mizil & Caruana (2005) for evaluating probability estimates (see §3 for calibration & proper losses).
* scikit‑learn example: **Prediction intervals with gradient boosting** (shows HGB with `loss='quantile'`, i.e., quantile/pinball loss for uncertainty/intervals).
* scikit‑learn example: **RF vs. Histogram GB** (user‑guide example; general HGB background). ([Scikit-learn][1])

---

## 2) Why those hyper‑parameter ranges in `_clf_candidates`?

We search broad, conservative ranges that reflect what typically matters for gradient‑boosted trees:

* **learning\_rate**: smaller → slower but more accurate; larger → faster but riskier. Searching roughly **\[0.01, 0.3]** covers the “reasonable” zone for shallow‑to‑medium depth trees.
* **max\_depth / max\_leaf\_nodes**: controls tree complexity per stage (regularization). We let depth vary from **very shallow to medium** to discover bias/variance trade‑offs.
* **l2\_regularization** (and sometimes subsampling): additional regularization knobs. We use a **log‑uniform** style range over orders of magnitude (e.g., 1e‑8–1e‑1).

We then use **RandomizedSearchCV** rather than exhaustive grid search because **random search** explores high‑impact dimensions more efficiently when some hyper‑parameters matter more than others (which is the case here). This is a well‑established result from **Bergstra & Bengio (2012)**. ([Scikit-learn][2])

---

## 3) Why optimize **negative log‑loss** (cross‑entropy) for the classifier guard instead of F1 or accuracy?

The job of the **AIA guard** is not just to predict the sensitive attribute, but to flag points where the sensitive attribute is **predictable with high confidence** from the other features. That is *probability estimation*, not just hard classification.

* **Log‑loss (cross‑entropy)** is a **strictly proper scoring rule** for probabilistic predictions—if you optimize it, the best strategy is to predict **true conditional probabilities** (in expectation). This directly improves **calibration**. See Gneiting & Raftery (2007) and Niculescu‑Mizil & Caruana (2005).
* Metrics like **accuracy or F1** only look at argmax class labels and **ignore probability quality**. You can get a high accuracy with **over‑confident but poorly calibrated** probabilities, which is exactly what we *don’t* want for a guard.

So we tune on `neg_log_loss` to pick the candidate with **the best probability quality**, then set the guard threshold `tau` from the **quantile of predicted confidences** on real data. (If you prefer, you *can* add a second selector on Brier score; it’s also a proper loss.)

---

## 4) Why `loss='quantile'` and **two** regressors for numeric sensitive attributes?

For a numeric sensitive column $S$, what we need is a **predictive interval** for $S\mid X\setminus\{S\}$. Quantile regression lets you estimate conditional quantiles:

* **Quantile loss** (a.k.a. pinball/check loss) directly estimates conditional quantiles $Q_q(Y\mid X)$. Training with $q=0.1$ yields a lower bound; training with $q=0.9$ yields an upper bound. (Koenker & Bassett, 1978; Friedman’s gradient boosting implements this loss.)

* **Two regressors** (low & high) give you a **predictive interval** $[Q_{q_\text{low}}(X), Q_{q_\text{high}}(X)]$. If the model is well specified, this interval has nominal coverage $q_\text{high}-q_\text{low}$.

* Train **one regressor for the lower quantile**, e.g. **$q_{\text{low}}=0.10$**.
* Train **another for the upper quantile**, e.g. **$q_{\text{high}}=0.90$**.
* The pair $[\hat{q}_{\text{low}}(x), \hat{q}_{\text{high}}(x)]$ is a **distribution‑free, asymmetric interval** around the conditional median.

* **Guard logic:** if the **predicted interval width** for the sensitive attribute $S$ is **too small** (below `tau_reg_width`), then $S$ is highly inferable from $X\setminus\{S\}$. You **flag** such rows and reject them in your sampling loop. This is a direct, interpretable uncertainty threshold.

We then use the **predicted interval width** $\hat{q}_{\text{high}} - \hat{q}_{\text{low}}$. If it’s **too narrow**, the sensitive value is *concentrated* (predictable) given the other features → **flag it**. This is standard quantile‑regression thinking (pinball loss / check loss). Core references: **Koenker & Bassett (1978)**; scikit‑learn’s **Prediction Intervals with Gradient Boosting** example.

**References**

* Koenker & Bassett (1978) — foundations of quantile regression.
* Friedman, J. H. (2001/2002) — gradient boosting & robust/quantile losses.
* Gneiting & Raftery (2007) — calibration vs sharpness for predictive distributions.

---

## 5) `_select_best_classifier`: line‑by‑line with rationale

```python
cls_counts = y.value_counts()
minc = int(cls_counts.min())
if minc < 2:
    return None, {'note': 'insufficient class support'}
```

* If the rarest class has < 2 samples, you cannot even form 2 CV folds. Stop early.

```python
cv_splits = _adaptive_cv(y, max_cv=5)
cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
```

* Build **stratified CV** with as many folds as the rarest class allows (≤5), to keep validation stable for imbalanced labels.

```python
cands = _clf_candidates(random_state)
```

* Try a **small, diverse model set**: LogisticRegression (good baseline probs), HistGBClassifier (fast, strong), RandomForest (robust with OHE).

```python
for name, pipe, pdist in cands:
    pipe.set_params(pre=_pre_linear(...) if name=='logreg' else _pre_tree(...))
```

* Choose a preprocessing pipeline suited to the learner (scaling for linear; passthrough + OHE for trees).

```python
full_grid_size = 1
for v in pdist.values(): full_grid_size *= len(v)
n_draws = min(n_iter, full_grid_size)
```

* Cap the number of random draws to the size of the discrete grid to avoid redundant sampling.

```python
search = RandomizedSearchCV(..., scoring='neg_log_loss', cv=cv, ...)
search.fit(X, y)
score = -float(search.best_score_)
```

* Tune hyperparameters under **negative log‑loss**.
  **Why log‑loss and not accuracy/F1?** Because your guard uses **probabilities** to set a threshold $\tau$ (quantile of confidence). Log‑loss is a **strictly proper scoring rule** (Gneiting & Raftery, 2007), so minimizing it encourages **calibrated probabilities**, unlike accuracy/F1 which ignore probability quality. (See also Niculescu‑Mizil & Caruana, 2005.)

```python
best = search.best_estimator_
best_report = {...}
```

* Keep the best pipeline and report metrics.

```python
method = 'isotonic' if (len(y) >= 3000 and y.nunique() <= 20) else 'sigmoid'
calibrated = CalibratedClassifierCV(best, method=method, cv=calib_cv).fit(X, y)
```

* **Calibrate** the chosen model:

  * **Platt scaling** (sigmoid) is robust, low‑variance (Platt, 1999).
  * **Isotonic** is more flexible but needs more data (Zadrozny & Elkan, 2002).
* This step further improves probability reliability before computing the threshold $\tau$.

**References**

* Gneiting & Raftery (2007) — proper scoring rules.
* Niculescu‑Mizil & Caruana (2005) — probability estimation & calibration.
* Platt (1999); Zadrozny & Elkan (2002) — calibration methods.

---

## 6) `_select_best_quantile_interval`: line‑by‑line

```python
pre = ColumnTransformer([('num','passthrough',num_feats),
                         ('cat',OneHotEncoder(...),cat_feats)])
```

* Build a single preprocessor for both families (HGBR/GBR) so they see identical features.

```python
X_tr, X_va, y_tr, y_va = train_test_split(..., test_size=0.2)
Xtr = pre.fit_transform(X_tr); Xva = pre.transform(X_va)
```

* Hold out a **validation** split to evaluate interval quality.

```python
q10, q90 = np.quantile(y, [0.10, 0.90]); iqr = ...
target_cov = q_high - q_low
```

* Compute **IQR** for **scale‑free** width normalization; the target coverage comes from the chosen quantiles.

```python
def eval_pair(low_pred, high_pred):
    width = np.maximum(high_pred - low_pred, 0.0)
    cover = ((yva >= low_pred) & (yva <= high_pred)).mean()
    cov_err = abs(cover - target_cov)
    norm_width = np.median(width) / (iqr + 1e-9)
    return cov_err + 0.25 * norm_width, cover, float(np.median(width))
```

* See §1 above—this balances calibration (coverage) and sharpness (width).

```python
# HGBR candidates: train low and high quantiles with shared hyperparams
low  = HistGradientBoostingRegressor(loss='quantile', quantile=q_low, **params).fit(Xtr, ytr)
high = HistGradientBoostingRegressor(loss='quantile', quantile=q_high, **params).fit(Xtr, ytr)
s, cov, w = eval_pair(low.predict(Xva), high.predict(Xva))
```

* Fit **two** quantile models; evaluate the pair.

```python
# GBR candidates: same idea with GradientBoostingRegressor(loss='quantile')
```

* Provide a second family of quantile learners. Pick the best pair by the score.

```python
report = dict(family=..., score=..., val_coverage=..., val_median_width=..., ...)
return pre, best['low'], best['high'], report, float(iqr)
```

* Return the full bundle used later by the guard.

**References**

* Koenker & Bassett (1978); Friedman (2001/2002); Meinshausen (2006).
* Gneiting & Raftery (2007) — evaluate intervals by coverage & sharpness.

---

* The **classification condition**
  `risky |= (conf > tau) & (gap >= margin_prob)`
  means: flag a row if the guard can **predict the sensitive class with confidence above `tau`**, and the **probability margin between best and second best is ≥ `margin_prob`** (i.e., the prediction is *decisive*, not just slightly above the threshold).
  *Why this helps:* We avoid rejecting rows where the model is only “barely confident” or where many classes are tied—reducing unnecessary rejections while still blocking **highly predictable** cases.

* The **regression condition**
  `risky |= (width <= tau_w)`
  means: the **predictive interval** for the sensitive numeric attribute is **narrower than an absolute threshold** (e.g., 20% of the empirical IQR of $S$). Narrow intervals indicate **low uncertainty / high predictability**—exactly the situations the guard should flag.

```python
def is_high_confidence_v2(df_batch, guards, margin=0.0) -> np.ndarray:
    # thin wrapper returning just the union mask
    risky_union, _, _ = evaluate_guards(
        df_batch, guards, margin_prob=margin, build_table=False
    )
    return risky_union
```

---

## 7) The `train_row_cap` subsampling block: why it speeds things up and the trade‑offs

```python
if train_row_cap is not None and len(X_full) > train_row_cap:
    idx = rng.choice(len(X_full), size=train_row_cap, replace=False)
    X = X_full.iloc[idx].copy()
    y = y_full.iloc[idx].copy()
else:
    X, y = X_full, y_full
```

* **What it does:** If your training set is huge, it **randomly subsamples** up to `train_row_cap` rows **for fitting the guard model only**.

* **Why it is fast:**

  * Tree ensembles and boosting have training cost roughly linear in the number of rows per iteration (plus $\log n$ and feature‑dependent factors). Halving $n$ often **more than halves** wall‑time because I/O, split‑finding, and CV all shrink.
  * You are fitting **separate models per sensitive column**; this compounds the cost. Subsampling keeps each fit bounded.

* **Do we lose information?**

  * Potentially some **small** degradation of the guard model’s accuracy.
  * But you still compute the **threshold** $\tau$ (for classification) and the **interval width** threshold (for regression) **using predictions on the full dataset** `X_full`—that is, after fitting on a subset, you run `predict_proba` (or predict quantiles) on all rows to set data‑aware thresholds. This mitigates most bias in the guard’s **operating point**.
  * If you’re worried about rare classes of $S$, you can make the subsample **stratified by $S$**. A simple tweak:

    ```python
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=train_row_cap, random_state=random_state)
    idx, _ = next(sss.split(X_full, y_full))
    X = X_full.iloc[idx]; y = y_full.iloc[idx]
    ```

* **Why subsampling is reasonable:**

  * For **model selection** and **hyperparameter tuning**, **random search** with fewer data points is often **more efficient** than exhaustive search on full data (Bergstra & Bengio, 2012).
  * Many industrial pipelines use **progressive/early‑stopping** or **subset‑of‑data** tuning to cap time while preserving decision quality.

**References**

* Bergstra, J. & Bengio, Y. (2012). *Random Search for Hyper‑Parameter Optimization.* JMLR.
* Hastie, Tibshirani & Friedman (2009). *Elements of Statistical Learning* — trees’ invariance to scaling; training cost intuition.

---

### How this all ties back to AIA mitigation

* **Classification guards (categorical $S$)**: pick the classifier with the **best probability quality** (log‑loss), calibrate it, compute the **quantile threshold** $\tau$ of max class probability on real data, and **block** synthetic rows whose predicted $\max p(S\mid X\setminus\{S\}) > \tau$ (optionally also require a **margin** between top‑1 and top‑2 to avoid over‑confident ties).

* **Regression guards (numeric $S$)**: fit **predictive intervals** via two quantile regressors, compute an **IQR‑scaled width threshold**, and **block** candidates with **too‑narrow** intervals (i.e., where $S$ is highly inferable).

## 8) References (why these pieces are sound)

* **Proper scoring rules & calibration:**
  *Gneiting & Raftery (2007)* (strictly proper scoring rules; log score/cross‑entropy) and
  *Niculescu‑Mizil & Caruana (2005)* (empirical study on probability estimation & calibration, Platt/Isotonic, log‑loss vs. accuracy).
* **Random search superiority:** *Bergstra & Bengio (2012)* (random search often outperforms grid search in high‑dimensional hyper‑parameter spaces). ([Scikit-learn][2])
* **Quantile regression & intervals:** *Koenker & Bassett (1978)* (regression quantiles) + scikit‑learn’s **Prediction intervals with gradient boosting** example (practical implementation with `loss='quantile'`).
* **Background on histogram‑based gradient boosting:** scikit‑learn user‑guide example comparing RF & HGB. ([Scikit-learn][1])

---

### A couple of practical notes

* If you prefer **Brier score** to cross‑entropy for selection, that’s also a **proper** scoring rule for probabilities; it sometimes behaves more gently with rare classes.
* If a sensitive categorical column is very imbalanced, consider setting class weights in the classifier candidate (we already do that in many defaults) **and** raising the calibration CV to avoid degenerate folds, or fallback to **prefit + calibrate** (as we do).

---

**Footnotes (classic sources without direct web links):**
¹ Breiman, Friedman, Olshen, Stone, *Classification and Regression Trees* (1984); Hastie, Tibshirani, Friedman, *Elements of Statistical Learning*, Ch. 9–10 (trees/boosting). These explain the split mechanism and why monotone transforms of features don’t affect the learned partitions.

[1]: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_hist_grad_boosting_comparison.html?utm_source=chatgpt.com "Comparing Random Forests and Histogram Gradient ..."
[2]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html?utm_source=chatgpt.com "HistGradientBoostingClassifier"



## 1) `_select_best_quantile_interval`: what `eval_pair` does and why

Inside `_select_best_quantile_interval` we define:

```python
target_cov = q_high - q_low

def eval_pair(low_pred, high_pred):
    width = np.maximum(high_pred - low_pred, 0.0)
    cover = ((yva >= low_pred) & (yva <= high_pred)).mean()
    cov_err = abs(cover - target_cov)
    norm_width = np.median(width) / (iqr + 1e-9)
    return cov_err + 0.25 * norm_width, cover, float(np.median(width))
```

**What each line means:**

* `target_cov = q_high - q_low`
  If your models are well‑calibrated conditional quantiles, then
  $\Pr\{Y \in [Q_{q_\text{low}}(X), Q_{q_\text{high}}(X)] \mid X\} \approx q_\text{high}-q_\text{low}$.
  So for $(q_\text{low}, q_\text{high})=(0.1,0.9)$, the nominal coverage is **80%**.

* `width = max(hi - lo, 0)`
  The **interval width** is a proxy for **sharpness** (narrower is “more confident”). We clip at zero to avoid negative widths due to occasional quantile crossing.

* `cover = mean(lo ≤ yva ≤ hi)`
  Empirical **coverage** on the validation set—how often the true $y$ falls inside the interval. This captures **calibration** of the interval (are we getting the promised coverage?).

* `cov_err = |cover - target_cov|`
  Deviation from the nominal coverage. We want this near zero.

* `norm_width = median(width) / IQR`
  A **scale‑free sharpness**: median width normalized by the **interquartile range** of the real target. Using the median makes it robust to a few extremely wide intervals.

* `return cov_err + 0.25 * norm_width`
  A simple **calibration‑plus‑sharpness** score: first minimize coverage error, then (gently) prefer sharper intervals. This mirrors the “**calibration & sharpness**” principle for probabilistic forecasts (Gneiting & Raftery, 2007). If you want a formal, strictly proper scoring rule for intervals, you could replace this with the **Winkler/interval score**; here we use a light‑weight surrogate that works well in practice.

**Why this is useful for your guard:**
For AIA mitigation you want to block candidates for which $S$ is **too predictable** given $X\setminus\{S\}$. Narrow intervals (small `width`) imply **low uncertainty** about $S$; enforcing a minimum width (via `tau_reg_width`) flags those risky rows.

**References**

* Gneiting, T. & Raftery, A. (2007). *Strictly proper scoring rules, prediction, and estimation.* JASA.
* Koenker, R. & Bassett, G. (1978). *Regression quantiles.* Econometrica.
* Meinshausen, N. (2006). *Quantile Regression Forests.* JMLR.

---



This gives you a principled, model‑agnostic filter you can plug into your ε‑rejection sampler.

If you want, I can also show variants that swap the custom `cov_err + λ * width` score for the **interval/Winkler score** from Gneiting & Raftery to make the interval selection step fully “proper” under theory.
