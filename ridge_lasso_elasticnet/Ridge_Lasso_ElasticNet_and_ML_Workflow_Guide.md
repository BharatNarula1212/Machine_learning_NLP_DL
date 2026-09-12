# Ridge, Lasso, Elastic Net & the ML Workflow — Quick Guide

A short, memorable reference for regularization and the everyday machine-learning workflow. Technical but plain-spoken. Builds on the linear/polynomial regression guides in `../linear_regression_practice/`.

**The one idea to hold onto:** plain regression only cares about *fitting the data*. Regularization adds a second goal — *keep the model simple* — so it doesn't overfit. Everything below is a variation on that theme.

---

## 1. Ridge Regression

### The problem: overfitting

Picture just **2 training points**. A plain linear regression best-fit line will pass **exactly** through both → training error = **0**. Sounds perfect, but it's a trap:

- On the **training** data: accuracy ~100%, error ~0 → **low bias**.
- Add a few **new test** points: the line misses them badly → error jumps → **high variance**.

That gap — great on train, poor on test — is **overfitting**. (Rule of thumb: **100% training accuracy is a warning sign, not a win** — it means the model memorized the training data instead of learning the pattern.)

### The fix: Ridge (a.k.a. L2 Regularization)

Ridge is a tweak to linear regression that **reduces overfitting**. Think of it as a way to *hyperparameter-tune* linear regression so its line can't cling too tightly to the training points.

**How:** take the normal cost (Mean Squared Error) and add a **penalty term** — **λ × (sum of squared slopes)**:

$$J_{ridge}(\theta) = \underbrace{\frac{1}{2m}\sum(\hat{y} - y)^2}_{\text{MSE: fit the data}} + \underbrace{\lambda \sum_{j=1}^{n} \theta_j^2}_{\text{penalty: keep slopes small}}$$

Why this stops overfitting: with 2 points, plain MSE can hit **exactly 0**. But the penalty term $\lambda \sum \theta_j^2$ is only 0 if every slope is 0 — so the total cost is **no longer minimized by the line that threads both points perfectly**. The model is forced to pick a slightly worse-fitting (but more general) line instead.

### The λ ↔ slope relationship (common interview question)

**λ (alpha in sklearn)** is a **hyperparameter** — you choose it. As you turn λ up, the slopes are squeezed down:

| λ | Effect on the model |
|---|---|
| λ = 0 | No penalty → identical to plain linear regression. |
| λ small | Slopes shrink a little; gentle smoothing. |
| λ large | Slopes shrink a lot; flatter, simpler line. |
| λ → ∞ | Slopes approach (but never reach) 0. |

**In one line: as λ increases, the slopes (θ) decrease** — but they **never become exactly zero** (that's the defining trait of Ridge).

### Why shrinking slopes helps — the multi-feature view

A coefficient tells you how strongly a feature moves the output: in `y = 0.52·x₁ + 0.48·x₂ + 0.24·x₃`, moving x₁ by 1 moves y by 0.52. A **big** coefficient = strongly correlated with the output; a **small** one (like x₃'s 0.24) = weakly related.

Ridge shrinks **all** coefficients, but the effect is smartest on the weak ones: pulling x₃'s 0.24 down to ~0.14 barely changes the fit, quietly **reducing the influence of features that aren't really correlated with the output**. That's exactly how it curbs overfitting — it damps the noisy, weakly-related features without dropping any feature entirely.

**Remember:** *Ridge = L2 = squared-slope penalty. ↑λ ⇒ ↓slopes (never 0). Shrinks all coefficients, keeps every feature.*

**Use when:** many features each matter a little, or features are correlated.

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)   # alpha is λ, the penalty strength
```

---

## 2. Lasso and Elastic Net

### Lasso Regression (L1 Regularization)

Lasso looks almost like Ridge — but the penalty uses the **absolute value** of slopes, not the square:

$$J_{lasso}(\theta) = \frac{1}{2m}\sum(\hat{y} - y)^2 + \lambda \sum_{j=1}^{n} |\theta_j|$$

That one change (`|θ|` instead of `θ²`) has a big consequence. Recall from Ridge: as λ increases, slopes shrink but **never hit exactly 0**. With Lasso's absolute-value penalty, as λ increases the slopes shrink and **eventually snap all the way to 0**.

**Why that matters — automatic feature selection:** a coefficient of 0 means that feature contributes nothing (0 × x = 0), so it's effectively **removed** from the model. Lasso drives the coefficients of **weakly-correlated (unimportant) features to zero**, while keeping the important ones. It's feature selection built into the training.

Example — `y = 0.52·x₁ + 0.72·x₂ + 0.034·x₃ + 0.12·x₄`. After Lasso, the weak features (x₃'s 0.034, x₄'s 0.12) get zeroed out; the strong ones stay (shrunk a little). The model now uses only the features that actually matter.

**Remember:** *Lasso = L1 = absolute-slope penalty. ↑λ ⇒ slopes shrink to exactly 0 ⇒ removes weak features (automatic selection).*

**Use when:** you have **many features** (dozens/hundreds) and want the model to automatically drop the useless ones.

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)   # alpha is λ
```

### Elastic Net — the combination of both

Elastic Net simply **combines Ridge + Lasso**, so it fixes **both** problems at once: reduce overfitting (Ridge's job) *and* do feature selection (Lasso's job). It adds both penalty terms, each with its own λ:

$$J_{elastic}(\theta) = \frac{1}{2m}\sum(\hat{y} - y)^2 + \underbrace{\lambda_1 \sum \theta_j^2}_{\text{Ridge: reduce overfitting}} + \underbrace{\lambda_2 \sum |\theta_j|}_{\text{Lasso: feature selection}}$$

In sklearn this is expressed as one `alpha` (overall strength) plus an **`l1_ratio`** (the Lasso-vs-Ridge blend: 1 = pure Lasso, 0 = pure Ridge).

**Use when:** your model is **overfitting AND has lots of features** — especially correlated ones. (Lasso alone gets shaky with correlated features, picking one at random; the Ridge part steadies it.)

**Remember:** *Elastic Net = Ridge + Lasso = fixes overfitting AND selects features.*

```python
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5)   # 50/50 blend of L1 and L2
```

### The Big Idea Tying All Three Together

Ridge, Lasso, and Elastic Net are all ways to **hyperparameter-tune linear regression** — you use plain linear regression first, then reach for one of these when it misbehaves:

| Method | Penalty | Shrinks θ? | Zeroes θ? | Reach for it when… |
|---|---|---|---|---|
| Ridge | L2 (squared) | Yes | **No** | Model is **overfitting** (high train acc, low test acc) |
| Lasso | L1 (absolute) | Yes | **Yes** | You have **many features** and want unimportant ones dropped |
| Elastic Net | L1 + L2 | Yes | **Yes** | **Both** — overfitting *and* too many features |

---

## 3. Types of Cross-Validation

### First — train / validation / test

- **Train set** → the model *learns* on this.
- **Validation set** → carved out of the training data; used to **tune hyperparameters** and check the model as we build it.
- **Test set** → locked away, **never shown** to the model until the very end, to measure real-world performance.

**Why CV exists:** if you split train/validation just once, the result depends on luck — change the `random_state` and you might get 85% one time, 92% another, 78% the next. That's not a number you can trust. **Cross-validation** runs *many* splits, so every point gets a turn in validation, then **averages** the scores → a stable, honest estimate (and you can also report the min/max/average).

### The main types

**1. Leave-One-Out CV (LOOCV)** — validation set = **exactly 1 record**; everything else trains. Repeat until *every* record has been the lone validation point (500 records → 500 experiments).
- ✅ Uses almost all data for training each time.
- ❌ Extremely **slow** on big data, and tends to **overfit** (validation of size 1 is noisy). Rarely used in practice.

**2. Leave-P-Out CV** — same as LOOCV but hold out **P records** at a time instead of 1. P is a hyperparameter (e.g. 10, 20). Even more experiments — mostly of theoretical interest.

**3. K-Fold CV** — the **default**. Split data into **K equal folds**. Each fold takes a turn as validation while the other K−1 train. K experiments, then average.
- Example: 500 records, K=5 → each fold = 500÷5 = **100 records** for validation, 400 for training. Fold 1 validates on records 1–100, fold 2 on 101–200, … 5 experiments cover everything.

**4. Stratified K-Fold** — K-Fold's smarter sibling for **classification**. Plain K-Fold can accidentally put mostly one class in a fold (e.g. a validation fold that's *all* 1s), which teaches the model nothing. Stratified K-Fold forces each fold to keep the **same class proportions** as the full data (e.g. a 60/40 split stays ~60/40 in every fold).

**5. Time Series CV** — for **time-ordered data** (reviews, sales, weather). You must **never shuffle**: always train on the **past** and validate on the **future**. Splits grow forward in time (day 1–4 train → day 5 validate; then 1–5 train → day 6 validate …). Random splitting would leak future info into the past.

### Quick recall

| Type | Validation set | Use when |
|---|---|---|
| Leave-One-Out | 1 record | Tiny data; thorough but slow (overfits) |
| Leave-P-Out | P records | Rarely — theoretical |
| **K-Fold** | 1 of K folds | **Default for most problems** |
| **Stratified K-Fold** | 1 fold, class-balanced | **Classification** (esp. imbalanced) |
| **Time Series** | future block | **Time-ordered data** (never shuffle) |

**Remember:** *K-Fold = default. Stratified = classification. Time Series = never shuffle time. LOOCV = 1-at-a-time, slow.*

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)   # 5-fold CV
print(scores.mean(), scores.std())            # average performance ± spread
```

---

## 4. Cleaning the Dataset

Real data is messy. Clean it *before* modeling — garbage in, garbage out.

| Problem | Common fix |
|---|---|
| **Missing values** | Drop rows/columns, or **impute** (fill with mean/median/mode). |
| **Duplicates** | Remove exact duplicate rows. |
| **Outliers** | Investigate; cap, remove, or transform (only if truly errors). |
| **Wrong types** | Convert (e.g. text "5" → number 5, strings → dates). |
| **Inconsistent categories** | Standardize labels ("NY" vs "New York"). |
| **Different scales** | **Scale** features (StandardScaler / MinMaxScaler) — critical for gradient descent and regularization. |

**Golden rule:** learn cleaning/scaling settings from the **training set only**, then apply them to test data. Fitting on test data = **data leakage** (cheating that inflates your score). A **Pipeline** enforces this automatically.

**Remember:** *Fix missing → dedupe → types → outliers → scale. Fit on train only.*

```python
df = df.drop_duplicates()
df['age'] = df['age'].fillna(df['age'].median())
```

---

## 5. Feature Selection

**Goal:** keep only the features that help. Fewer, better features → simpler, faster, more accurate models that generalize better.

Three families of methods:

| Type | Idea | Examples |
|---|---|---|
| **Filter** | Score each feature on its own, before modeling. | Correlation, chi-square, variance threshold |
| **Wrapper** | Try feature subsets, keep what improves the model. | Recursive Feature Elimination (RFE), forward/backward selection |
| **Embedded** | Selection happens *during* training. | **Lasso** (zeroes features), tree feature importances |

**Remember:** *Filter = cheap & fast. Wrapper = accurate & slow. Embedded = built into the model (Lasso!).*

```python
from sklearn.feature_selection import SelectKBest, f_regression
X_best = SelectKBest(f_regression, k=5).fit_transform(X, y)
```

---

## 6. Model Training

**Training** = the model learns the best parameters (θ) from the data by minimizing the cost function.

The standard flow:

1. **Split** data → train / test (e.g. 80/20).
2. **Fit** the model on the training set (`model.fit(X_train, y_train)`).
3. **Predict** on the test set (`model.predict(X_test)`).
4. **Evaluate** with metrics (R², RMSE, MAE for regression; accuracy, F1 for classification).
5. **Diagnose** — compare train vs test scores:
   - Both bad → **underfitting** (too simple).
   - Train good, test bad → **overfitting** (too complex → add regularization).
   - Both good and close → **healthy fit**.

**Remember:** *Split → Fit → Predict → Evaluate → compare train vs test.*

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
```

---

## 7. Hyperparameter Tuning

**Parameters vs hyperparameters:**
- **Parameters** (θ) — *learned* by the model during training.
- **Hyperparameters** — *set by you* before training (e.g. Ridge's `alpha`, Elastic Net's `l1_ratio`, polynomial `degree`, K in K-Fold).

**Tuning** = searching for the hyperparameter values that give the best cross-validated score.

| Method | How | Note |
|---|---|---|
| **Grid Search** | Try every combination in a grid. | Thorough but slow if the grid is big. |
| **Random Search** | Try random combinations. | Faster; often finds near-best with less compute. |
| **Bayesian / Optuna** | Learn from past tries to pick smarter. | Most efficient for large search spaces. |

**Always tune using cross-validation** so you don't overfit to one test split.

**Remember:** *Parameters are learned; hyperparameters are chosen. Tune them with CV (Grid/Random search).*

```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1, 10]}, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)   # best alpha
```

---

## The Big Picture — How It All Fits Together

```
Clean data → Select features → Split → Train model
                                          ↓
                    Tune hyperparameters (Grid/Random search)
                                          ↓
                         evaluated by Cross-Validation
                                          ↓
                    Regularize (Ridge/Lasso/ElasticNet) to fight overfitting
                                          ↓
                              Evaluate on the test set
```

**One-line summary of the whole guide:**
> Clean your data, keep the useful features, train a model, use regularization + cross-validation to keep it honest, and tune the hyperparameters to get the best version.
