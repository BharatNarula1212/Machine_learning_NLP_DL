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

### Lasso Regression

Same idea as Ridge, but with an **L1 penalty** (sum of *absolute* coefficients):

$$J_{lasso}(\theta) = \frac{1}{2m}\sum(\hat{y} - y)^2 + \lambda \sum_{j=1}^{n} |\theta_j|$$

- **Key trait:** Lasso can push coefficients **exactly to zero** — it deletes useless features. This makes it a **built-in feature selector**.

**Remember:** *Lasso = absolute penalty = zeroes out weak features (automatic selection).*

**Use when:** you suspect many features are useless and want a sparse, simpler model.

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
```

### Elastic Net

The **best of both** — mixes L1 (Lasso) and L2 (Ridge):

$$J_{elastic}(\theta) = \frac{1}{2m}\sum(\hat{y} - y)^2 + \lambda \left( r\sum|\theta_j| + (1-r)\sum \theta_j^2 \right)$$

- **r = `l1_ratio`** controls the blend: r=1 is pure Lasso, r=0 is pure Ridge.
- **Use when:** you have many correlated features *and* want feature selection. Lasso alone struggles with correlated features (picks one at random); Elastic Net handles groups better.

**Remember:** *Elastic Net = Lasso + Ridge blended = selection + stability.*

```python
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
```

### Quick Comparison

| Method | Penalty | Shrinks θ? | Zeroes θ? | Best for |
|---|---|---|---|---|
| Ridge | L2 (squared) | Yes | No | Correlated / many small effects |
| Lasso | L1 (absolute) | Yes | Yes | Sparse models, feature selection |
| Elastic Net | L1 + L2 | Yes | Yes | Correlated features + selection |

---

## 3. Types of Cross-Validation

**Why:** a single train/test split can be lucky or unlucky. **Cross-validation (CV)** tests on multiple splits and averages the scores → a more trustworthy estimate of real-world performance.

| Type | How it works | When to use |
|---|---|---|
| **K-Fold** | Split data into K parts; each part is the test set once, rest is train. Average K scores. | The default for most problems. |
| **Stratified K-Fold** | Like K-Fold but keeps class proportions equal in each fold. | **Classification**, especially imbalanced classes. |
| **Leave-One-Out (LOOCV)** | K = number of samples (test on 1 point at a time). | Tiny datasets; very thorough but slow. |
| **Repeated K-Fold** | Run K-Fold several times with different random splits. | When you want extra-stable estimates. |
| **Time Series Split** | Train on past, test on future (never shuffle time). | **Time-ordered data** (stock, weather, sales). |

**Remember:** *K-Fold = default. Stratified = classification. TimeSeriesSplit = never shuffle time.*

```python
from sklearn.model_selection import cross_val_score, KFold
scores = cross_val_score(model, X, y, cv=5)   # 5-fold CV
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
