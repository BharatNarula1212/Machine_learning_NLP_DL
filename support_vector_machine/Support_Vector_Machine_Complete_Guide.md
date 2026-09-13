# Support Vector Machine — Complete Guide

A short, memorable reference for Support Vector Machines: the geometry, the math, kernels, and practical implementation. Technical but plain-spoken. Builds on the logistic regression guide in `../logistic_regression/`.

**The one idea to hold onto:** SVM doesn't just draw *a* separating line — it draws the line that leaves the **widest possible gap** on both sides. When the data can't be split cleanly, SVM either tolerates some errors (soft margin) or **reshapes the space** (kernels) until it can.

---

## 1. Introduction: SVC and SVR

SVM solves **both** classification and regression:
- **Support Vector Classifier (SVC)** — classification.
- **Support Vector Regression (SVR)** — regression.

Like logistic regression, SVC separates classes with a line (2D), plane (3D), or **hyperplane** (n-D). But SVM adds one more idea: alongside that best-fit line, it also draws two parallel **marginal planes** — one touching the closest point of each class — and picks the fit that makes the **gap between them as wide as possible**.

- The points that lie exactly on the marginal planes (the closest ones to the boundary) are called **support vectors** — they're the only points that actually determine where the line goes.
- Wider gap = more confident, more generalizable separation.

**Remember:** *SVM = best-fit line + widest possible margin on both sides, defined only by the closest points (support vectors).*

---

## 2. Soft Margin vs. Hard Margin

Real-world data is rarely perfectly separable — classes overlap.

- **Hard margin:** the two classes are cleanly separable; the margin has **zero errors**. Rare in practice.
- **Soft margin:** classes overlap, so the model **tolerates some misclassified/borderline points** inside or across the margin, in exchange for a much better overall fit. This is what real datasets need.

**Remember:** *Hard margin = zero tolerance, only works on perfectly separable data. Soft margin = allows some errors, works in the real world.*

---

## 3. The Math: Hyperplane, Margins, and the Optimization Goal

### The hyperplane equation

$$w^Tx + b = 0$$

`w` is a vector **perpendicular** to this hyperplane. For any point, plugging it into `w^Tx + b` tells you which side it's on:
- Result **> 0** → point is above the plane.
- Result **< 0** → point is below the plane.

### The two marginal planes

$$w^Tx + b = +1 \quad \text{(upper margin)} \qquad w^Tx + b = -1 \quad \text{(lower margin)}$$

### The distance between them — what we want to maximize

Subtracting the two equations and normalizing (dividing by `‖w‖`, the magnitude of w, to get a proper unit distance) gives the margin width:

$$\text{margin width} = \frac{2}{\|w\|}$$

**The objective:** maximize $\frac{2}{\|w\|}$ by adjusting `w` and `b` — equivalently (and more conventionally in ML), **minimize** $\frac{\|w\|}{2}$, since minimizing is the standard direction we optimize in.

### The constraint — correct classification

For every correctly classified point:

$$y_i \cdot (w^Tx_i + b) \geq 1$$

Why this works: label $y_i = +1$ for points above the plane (where $w^Tx+b \geq 1$) and $y_i = -1$ for points below (where $w^Tx+b \leq -1$). Multiplying same-signed values always gives a positive result ≥ 1 — so this single inequality elegantly enforces "every point is on its correct side, with margin."

**Remember:** *Objective: minimize ‖w‖/2 (maximize the margin). Constraint: yᵢ(w^Tx+b) ≥ 1 for every correctly classified point.*

---

## 4. The SVC Cost Function (with Soft Margin / Hinge Loss)

The pure hard-margin objective only works for perfectly separable data. For real (overlapping) data, add a penalty for violations — this is the **hinge loss**:

$$J(w,b) = \frac{\|w\|}{2} + C\sum_{i=1}^{n}\zeta_i$$

Two new pieces:

- **C** — a hyperparameter controlling **how many misclassifications you're willing to tolerate**. Small C → very tolerant (wider margin, more errors allowed). Large C → strict (narrower margin, fewer errors allowed, but risk of overfitting).
- **ζᵢ (zeta / slack variable)** — for each misclassified or margin-violating point, this is its **distance from the marginal plane it should have been correctly on**. The sum $\sum \zeta_i$ totals up how "bad" all the violations are.

**Remember:** *Hinge loss = ‖w‖/2 (maximize margin) + C × total slack (penalize violations). C is the knob between wide-margin-tolerant and strict-narrow.*

---

## 5. Support Vector Regression (SVR)

SVR reuses the same margin idea, but instead of separating classes, it tries to fit **most points within a tolerance tube** around the regression line.

### The three lines

$$w^Tx + b \quad \text{(the fit line)} \qquad w^Tx + b + \epsilon \quad \text{(upper tube)} \qquad w^Tx + b - \epsilon \quad \text{(lower tube)}$$

- **ε (epsilon)** — the **margin of error** you're willing to accept: how far a prediction can be from the actual line and still count as "close enough" (i.e., inside the tube).

### The constraint and the slack for outliers

For points that land inside the tube, the error is automatically ≤ ε — no penalty needed. But some points will inevitably fall **outside** the tube. For those, we add:

- **ζᵢ (zeta)** — the extra distance a point falls *beyond* the ε-tube.

The full cost function:

$$J(w,b) = \frac{\|w\|}{2} + C\sum_{i=1}^{n}\zeta_i, \quad \text{s.t. } |y_i - (w^Tx_i+b)| \leq \epsilon + \zeta_i$$

- **C** again controls the tradeoff — how much you penalize points that stray outside the ε-tube. As C increases, the model tolerates less error (loss decreases as C grows, up to a point).

**Remember:** *SVR fits a "tube" of width 2ε around the line. ε = built-in tolerance (no penalty inside it). ζ = extra penalty for points that fall outside the tube. C tunes how strict that penalty is.*

---

## 6. SVM Kernels — Handling Non-Linearly-Separable Data

**The problem:** sometimes classes are hopelessly tangled in their original dimension — no straight line or plane can separate them, no matter how you draw it.

**The trick:** apply a **transformation** that projects the data into a **higher dimension**, where it *does* become linearly separable. Once transformed, a plain linear SVC can slice right through it.

> **Simple example:** 1D points where the classes alternate along a line (can't be split by a single point). Apply the transformation `y = x²` to create a second axis. Now plotted in 2D (x, y), the two classes separate into a clean "outer" and "inner" group — a straight line can now divide them.

This is exactly how SVM kernels work — different kernels use different transformation formulas:

| Kernel | Idea |
|---|---|
| **Linear** | No transformation — use when data is already linearly separable. |
| **Polynomial** | Projects into new dimensions using polynomial combinations (e.g. x₁², x₂², x₁·x₂). |
| **RBF (Radial Basis Function)** | Uses a distance-based exponential formula; effectively "lifts" clustered points upward, letting a flat plane slice below them. The default and most versatile choice for unknown-shaped data. |
| **Sigmoid** | Uses a sigmoid-shaped transformation, similar in spirit to logistic regression's activation. |

**Remember:** *If data isn't linearly separable, don't panic — transform it into a higher dimension with a kernel until it is, then draw a straight line there.*

```python
from sklearn.svm import SVC
model = SVC(kernel='rbf')   # try 'linear', 'poly', 'rbf', 'sigmoid'
```

---

## 7. SVC Implementation — Practical Workflow

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Start simple: if data looks linearly separable on a scatter plot, try linear first
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(classification_report(y_test, y_pred))
```

**Key lesson from practice:** cleanly-separable data gets near-100% accuracy with `kernel='linear'`. The moment classes overlap, linear accuracy drops sharply — that's your cue to try `'rbf'`, `'poly'`, or `'sigmoid'` and compare.

### Hyperparameter tuning with GridSearchCV

```python
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(), param_grid, cv=5, refit=True, verbose=3)
grid.fit(X_train, y_train)
grid.best_params_          # the winning combination
```

- **`C`** and **`gamma`** (kernel coefficient, mainly for RBF/poly/sigmoid) are the main knobs to search alongside `kernel`.
- `.coef_` is only available for the **linear** kernel (a straight line has coefficients; a curved kernel-transformed boundary doesn't).

**Remember:** *Always compare kernels — don't assume linear works. Then GridSearchCV over C, gamma, and kernel together for the best combination.*

---

## 8. SVM Kernels Implementation — Seeing the Transformation Directly

You can **manually** build the features a kernel would create, to see the trick explicitly:

1. Take overlapping 2D data (e.g. one class forming an inner circle, another an outer ring).
2. Manually engineer new features: `x1²`, `x2²`, `x1*x2` (this is what the **polynomial kernel** does internally).
3. Plot these three new features in 3D — the two classes now separate cleanly into distinct clusters.
4. Fit a plain **linear** SVC on these engineered features → near-perfect accuracy.

**The punchline:** doing this manually and using `kernel='poly'` directly give the **same result** — the kernel is just automating that feature engineering for you. The same logic applies to `'rbf'` (lifts clustered points into a new dimension via a distance-based exponential formula) and `'sigmoid'`.

**Remember:** *A kernel is just automated feature engineering into a higher dimension — you can replicate simple ones by hand (e.g. squaring features) to build intuition, but let the kernel parameter do it in practice.*

---

## 9. SVR Implementation — Practical Workflow (with Feature Encoding)

Real datasets mix numeric and categorical columns, so **encoding** comes before modeling:

- **Label Encoding** — for **binary** categorical features (e.g. male/female, yes/no). Replaces the two categories with 0/1.
- **One-Hot Encoding** — for categorical features with **more than 2** categories (e.g. 4 days of the week). Creates one column per category; use `drop='first'` to avoid redundancy (N categories only need N−1 columns).

**Critical rule — avoid data leakage:** always **split train/test first**, then `fit_transform` encoders on the **training set only**, and just `transform` (no re-fitting) on the test set. The model must never learn anything from test data.

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Label-encode binary columns: fit_transform on train, transform only on test
le = LabelEncoder()
X_train['sex'] = le.fit_transform(X_train['sex'])
X_test['sex']  = le.transform(X_test['sex'])

# One-hot encode a multi-category column via ColumnTransformer
ct = ColumnTransformer([('onehot', OneHotEncoder(drop='first'), [day_col_index])], remainder='passthrough')
X_train = ct.fit_transform(X_train)
X_test  = ct.transform(X_test)

# Fit SVR and tune with GridSearchCV
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
print(r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred))

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'linear', 'poly']}
grid = GridSearchCV(SVR(), param_grid, refit=True)
grid.fit(X_train, y_train)
```

**Remember:** *Encode binary columns with LabelEncoder, multi-category columns with OneHotEncoder(drop='first'). Fit on train only, transform on test, to avoid data leakage. Then GridSearchCV over C/gamma/kernel as usual.*

---

## Summary Table

| Topic | One-line takeaway |
|---|---|
| SVC intuition | Best-fit line + widest possible margin, defined by support vectors |
| Soft vs Hard margin | Hard = zero errors (rare); Soft = tolerates overlap (realistic) |
| Core math | Minimize ‖w‖/2 (maximize margin) subject to yᵢ(w^Tx+b) ≥ 1 |
| Cost function (hinge loss) | ‖w‖/2 + C·Σζᵢ — C trades off margin width vs. tolerated errors |
| SVR | Fits an ε-wide tolerance tube instead of a single line; ζ penalizes points outside it |
| Kernels | Transform data into higher dimensions to make the non-separable separable |
| Kernel types | Linear (no transform), Polynomial, RBF (default, versatile), Sigmoid |
| SVC implementation | Try linear first; if accuracy is poor, sweep kernels + GridSearchCV on C/gamma |
| SVR implementation | Encode categoricals correctly (train-only fit) before modeling; tune the same way |

### Where to Go Next
- **Regularization intuition (C parameter)** — conceptually parallels λ in Ridge/Lasso (see `../ridge_lasso_elasticnet/`).
- **One-vs-One / One-vs-Rest** — extending SVC to multi-class problems (see the OVR section in `../logistic_regression/`).
- **Precision/Recall/ROC-AUC** — the same classification metrics apply to SVC's predictions.
- **Decision Trees, Random Forests** — non-linear alternatives that don't require kernel tricks.
