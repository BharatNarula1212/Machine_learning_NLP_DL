# Logistic Regression — Complete Guide

A short, memorable reference for logistic regression: the math, why it's needed, multi-class handling, performance metrics, and hyperparameter tuning. Technical but plain-spoken. Builds on the linear regression guides in `../linear_regression_practice/` and the regularization guide in `../ridge_lasso_elasticnet/`.

**The one idea to hold onto:** logistic regression is linear regression's best-fit line, **squashed** into the range [0, 1] with the sigmoid function, so it can answer yes/no questions instead of predicting numbers.

---

## 1. Why Not Just Use Linear Regression?

**The problem:** logistic regression solves **binary classification** — the output is a category (pass/fail, spam/not-spam), not a number. Naturally you'd ask: can't we just fit a line and say "above 0.5 → pass, below 0.5 → fail"?

**Try it and two things break:**

1. **Outliers wreck the line.** Add one unusual training point (e.g. someone who studied 12 hours) and the whole best-fit line tilts to accommodate it — flipping predictions for points that were correctly classified before. A classifier shouldn't be this fragile to one weird data point.
2. **The output isn't bounded.** A straight line keeps going forever — it can predict values **way above 1** or **below 0**. But a "probability of passing" only makes sense between 0 and 1.

**The fix:** don't let the line extend forever — **squash** it so the output always lands between 0 and 1. That squashing is exactly what logistic regression adds on top of the linear equation.

**Remember:** *Linear regression's line is unbounded and outlier-sensitive. Classification needs output confined to [0, 1] — that means squashing, not extending.*

---

## 2. The Math: Sigmoid + the Log-Loss Cost Function

### Step 1 — Squash with the Sigmoid function

$$\text{sigmoid}(z) = \frac{1}{1 + e^{-z}}$$

Feed in *any* real number z (positive, negative, huge, tiny) and the sigmoid always returns a value between **0 and 1** — that's the squashing. Key facts:

- **z > 0 → sigmoid(z) > 0.5** (leans toward class 1).
- **z < 0 → sigmoid(z) < 0.5** (leans toward class 0).
- **z = 0 → sigmoid(z) = 0.5** exactly (the decision boundary).

### Step 2 — The logistic regression hypothesis

Take the familiar linear equation and run it through sigmoid:

$$h_\theta(x) = \text{sigmoid}(z), \quad \text{where } z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots$$

So training still fits a straight line (or plane) — sigmoid just bends its output into a valid probability.

### Step 3 — Why we can't reuse the linear regression cost function

Plugging this sigmoid-based $h_\theta(x)$ directly into the old Mean Squared Error cost function seems natural, but it creates a **non-convex** cost surface — full of bumpy **local minima**. Gradient descent can get stuck in one of those bumps (slope = 0) and never reach the true best answer (the **global minimum**).

### Step 4 — The fix: Log Loss

Instead, logistic regression uses **log loss** (a.k.a. binary cross-entropy), which *is* convex — one smooth bowl, one global minimum:

$$\text{Cost}(h_\theta(x), y) = \begin{cases} -\log(h_\theta(x)) & \text{if } y = 1 \\ -\log(1 - h_\theta(x)) & \text{if } y = 0 \end{cases}$$

Combined into one formula (only one branch is ever "active" per row — whichever `y` isn't 1 or 0 zeroes out the other term):

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i \log(h_\theta(x_i)) + (1-y_i)\log(1-h_\theta(x_i))\Big]$$

Then it's the same convergence loop as linear regression — repeatedly nudge θ downhill via gradient descent until the cost stops improving.

**Remember:** *Sigmoid squashes the line into [0,1]. Plain MSE on that gives a bumpy (non-convex) cost — use Log Loss instead, which is convex and gradient-descent friendly.*

---

## 3. Performance Metrics for Classification

### The Confusion Matrix — the foundation for everything else

A 2×2 grid (for binary classification) comparing **actual** vs **predicted**:

| | Predicted 1 | Predicted 0 |
|---|---|---|
| **Actual 1** | True Positive (TP) | False Negative (FN) |
| **Actual 0** | False Positive (FP) | True Negative (TN) |

- **TP / TN** = correct predictions (the diagonal).
- **FP** = predicted 1, actually 0 ("false alarm").
- **FN** = predicted 0, actually 1 ("missed it").

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**The trap:** on an **imbalanced dataset** (e.g. 900 ones, 100 zeros), a model that blindly always predicts "1" scores 90% accuracy while learning nothing. **Never trust accuracy alone on imbalanced data.**

### Precision — "out of everything I flagged, how much was right?"

$$\text{Precision} = \frac{TP}{TP + FP}$$

Precision punishes **false positives**. Use it when a false alarm is the costly mistake.
> **Example — spam filter.** A real (non-spam) email wrongly flagged as spam (FP) might get lost forever. That's the expensive error → optimize **precision**.

### Recall — "out of everything that was actually true, how much did I catch?"

$$\text{Recall} = \frac{TP}{TP + FN}$$

Recall punishes **false negatives**. Use it when *missing* a positive case is the costly mistake.
> **Example — disease diagnosis.** Telling a diabetic patient "you're fine" (FN) is dangerous — they won't seek treatment. That's the expensive error → optimize **recall**. (Wrongly telling a healthy patient they might be sick (FP) is a much smaller cost — they just get retested.)

### F-beta Score — balancing precision and recall

$$F_\beta = (1+\beta^2)\cdot\frac{\text{Precision}\times\text{Recall}}{\beta^2\cdot\text{Precision}+\text{Recall}}$$

- **β = 1 (F1 score):** precision and recall matter **equally**. This is the harmonic mean of the two.
- **β = 0.5:** weights **precision higher** than recall (false positives matter more).
- **β = 2:** weights **recall higher** than precision (false negatives matter more).

**Remember:** *Accuracy lies on imbalanced data. Precision ⇒ minimize false alarms (spam). Recall ⇒ minimize missed cases (disease). F1 ⇒ balance both.*

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
cm = confusion_matrix(y_test, y_pred)
```

---

## 4. Multi-Class Classification: One-vs-Rest (OVR)

Logistic regression is naturally binary (two classes). For **3+ classes**, the **One-vs-Rest (OVR)** strategy trains **one binary classifier per class**:

- **Model 1:** "Is it class A, or NOT class A (i.e. B or C combined)?"
- **Model 2:** "Is it class B, or NOT class B?"
- **Model 3:** "Is it class C, or NOT class C?"

**At prediction time:** feed the new data point into *all* models. Each spits out a probability (e.g. Model 1 → 0.25, Model 2 → 0.20, Model 3 → 0.55). **Pick the class whose model gave the highest probability** — here, Model 3 wins, so the prediction is class C.

**Remember:** *N classes ⇒ N binary models, each asking "this class vs everyone else." Highest probability wins.*

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='ovr')
```

---

## 5. Handling Imbalanced Datasets: `class_weight`

An **imbalanced dataset** (e.g. 9,846 of class 0 vs 154 of class 1) makes the model lazy — it can get high accuracy by mostly ignoring the minority class.

**The fix:** the `class_weight` parameter tells the model to pay **extra attention** to the minority class by penalizing its misclassifications more heavily during training.

- `class_weight='balanced'` → sklearn automatically weights classes inversely to their frequency (equal importance).
- `class_weight={0: 1, 1: 10}` → a custom dictionary, e.g. "treat every class-1 mistake as 10× worse than a class-0 mistake." Tune this like any other hyperparameter (via Grid/Randomized Search).

**Remember:** *Imbalanced data ⇒ use `class_weight` to force the model to take the minority class seriously.*

```python
model = LogisticRegression(class_weight={0: 1, 1: 10})
```

---

## 6. Hyperparameter Tuning: GridSearchCV vs RandomizedSearchCV

Key logistic regression hyperparameters to tune: **`penalty`** (l1/l2/elasticnet — see the Ridge/Lasso guide), **`C`** (inverse of λ — smaller C = stronger regularization), **`solver`** (the optimization algorithm), and **`class_weight`**.

### GridSearchCV — exhaustive search

Tries **every single combination** of the given parameter values. Thorough, guaranteed to find the best combo *within the grid* — but slow, since the number of combinations multiplies fast.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
params = {'penalty': ['l1', 'l2', 'elasticnet'], 'C': [100, 10, 1.0, 0.1, 0.01], 'solver': [...]}
grid = GridSearchCV(LogisticRegression(), params, scoring='accuracy', cv=StratifiedKFold(n_splits=5), n_jobs=-1)
grid.fit(X_train, y_train)
grid.best_params_, grid.best_score_
```

### RandomizedSearchCV — sampled search

Instead of trying every combination, it tries a **random sample** of them. Much faster with large search spaces, and often lands on a similarly good (if not identical) combination — a good tradeoff when you have many hyperparameters or values to try.

```python
from sklearn.model_selection import RandomizedSearchCV
random_cv = RandomizedSearchCV(LogisticRegression(), param_distributions=params, cv=5, scoring='accuracy')
random_cv.fit(X_train, y_train)
```

**Remember:** *GridSearchCV = try everything, thorough but slow. RandomizedSearchCV = try a sample, faster, nearly as good. Both should be paired with cross-validation (e.g. Stratified K-Fold) to pick reliably.*

---

## 7. ROC Curve and AUC Score — Choosing the Right Threshold

By default, logistic regression uses **0.5** as the cutoff: probability ≥ 0.5 → class 1, else class 0. But **0.5 isn't always the right choice** — the ideal threshold depends on the problem (e.g. a stricter 0.7 for a critical use case, or a looser 0.3 elsewhere).

### The two rates that matter

- **True Positive Rate (TPR)** = Recall = $\frac{TP}{TP+FN}$ — how many actual positives you correctly caught.
- **False Positive Rate (FPR)** = $\frac{FP}{FP+TN}$ — how many actual negatives you wrongly flagged.

### The ROC Curve

Plot **TPR (y-axis)** against **FPR (x-axis)** as the threshold slides from 0 to 1. Each point on the curve corresponds to one threshold value. A model that's just guessing randomly produces a diagonal line; a good model bulges up toward the top-left corner (high TPR, low FPR).

### AUC (Area Under the Curve)

A single number summarizing the whole ROC curve — **the bigger the area, the better the model** across all thresholds. A random/dummy model scores **AUC ≈ 0.5**; a strong model scores close to **1.0**.

### Picking the actual threshold

Walk along the curve and find the point that gives **high TPR with acceptably low FPR** — this is a business/domain decision, not a pure math one. A domain expert might say "false positives are cheap here, maximize TPR" (pick a point further along the curve), or "false positives are costly, keep FPR low" (pick a stricter point) — the annotated threshold at that chosen point becomes your new cutoff, replacing the default 0.5.

**Remember:** *ROC plots TPR vs FPR across all thresholds. AUC summarizes it in one number (closer to 1 = better, 0.5 = random guessing). Pick your threshold off the curve based on which error (FP or FN) is costlier for your problem.*

```python
from sklearn.metrics import roc_curve, roc_auc_score
model_probs = model.predict_proba(X_test)[:, 1]           # probability of class 1
auc = roc_auc_score(y_test, model_probs)
fpr, tpr, thresholds = roc_curve(y_test, model_probs)
```

---

## Summary Table

| Topic | One-line takeaway |
|---|---|
| Why not linear regression | Unbounded output + outlier-sensitive; classification needs [0,1] |
| Sigmoid + Log Loss | Sigmoid squashes to [0,1]; Log Loss keeps the cost function convex |
| Confusion Matrix | TP/TN correct, FP/FN wrong — the base for every other metric |
| Precision vs Recall | Precision = avoid false alarms; Recall = avoid missed cases |
| F-beta | Tunable blend of precision and recall (β=1 balanced, β<1 favors precision, β>1 favors recall) |
| One-vs-Rest | N classes → N binary models → highest probability wins |
| class_weight | Forces attention onto the minority class in imbalanced data |
| GridSearchCV / RandomizedSearchCV | Exhaustive vs sampled hyperparameter search |
| ROC / AUC | Visualizes all thresholds at once; AUC scores overall separability; pick the threshold that fits your cost tradeoff |

### Where to Go Next
- **Regularized logistic regression** — combine with Ridge/Lasso/Elastic Net penalties (see `../ridge_lasso_elasticnet/`).
- **One-vs-One (OVO)** — the other multi-class strategy, pairing classes two at a time.
- **Precision-Recall curves** — often more informative than ROC on heavily imbalanced data.
- **SVM, Decision Trees** — other classification algorithms to compare against.
