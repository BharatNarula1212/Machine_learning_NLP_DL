# Gradient Boosting — Complete Guide

A short, memorable reference for Gradient Boosting: residual learning, classification, regression, tuning, feature importance, and practical implementation. Technical but plain-spoken. Builds on the boosting ideas in `../adaboost/`, the bagging ideas in `../random_forest/`, and the tree concepts in `../decision_tree/`.

**The one idea to hold onto:** Gradient Boosting builds small decision trees one after another. Every new tree learns a small correction to the ensemble's mistakes so far; adding many controlled corrections creates a strong model.

---

## 1. From AdaBoost to Gradient Boosting

Both AdaBoost and Gradient Boosting are **boosting** algorithms. They train models **sequentially**, not independently. Each new model depends on the work done by the earlier models.

The important difference is *what* the next learner focuses on:

| Algorithm | What later learners focus on | How learners combine |
|---|---|---|
| AdaBoost | Rows that earlier learners classified incorrectly | Weighted vote or weighted median |
| Gradient Boosting | The remaining prediction error, more precisely the negative gradient of a loss function | Adds small prediction corrections |
| Random Forest | Nothing from earlier trees; trees train independently | Vote or average |

For squared-error regression, the negative gradient is simply the familiar **residual**:

$$r_i = y_i - \hat{y}_i$$

where $y_i$ is the real value and $\hat{y}_i$ is the current prediction. This makes regression the easiest way to understand Gradient Boosting.

> **Example:** imagine estimating a used car's price. The first model says every car is worth ₹5 lakh. A second small tree learns corrections such as “older cars need a negative correction” and “newer automatic cars need a positive correction.” The next tree corrects what still remains wrong. The final prediction is the starting estimate plus all of those small corrections.

**Remember:** *AdaBoost changes the importance of difficult rows; Gradient Boosting learns an additive correction to the current prediction.*

---

## 2. The Core Idea: Learn the Remaining Error

Gradient Boosting starts with a very simple prediction and improves it in rounds.

For regression with squared error, the usual starting model is the training-target mean:

$$F_0(x) = \bar{y}$$

Suppose salaries, in thousands, are `[50, 70, 80, 100]`. The initial prediction for every row is:

$$\bar{y} = \frac{50 + 70 + 80 + 100}{4} = 75$$

The first residuals are:

| Actual salary $y$ | Initial prediction $F_0(x)$ | Residual $r_1 = y - F_0(x)$ |
|---:|---:|---:|
| 50 | 75 | -25 |
| 70 | 75 | -5 |
| 80 | 75 | 5 |
| 100 | 75 | 25 |

A small regression tree receives the input features, such as experience and degree, and learns to predict those residuals. Assume it predicts these approximate corrections: `[-23, -3, 3, 20]`. They do not have to equal the individual residuals exactly: a tree can place several rows in one leaf and use that leaf's average residual as its prediction.

If the learning rate $\eta$ is `0.1`, the first updated prediction for the first salary is:

$$F_1(x) = F_0(x) + \eta h_1(x) = 75 + 0.1(-23) = 72.7$$

It did not jump all the way from 75 to 52. The learning rate deliberately makes a **small step**. The next tree then learns the new residuals, and the cycle repeats.

> **Why not add the entire correction?** A full correction can fit training details too aggressively. Small steps give later trees a chance to refine the model gradually and often generalize better.

**Remember:** *A Gradient Boosting tree is not usually predicting the final target. It is predicting the correction that the ensemble currently needs.*

---

## 3. Gradient Boosting Regression: Step by Step

For a continuous target, such as salary, house price, or used-car value, Gradient Boosting Regressor commonly uses squared error.

### Step 1 — create a starting prediction

Predict the training-target mean for every row:

$$F_0(x) = \bar{y}$$

### Step 2 — calculate residuals

For round $m$, compare the actual value with the current ensemble prediction:

$$r_{im} = y_i - F_{m-1}(x_i)$$

### Step 3 — fit a shallow regression tree

Train tree $h_m(x)$ using the original input features $X$ and the current residuals as its target. The tree uses the same regression-tree ideas as before: it selects splits that reduce squared error within its leaves.

### Step 4 — add a shrunken correction

Update the ensemble with the tree's correction:

$$F_m(x) = F_{m-1}(x) + \eta h_m(x)$$

Here, $\eta$ is the `learning_rate`. A lower value means smaller changes per tree.

### Step 5 — repeat

Repeat residual calculation, tree fitting, and updating up to `n_estimators` times. The final model is:

$$F_M(x) = F_0(x) + \eta\sum_{m=1}^{M}h_m(x)$$

This residual story is exact for squared-error regression. With other loss functions, Gradient Boosting fits the **negative gradient of the chosen loss**, which plays the role of the best next correction.

**Remember:** *For squared-error regression, each new tree learns residuals; the ensemble adds that tree's prediction after shrinking it by the learning rate.*

---

## 4. Gradient Boosting Classification: The Same Principle, a Different Loss

Classification still adds trees sequentially, but a simple numeric residual is not enough. The model must represent class confidence and optimize a classification loss.

For binary classification, `GradientBoostingClassifier` uses **log loss** by default. For multiple classes, scikit-learn extends the same idea to multiclass log loss.

At every boosting round, the algorithm:

1. computes how the current model's predicted probabilities differ from the correct labels according to log loss;
2. turns that information into **pseudo-residuals**, the negative gradients of the loss;
3. trains shallow regression trees from those pseudo-residuals; and
4. estimates each tree's leaf corrections for the selected loss, then adds their scaled outputs to the model's class scores.

For classification, the exact leaf correction depends on log loss and is not merely an ordinary residual average. Scikit-learn performs this optimization internally; you normally do not calculate it by hand.

The score is then converted into a probability. In binary classification, this conversion uses the sigmoid function:

$$p(y=1 \mid x) = \frac{1}{1 + e^{-F(x)}}$$

You do not need to calculate these gradients manually for normal use. The key distinction is:

- **Regression with squared error:** a tree learns ordinary residuals.
- **Classification with log loss:** a tree learns negative-gradient pseudo-residuals that improve probability estimates.

> **Example:** a holiday-package model may currently assign only a 35% purchase probability to customers who have several successful follow-ups. A new tree can learn a positive correction for that pattern, raising the final probability for similar customers.

**Remember:** *Gradient Boosting always learns the next useful correction. For regression that is often a residual; for classification it is a loss-gradient-based pseudo-residual.*

---

## 5. Gradient Boosting Classification — Practical Example

### Problem example: wellness package prediction

A travel company may use age, occupation, follow-up count, pitch duration, marital status, and travel history to predict whether a customer will purchase a wellness package.

Before training a classic `GradientBoostingClassifier`:

1. clean duplicate rows and inconsistent category labels;
2. impute missing values;
3. encode categorical columns; and
4. fit transformations on training data only, then apply them to validation or test data.

Classic sklearn Gradient Boosting models do not accept missing values directly. Trees normally do not need feature scaling, but standard sklearn tree estimators still require numeric, non-missing inputs.

The self-contained example below uses Iris data:

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    loss="log_loss",
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Setting `subsample` below `1.0` makes each round train on a random fraction of rows. This is called **stochastic gradient boosting**. It can reduce variance, but it also adds randomness, so keep `random_state` fixed when comparing experiments.

For an imbalanced target, accuracy can be misleading. Check precision, recall, F1-score, and potentially ROC-AUC, based on the business cost of false positives and false negatives.

**Remember:** *Gradient Boosting classification adds shallow trees that improve probability-based predictions. Impute and encode data first, then evaluate with metrics that fit the problem.*

---

## 6. Gradient Boosting Regression — Practical Example

### Problem example: used-car price prediction

A used-car seller wants to predict a continuous sale price using features such as age, kilometers driven, fuel type, transmission, engine size, power, and seats.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    loss="squared_error",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=2,
    min_samples_leaf=3,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
```

### Regression metrics in simple words

- **MAE:** the average absolute size of errors. If MAE is ₹40,000, the prediction misses the real price by about ₹40,000 on average.
- **RMSE:** penalizes larger errors more heavily than MAE.
- **$R^2$:** measures explained variation. Closer to 1 is generally better; a negative test $R^2$ is worse than predicting the test-target mean.

For regression, useful loss choices include:

| `loss` | When it can help |
|---|---|
| `"squared_error"` | Default; penalizes large errors strongly |
| `"absolute_error"` | More resistant to unusually large target errors |
| `"huber"` | A compromise: squared-error-like near zero, less sensitive to large errors |
| `"quantile"` | Predicting a conditional quantile, such as an upper-price estimate |

**Remember:** *Gradient Boosting Regressor starts from a baseline and keeps adding small tree-based corrections. Use MAE, RMSE, and $R^2$ together to judge it.*

---

## 7. Compare Several Models, Not Only Gradient Boosting

Gradient Boosting is powerful, but it is not automatically the best choice. Use the same split, preprocessing, and evaluation metric when comparing suitable models.

The following example uses the Iris train/test split from Section 5:

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "adaboost": AdaBoostClassifier(random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}

for name, candidate in models.items():
    candidate.fit(X_train, y_train)
    print(name, "accuracy:", candidate.score(X_test, y_test))
```

Use cross-validation while choosing the model, then evaluate the selected model once on an untouched test set. Repeatedly changing choices after seeing test performance leaks test-set information into model selection.

Gradient Boosting can overfit when trees are too deep, too many estimators are used without sufficient shrinkage, or hyperparameters are repeatedly selected on the same small validation set.

**Remember:** *A fair comparison beats an assumption. Test Gradient Boosting alongside simpler models, Random Forest, and AdaBoost using the same data split and metric.*

---

## 8. Important Hyperparameters

The most important Gradient Boosting controls balance the size of each correction against the number and complexity of trees.

| Hyperparameter | What it controls | Practical effect |
|---|---|---|
| `n_estimators` | Number of sequential trees | More rounds add capacity and training time |
| `learning_rate` | Shrinkage applied to every tree's correction | Smaller values usually need more trees and may generalize better |
| `max_depth` | Maximum depth of each individual tree | Shallower trees learn simpler corrections and reduce overfitting risk |
| `min_samples_split` | Rows required before splitting a node | Larger values block overly specific splits |
| `min_samples_leaf` | Minimum rows in a leaf | Larger values smooth predictions and reduce noise fitting |
| `subsample` | Fraction of rows used per boosting round | Values below `1.0` create stochastic Gradient Boosting |
| `max_features` | Features considered at each split | Smaller values add randomness and can reduce variance |
| `loss` | Objective being minimized | Classifier: `"log_loss"` (default) or `"exponential"` for binary targets only; regression offers several choices |
| `n_iter_no_change` | Early-stopping patience | Stops when validation performance does not improve for several rounds |

Useful defaults in current scikit-learn are `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, and `subsample=1.0`.

The main trade-off is between `learning_rate` and `n_estimators`:

- a **larger** learning rate uses bolder corrections and may need fewer trees;
- a **smaller** learning rate uses gentler corrections and usually needs more trees.

Start with a modest tree depth, such as 2 or 3. Gradient Boosting normally uses **shallow regression trees**, not AdaBoost-style stumps and not full-depth trees.

> **Version note:** older examples may show `loss="deviance"` for classification or `loss="ls"` for regression. Modern scikit-learn uses `"log_loss"` and `"squared_error"` instead. Unlike AdaBoost, classic Gradient Boosting does not accept `estimator=` or `base_estimator=`; it constructs its internal regression trees itself.

**Remember:** *Tune learning rate, number of trees, and tree complexity together. Smaller, slower corrections are often safer than a few aggressive trees.*

---

## 9. Hyperparameter Tuning with RandomizedSearchCV

`GridSearchCV` evaluates every supplied combination. `RandomizedSearchCV` samples a selected number of combinations, which is often more practical when several parameters have wide ranges.

This example is self-contained and tunes a classifier with cross-validation:

```python
from scipy.stats import loguniform, randint
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_distributions = {
    "n_estimators": randint(50, 301),
    "learning_rate": loguniform(0.01, 0.3),
    "max_depth": randint(1, 5),
    "min_samples_leaf": randint(1, 11),
    "subsample": [0.7, 0.85, 1.0],
}

search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print("Best cross-validation score:", search.best_score_)
print("Test accuracy:", search.best_estimator_.score(X_test, y_test))
```

For a regression problem, change the estimator to `GradientBoostingRegressor` and choose a regression scoring metric such as `"neg_root_mean_squared_error"` or `"neg_mean_absolute_error"`.

Use the test score once after tuning. If you use it repeatedly to decide the search space, it is no longer a truly untouched test score.

**Remember:** *Randomized search explores useful combinations efficiently. Tune on cross-validation, then use the held-out test set for a final honest check.*

---

## 10. Feature Importance — Useful, but Interpret Carefully

Classic `GradientBoostingClassifier` and `GradientBoostingRegressor` provide `feature_importances_`. It measures total impurity reduction from each feature across the ensemble.

The following snippets continue the Iris classifier example from Section 5, where `model`, `X_test`, and `y_test` are already defined.

```python
import pandas as pd
from sklearn.datasets import load_iris

feature_names = load_iris().feature_names
importance = pd.Series(
    model.feature_importances_,
    index=feature_names,
).sort_values(ascending=False)

print(importance)
```

Impurity importance is quick, but features with many possible split points can appear overly important. Confirm important findings with **permutation importance**, which measures the drop in test performance after shuffling one feature at a time.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

permutation_scores = pd.Series(
    result.importances_mean,
    index=feature_names,
).sort_values(ascending=False)
print(permutation_scores)
```

Feature importance shows an association with this trained model. It does **not** prove that changing a feature causes the predicted outcome to change.

**Remember:** *Use `feature_importances_` for a quick view, then check the result with permutation importance on held-out data. Importance is not causation.*

---

## 11. HistGradientBoosting — A Faster Modern Variant

`GradientBoostingClassifier` and `GradientBoostingRegressor` are useful for learning the basic algorithm. For larger tabular datasets, scikit-learn also provides **histogram-based** versions:

- `HistGradientBoostingClassifier`
- `HistGradientBoostingRegressor`

Instead of considering every possible numeric split point, they group feature values into bins (histograms). This is generally much faster on datasets with more than about 10,000 rows.

| Capability | Classic Gradient Boosting | HistGradientBoosting |
|---|---|---|
| Tree count parameter | `n_estimators` | `max_iter` |
| Large-data training | Can become slow | Usually faster through histogram binning |
| Missing values | Impute first | Handles `NaN` values natively |
| Early stopping | Configure `n_iter_no_change` | `early_stopping="auto"` by default |
| `feature_importances_` | Available | Not available; use permutation importance |

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

hist_model = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=15,
    random_state=42,
)
hist_model.fit(X_train, y_train)
print("Test accuracy:", hist_model.score(X_test, y_test))
```

The histogram version can also use constraints such as `monotonic_cst` when domain knowledge says a prediction must move only in one direction as a feature increases. Use constraints only when that relationship is genuinely appropriate.

**Remember:** *Classic Gradient Boosting teaches the idea clearly. HistGradientBoosting is often the better practical choice for larger tabular datasets and can handle missing values directly.*

---

## 12. Gradient Boosting vs. AdaBoost vs. Random Forest

| Topic | AdaBoost | Random Forest | Gradient Boosting |
|---|---|---|---|
| Ensemble type | Boosting, sequential | Bagging, parallel | Boosting, sequential |
| Typical base learner | Decision stump | Full decision tree | Shallow regression tree |
| Main learning signal | Higher weight for earlier mistakes | Independent bootstrap samples | Prediction corrections / loss gradients |
| Main strength | Simple boosting baseline | Robust general-purpose baseline | Flexible, often strong tabular performance |
| Main weakness | Sensitive to noisy labels and outliers | Can leave bias uncorrected | Sequential training and more tuning sensitivity |
| Missing values in classic sklearn version | Impute first | Impute first | Impute first; HistGradientBoosting is the exception |
| Training style | Sequential | Trees can train in parallel | Sequential |

Gradient Boosting often performs strongly on structured tabular data when its depth, learning rate, and number of estimators are tuned carefully. Random Forest is often a simpler, robust first baseline. AdaBoost is valuable for understanding boosting but can overfocus on persistent noise.

XGBoost, LightGBM, and CatBoost build on Gradient Boosting ideas, while adding specialized regularization and engineering choices. They are related algorithms, not exact replacements with identical behavior or APIs.

**Remember:** *Random Forest reduces variance through independent trees. AdaBoost emphasizes hard examples. Gradient Boosting repeatedly adds loss-reducing corrections.*

---

## 13. When to Use Gradient Boosting — and When Not To

Gradient Boosting is a good candidate when:

- the task uses structured or tabular data;
- patterns are nonlinear and involve feature interactions;
- you can validate and tune hyperparameters carefully;
- prediction quality matters more than explaining one short set of rules; and
- the data is clean enough that a sequence of corrections can learn stable patterns.

Be careful or consider another model when:

- the dataset is extremely noisy or labels are unreliable;
- very fast training is essential and the data is large—consider `HistGradientBoosting` or another histogram-based library;
- a single transparent decision tree is required for explanation;
- many categorical features need specialized handling—consider appropriate preprocessing or a categorical-aware algorithm; or
- you have not reserved validation and test data for fair evaluation.

Gradient Boosting is not a replacement for good data preparation. Handle missing values, encode categories where needed, inspect leakage, and compare against simple baselines before trusting a higher score.

**Remember:** *Use Gradient Boosting as a carefully validated tabular-data candidate, not as an automatic winner. Good preprocessing and honest evaluation still matter most.*

---

## Summary Table

| Topic | One-line takeaway |
|---|---|
| Boosting | Trains models sequentially so each new learner improves the current ensemble |
| AdaBoost vs. Gradient Boosting | AdaBoost raises the influence of hard rows; Gradient Boosting learns prediction corrections |
| Starting model | Regression commonly begins by predicting the target mean |
| Residuals | With squared error, a residual is actual value minus current prediction |
| Negative gradient | The general form of the next correction for any differentiable loss |
| Learning rate | Shrinks every tree's contribution and trades off with the number of trees |
| Shallow trees | Small trees learn controlled corrections and reduce overfitting risk |
| Classification | Uses log loss and pseudo-residuals to improve class probabilities |
| Regression | Uses losses such as squared error, absolute error, Huber, or quantile |
| Tuning | Tune `learning_rate`, `n_estimators`, depth, leaf size, and subsampling with cross-validation |
| Feature importance | Check impurity importance with permutation importance; neither proves causation |
| HistGradientBoosting | A faster histogram-based sklearn variant with native missing-value support |

### Where to Go Next

- **AdaBoost** — compare sample reweighting and weighted voting in `../adaboost/`.
- **Random Forest** — revisit bagging, bootstrap samples, OOB evaluation, and feature randomness in `../random_forest/`.
- **Decision Trees** — review splitting, impurity, variance reduction, and pruning in `../decision_tree/`.
- **Classification metrics** — see precision, recall, F1-score, ROC curves, and AUC in `../logistic_regression/`.
- **Regularization and model workflow** — revisit cross-validation and evaluation in `../ridge_lasso_elasticnet/`.
- **Next boosting algorithms** — study XGBoost, LightGBM, and CatBoost after mastering the Gradient Boosting foundation.
