# AdaBoost — Complete Guide

A short, memorable reference for AdaBoost: boosting, sample weights, decision stumps, classification, regression, and practical implementation. Technical but plain-spoken. Builds on the ensemble concepts in `../random_forest/` and the single-tree concepts in `../decision_tree/`.

**The one idea to hold onto:** AdaBoost builds many small models in sequence. Each new model pays more attention to examples earlier models got wrong, and more accurate models receive a stronger vote in the final prediction.

---

## 1. From Bagging to Boosting

Ensemble learning combines multiple models to make one stronger model. The two major ensemble families are **bagging** and **boosting**.

| | Bagging | Boosting |
|---|---|---|
| Training style | Parallel | Sequential |
| Main goal | Reduce variance / overfitting | Reduce bias by correcting earlier errors |
| Typical model | Random Forest | AdaBoost, Gradient Boosting, XGBoost |
| Relationship between models | Independent | Each model depends on earlier ones |

**Random Forest** is bagging: many trees train independently on different bootstrap samples, then vote or average their results.

**AdaBoost** is boosting: it trains a weak learner, identifies the examples it handled badly, increases their importance, then trains the next learner. The final ensemble combines all learners using weighted voting.

> **Example:** imagine a tutor giving short quizzes. After the first quiz, the tutor spends more time on the questions the class missed. After several rounds, the teaching plan focuses repeatedly on the difficult concepts. That is the main idea of boosting.

**Remember:** *Bagging trains independent models in parallel; boosting trains models in sequence so later models can focus on earlier mistakes.*

---

## 2. Weak Learners and Decision Stumps

A **weak learner** is a model that performs only slightly better than random guessing. In classic AdaBoost classification, the usual weak learner is a **decision stump**: a decision tree with only one split.

```text
Is credit score good?
├── Yes → approve credit card
└── No  → do not approve credit card
```

A stump has `max_depth=1`: one question and two leaf nodes. By itself, it may underfit because it sees only one simple rule. AdaBoost turns many such weak learners into a **strong learner**.

In current scikit-learn:

- `AdaBoostClassifier` uses `DecisionTreeClassifier(max_depth=1)` by default.
- `AdaBoostRegressor` uses `DecisionTreeRegressor(max_depth=3)` by default.

You can provide a different estimator, including a slightly deeper decision tree. However, keep the base model reasonably weak; if every base tree is too complex, the ensemble can overfit and lose the simple boosting advantage.

**Remember:** *A decision stump is a one-split tree. It is weak alone, but AdaBoost combines many stumps into a strong classifier.*

---

## 3. Sample Weights — Making Hard Examples Matter More

AdaBoost assigns a **sample weight** to every training row. At the beginning, all rows are equally important:

$$w_i^{(1)} = \frac{1}{N}$$

where $N$ is the number of training rows.

After a weak learner makes predictions:

- a row predicted **incorrectly** receives more importance for the next round;
- a row predicted **correctly** becomes relatively less important after normalization.

> **Example:** assume there are five customer records. At the start, each has weight `1/5 = 0.20`. If the first stump incorrectly predicts Customer 4, AdaBoost raises Customer 4's weight. The next stump is therefore pushed to pay more attention to cases like Customer 4.

You may see two descriptions of this process:

1. **Weighted fitting** — the learner receives sample weights directly. This is how current scikit-learn `AdaBoostClassifier` trains its base estimator.
2. **Weighted resampling** — a new training set is sampled so high-weight rows are selected more often. Current scikit-learn `AdaBoostRegressor` uses this approach as part of the AdaBoost.R2 algorithm.

Both descriptions express the same intuition: difficult examples have more influence in the next round.

**Remember:** *Every row starts equally important. Errors receive more weight, so the next learner concentrates on cases the current learner missed.*

---

## 4. AdaBoost Classification: The Algorithm Step by Step

For classification, current scikit-learn uses the discrete **SAMME** boosting algorithm. Let $h_t(x)$ be the prediction from weak learner $t$.

### Step 1 — initialize equal weights

Set every training row to $1/N$ so all sample weights sum to 1.

### Step 2 — train a weak learner

Fit a stump to the data using the current sample weights.

### Step 3 — calculate weighted error

$$\varepsilon_t = \sum_{i=1}^{N}w_i^{(t)}\,\mathbf{1}[y_i \neq h_t(x_i)]$$

The indicator $\mathbf{1}[\cdot]$ is 1 when the prediction is wrong and 0 when it is correct. Because the weights already sum to 1, this is the weighted fraction of mistakes.

### Step 4 — assign the learner's vote weight

For $K$ classes, SAMME gives the learner this amount of influence:

$$\alpha_t = \text{learning\_rate}\left(\ln\left(\frac{1-\varepsilon_t}{\varepsilon_t}\right) + \ln(K-1)\right)$$

For binary classification, $K=2$ and $\ln(K-1)=\ln(1)=0$, so the formula becomes:

$$\alpha_t = \text{learning\_rate}\ln\left(\frac{1-\varepsilon_t}{\varepsilon_t}\right)$$

A learner with a lower error gets a larger positive weight. For $K$ classes, a learner no better than random guessing ($\varepsilon_t \geq 1 - \frac{1}{K}$) is not useful; AdaBoost stops or discards it instead of letting it weaken the ensemble. For binary classification, this threshold is $0.5$.

### Step 5 — update the sample weights

The practical idea is simple: increase the influence of wrongly predicted rows, then normalize all weights so they sum to 1.

For SAMME classification, the update is:

$$w_i^{(t+1)} = w_i^{(t)}\exp\left(\alpha_t\,\mathbf{1}[y_i \neq h_t(x_i)]\right)$$

This same update applies to binary and multi-class SAMME; the multi-class difference appears in the $\alpha_t$ formula above. Then normalize every $w_i^{(t+1)}$ by dividing by the sum of all updated weights.

Correct examples do not receive the exponential multiplier; their **relative** importance falls because error weights rose before normalization.

### Step 6 — repeat

Repeat the train → measure error → weight learner → update sample weights cycle for up to `n_estimators` rounds.

AdaBoost can stop before reaching that maximum when it gets a perfect learner or finds a learner no better than random guessing.

**Remember:** *Each round measures weighted mistakes, gives the learner a vote based on its error, and raises the importance of examples it missed.*

---

## 5. Final Classification Prediction: Weighted Voting

Random Forest uses ordinary majority voting. AdaBoost uses **weighted voting**: a reliable learner has a louder voice than an unreliable one.

For binary labels represented as $-1$ and $+1$, the final prediction is:

$$H(x) = \operatorname{sign}\left(\sum_{t=1}^{T}\alpha_t h_t(x)\right)$$

`sign` returns $+1$ when the weighted sum is positive and $-1$ when it is negative.

> **Example:** suppose three stumps make predictions for one loan application:
>
> - Stump 1 says **Approve**, with $\alpha_1 = 0.90$.
> - Stump 2 says **Reject**, with $\alpha_2 = 0.30$.
> - Stump 3 says **Approve**, with $\alpha_3 = 0.50$.
>
> The total support for Approve is $0.90 + 0.50 = 1.40$, while Reject has $0.30$. The final result is **Approve**, even though AdaBoost is not simply counting votes.

For multi-class SAMME, scikit-learn sums the weighted class votes and returns the class with the strongest total score.

**Remember:** *AdaBoost does not treat all weak learners equally. Better learners get larger alpha weights and have more influence on the final answer.*

---

## 6. AdaBoost Classification — Practical Example

### Problem example: wellness package prediction

A travel company wants to predict whether a customer will purchase a wellness package. Inputs can include age, occupation, number of follow-ups, pitch duration, marital status, and travel history. The target is `1` for purchase and `0` for no purchase.

Before training, clean and prepare the data:

1. fix inconsistent category labels and duplicates;
2. handle missing numerical and categorical values;
3. encode categorical values; and
4. split into training and test data **before** fitting transformations, avoiding data leakage.

Decision-tree-based AdaBoost normally does **not** require scaling. Category columns still need encoding for standard scikit-learn estimators.

The following self-contained example uses Iris so it can run without an external dataset:

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

stump = DecisionTreeClassifier(max_depth=1, random_state=42)
model = AdaBoostClassifier(
    estimator=stump,
    n_estimators=100,
    learning_rate=0.5,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

If the target is imbalanced, accuracy alone can hide poor minority-class performance. Also inspect precision, recall, F1-score, and ROC-AUC. AdaBoost itself has no `class_weight` parameter; if class weighting is needed, use a compatible base estimator that supports it or provide `sample_weight` to `.fit()`.

**Remember:** *AdaBoost classification stacks weighted stumps. Prepare categorical data, avoid leakage, and evaluate minority classes with more than accuracy.*

---

## 7. AdaBoost Regression — AdaBoost.R2

AdaBoost also supports continuous targets through **AdaBoost.R2**. Instead of wrong vs. correct labels, it measures the size of each prediction error and gives larger influence to difficult regression rows.

For a regression round, AdaBoost.R2:

1. trains a base regressor using a weighted bootstrap sample;
2. calculates each absolute error $|y_i - \hat{y}_i|$;
3. normalizes errors and applies the selected loss function; and
4. increases the focus on rows with larger loss in later rounds.

Current scikit-learn supports the following regressor loss choices:

| `loss` | What it does |
|---|---|
| `"linear"` | Uses normalized error directly; default choice |
| `"square"` | Penalizes larger errors more heavily |
| `"exponential"` | Increases emphasis nonlinearly for larger errors |

A key difference from Random Forest Regressor: AdaBoostRegressor combines its base-regressor outputs using a **weighted median**, not a simple average. A weighted median is more resistant to one extreme base prediction.

### Used-car price example

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

base_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
model = AdaBoostRegressor(
    estimator=base_tree,
    n_estimators=100,
    learning_rate=0.5,
    loss="linear",
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
```

- **MAE:** average size of the prediction error.
- **RMSE:** penalizes large errors more strongly than MAE.
- **$R^2$:** describes how much target variation is explained; a negative test value is worse than predicting the test-target mean.

**Remember:** *AdaBoost.R2 focuses on larger regression errors and combines regressors with a weighted median, not an ordinary mean.*

---

## 8. Compare Several Models, Not Only AdaBoost

Do not assume AdaBoost will be best for every dataset. Train several suitable models on the same training/test split, then compare the metric that matters for the real problem.

The following example uses the Iris train/test split from Section 6.

```python
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "adaboost": AdaBoostClassifier(random_state=42),
}

for name, candidate in models.items():
    candidate.fit(X_train, y_train)
    print(name, "accuracy:", candidate.score(X_test, y_test))
```

Use cross-validation during selection, then use an untouched test set once for final evaluation. A model with extremely high training performance but much lower test performance is likely overfitting.

AdaBoost can work very well with weak learners and clean patterns. It can also become sensitive to mislabeled rows and outliers because it repeatedly increases their influence. Random Forest is usually more robust to noisy data.

**Remember:** *AdaBoost is a strong candidate, not an automatic winner. Compare it fairly with other models and watch for sensitivity to noisy labels.*

---

## 9. Important Hyperparameters

| Hyperparameter | What it controls | Practical effect |
|---|---|---|
| `estimator` | Base weak learner | Default is a stump for classifier and a depth-3 tree for regressor; deeper trees increase complexity per round |
| `n_estimators` | Maximum number of boosting rounds | More rounds add capacity, but training is sequential and can amplify noise |
| `learning_rate` | Shrinkage applied to each learner weight | Smaller values usually need more estimators and can generalize better |
| `loss` | Regression error transformation | Regressor only: `"linear"`, `"square"`, or `"exponential"` |
| `random_state` | Randomness in base estimators and regression sampling | Makes experiments reproducible |

Useful defaults in current scikit-learn:

- `n_estimators=50`;
- `learning_rate=1.0`;
- `AdaBoostClassifier(estimator=None)` creates a decision stump internally;
- `AdaBoostRegressor(estimator=None)` creates a decision tree with `max_depth=3` internally;
- `AdaBoostRegressor(loss="linear")` uses linear loss.

`n_estimators` and `learning_rate` should be tuned together. A common strategy is to try a lower learning rate, such as `0.05`, `0.1`, or `0.5`, with more estimators. Use cross-validation instead of guessing from a single split.

> **Version note:** older tutorials may use `base_estimator=`. In modern scikit-learn, the correct parameter name is `estimator=`. They may also show `algorithm="SAMME"` or `algorithm="SAMME.R"`; current scikit-learn uses SAMME internally and no longer accepts an `algorithm` parameter.

**Remember:** *Tune learner complexity, number of rounds, and learning rate together. Modern scikit-learn uses `estimator=`, not the older `base_estimator=` keyword.*

---

## 10. Hyperparameter Tuning with RandomizedSearchCV

`GridSearchCV` tries every parameter combination. `RandomizedSearchCV` samples combinations, which is often quicker when you want to explore several parameters.

```python
from scipy.stats import loguniform, randint
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

param_distributions = {
    "estimator__max_depth": randint(1, 4),
    "n_estimators": randint(50, 301),
    "learning_rate": loguniform(0.01, 1.0),
}

search = RandomizedSearchCV(
    estimator=AdaBoostClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        random_state=42,
    ),
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
```

After tuning, evaluate `search.best_estimator_` once on the untouched test data. Do not keep changing hyperparameters according to test-set scores; that leaks test information into model selection.

**Remember:** *Randomized search tests a broad range efficiently. Tune through cross-validation, then reserve the test set for one final honest evaluation.*

---

## 11. AdaBoost vs. Random Forest vs. Gradient Boosting

| | AdaBoost | Random Forest | Gradient Boosting |
|---|---|---|---|
| Ensemble type | Boosting (sequential) | Bagging (parallel) | Boosting (sequential) |
| Typical weak learner | Decision stump | Full decision tree | Shallow decision tree |
| Main strength | Simple boosting baseline | Robust, strong general tabular baseline | Flexible modelling of complex patterns |
| Main weakness | Sensitive to noisy data and outliers | May not reduce bias as strongly | More hyperparameters and slower tuning |
| How models combine | Weighted vote / weighted median | Vote / average | Adds small prediction corrections |
| Noise sensitivity | High | Lower | Medium to high |

AdaBoost is historically important and excellent for learning boosting. In many modern tabular-data projects, Random Forest, Gradient Boosting, XGBoost, LightGBM, or CatBoost may be stronger starting points. Still, AdaBoost is useful as a simple, interpretable boosting baseline.

**Remember:** *AdaBoost is sequential and error-focused; Random Forest is parallel and robust; Gradient Boosting learns incremental prediction corrections.*

---

## 12. When to Use AdaBoost — and When Not To

AdaBoost is a good choice when:

- the data is reasonably clean;
- simple trees can detect useful patterns;
- you want a lightweight boosting baseline; and
- you will compare it against other models using validation data.

Be careful or choose another method when:

- labels contain many errors or the dataset has strong outliers;
- the target relationship needs much more complex base learners;
- training must be highly parallel—boosting rounds are sequential; or
- calibrated class probabilities are essential. AdaBoost's `predict_proba` values are derived from its decision scores and should not automatically be treated as calibrated probabilities.

**Remember:** *AdaBoost works best on reasonably clean data with useful weak rules. Its repeated focus on hard points makes it vulnerable to persistent noise and label errors.*

---

## Summary Table

| Topic | One-line takeaway |
|---|---|
| Boosting | Trains models in sequence so later learners focus on earlier mistakes |
| Decision stumps | One-split trees that are weak alone but useful together |
| Sample weights | Incorrectly predicted rows become more influential in the next round |
| Weighted error | Measures mistakes using current sample importance, not just the row count |
| Alpha weight | Better weak learners receive a stronger vote in the final ensemble |
| Final classification | Uses a weighted vote, not ordinary majority voting |
| AdaBoost.R2 | Regression version that focuses on large errors and uses a weighted median |
| Hyperparameters | Tune `estimator`, `n_estimators`, and `learning_rate` together |
| Noise sensitivity | AdaBoost can overfocus on noisy or mislabeled observations |
| Model selection | Compare it fairly with Random Forest and other suitable models |

### Where to Go Next

- **Random Forest** — revisit bagging, bootstrap sampling, and parallel tree ensembles in `../random_forest/`.
- **Decision Trees** — review tree splitting, impurity, variance reduction, and pruning in `../decision_tree/`.
- **Classification metrics** — precision, recall, F1-score, ROC curves, and AUC apply to AdaBoost just as they do to logistic regression; see `../logistic_regression/` for worked examples.
- **Gradient Boosting** — continue to residual and loss-gradient-based boosting in `../gradient_boosting/`; then study XGBoost and related algorithms.
- **Regularization and cross-validation** — revisit evaluation and model-selection workflow in `../ridge_lasso_elasticnet/`.
