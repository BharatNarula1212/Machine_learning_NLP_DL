# Random Forest — Complete Guide

A short, memorable reference for Random Forest: ensemble learning, bagging, feature randomness, classification, regression, model evaluation, and tuning. Technical but plain-spoken. Builds on the single-tree concepts in `../decision_tree/`.

**The one idea to hold onto:** a random forest is a crowd of decision trees. Each tree sees a slightly different version of the data, and the forest combines their answers to make a more stable prediction than one tree alone.

---

## 1. Why Move from One Decision Tree to a Forest?

A single decision tree is easy to understand, but it can **overfit**. If the tree grows very deep, it can memorize small details and noise in the training data. It then performs well on training data but less well on new data.

A **Random Forest** reduces this problem by training many decision trees and combining their predictions. The trees make different mistakes, so averaging or voting reduces the effect of any one tree's mistake.

> **Example:** ask one person whether a customer will buy a holiday package and their answer may be unreliable. Ask 200 people who saw slightly different customer samples, then use the majority answer: the group is usually more reliable.

Random Forest is an **ensemble learning** method: it combines many models to form one stronger model. More precisely, it is a **bagging** method built from decision trees.

**Remember:** *One deep decision tree can have low bias but high variance. A random forest reduces that variance by combining many different trees.*

---

## 2. Ensemble Learning: Bagging vs. Boosting

Ensemble learning combines multiple models, often called **base learners**, to make a better final prediction.

### Bagging — train models in parallel

**Bagging** means **bootstrap aggregating**:

1. Create many random samples from the training dataset.
2. Train one model on each sample, independently.
3. Combine the models' predictions.

The models can train **in parallel** because one model does not need the result from another model.

- For **classification**, use **majority voting**.
- For **regression**, use the **average** of predictions.

Random Forest is a bagging algorithm in which every base learner is a decision tree.

### Boosting — train models in sequence

**Boosting** also combines models, but it works differently. Models train **sequentially**: later models focus more on examples that earlier models predicted poorly. AdaBoost, Gradient Boosting, and XGBoost are common boosting algorithms.

| | Bagging | Boosting |
|---|---|---|
| Training style | Parallel | Sequential |
| Main goal | Reduce variance / overfitting | Reduce bias by correcting earlier errors |
| Typical model | Random Forest | AdaBoost, Gradient Boosting, XGBoost |
| Relationship between models | Independent | Each model depends on earlier ones |

**Remember:** *Bagging builds independent models in parallel and combines them. Boosting builds models one after another, with later models learning from earlier mistakes.*

---

## 3. Bootstrap Sampling — How Each Tree Gets Its Data

Suppose the training dataset has $N$ rows. For every tree, Random Forest creates a new training sample by selecting $N$ rows **with replacement**.

**With replacement** means that after a row is selected, it can be selected again. Therefore:

- some original rows appear more than once in a tree's sample;
- some original rows do not appear in that sample at all;
- every tree receives a different, but overlapping, view of the data.

This sample is called a **bootstrap sample**.

> **Example:** imagine a dataset with customer rows `[A, B, C, D, E]`. A bootstrap sample for one tree could be `[A, C, C, D, E]`. Row `C` appears twice and row `B` is missing. Another tree gets a different sample.

For a large dataset, a bootstrap sample contains about **63.2% unique rows** on average. The remaining roughly **36.8%** are not used by that tree; these are its **out-of-bag** rows (Section 6).

**Remember:** *Bootstrap sampling draws rows with replacement. It makes every tree different, even though every tree starts from the same original training dataset.*

---

## 4. The “Random” Part: Random Feature Selection

Bagging alone makes trees different because they see different rows. Random Forest adds one more source of randomness: **at every split, each tree considers only a random subset of features**.

A normal decision tree checks all possible features at a split and selects the best one. In a Random Forest, a tree might see only a few randomly selected features at that split.

Why is that helpful? If one feature is very strong, every normal tree tends to split on it first. The trees then become too similar, so combining them adds little value. Feature randomness makes the trees less correlated, allowing the forest to benefit more from voting or averaging.

> **Example:** to predict package purchase, `MonthlyIncome` might be very powerful. If every tree always uses it first, all trees can look alike. Randomly limiting some trees to features such as `Age`, `DurationOfPitch`, and `NumberOfFollowUps` creates different decision paths.

In current scikit-learn defaults:

- `RandomForestClassifier` uses `max_features="sqrt"`: at each split, it considers approximately $\sqrt{p}$ features, where $p$ is the total number of features.
- `RandomForestRegressor` uses `max_features=1.0`: it considers all features by default. Use a smaller value, such as `"sqrt"` or `0.3`, when you want more feature randomness.

**Remember:** *Random Forest randomizes both rows and features. Random feature selection prevents trees from becoming copies of each other.*

---

## 5. How Random Forest Makes Predictions

After training many trees, Random Forest sends a new observation to every tree.

### Classification: majority vote

Each tree predicts a class. The final class is the class with the most votes.

> **Example:** five trees predict whether a customer will purchase a package:
>
> `No, Yes, Yes, Yes, No`
>
> `Yes` receives three votes, so the forest predicts **Yes**.

For class probabilities, scikit-learn averages the class probabilities produced by all trees, then selects the class with the highest average probability.

### Regression: average prediction

Each tree predicts a continuous number. The final prediction is their average:

$$\hat{y}_{forest} = \frac{1}{T}\sum_{t=1}^{T}\hat{y}_t$$

where $T$ is the number of trees and $\hat{y}_t$ is tree $t$'s prediction.

> **Example:** five trees estimate a used car price as ₹5.2L, ₹5.5L, ₹5.3L, ₹5.7L, and ₹5.4L. The Random Forest prediction is their average: **₹5.42L**.

**Remember:** *Classification uses voting; regression uses averaging. Combining many tree outputs makes the final prediction less sensitive to one unusual tree.*

---

## 6. Out-of-Bag (OOB) Evaluation

Because every tree is trained on a bootstrap sample, some training rows are left out of that tree's sample. These rows are called **out-of-bag (OOB)** observations.

For every training row, Random Forest can collect predictions only from trees that did **not** train on that row. Comparing those predictions with the real label gives an **OOB score**:

- classifier: OOB **accuracy** by default;
- regressor: OOB **$R^2$** score by default.

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
forest.fit(X_train, y_train)

print("OOB accuracy:", forest.oob_score_)
```

OOB evaluation is a useful, convenient validation estimate, but it does **not** replace a final held-out test set. Keep a test set untouched until you are ready to report final performance.

**Remember:** *OOB rows are the samples a particular tree never saw. Their predictions provide a built-in validation estimate while training the forest.*

---

## 7. Random Forest Classification — Example

### Problem example: wellness package prediction

A travel company wants to predict whether a customer is likely to buy a new wellness package. Its data may include age, occupation, number of follow-ups, product pitch duration, marital status, and travel history.

The target is categorical:

- `1` — customer purchases the package;
- `0` — customer does not purchase it.

Before modelling, clean the data:

1. inspect and remove duplicates when appropriate;
2. fix inconsistent category labels, such as spelling variants;
3. fill missing numerical values with a suitable statistic, often the median;
4. fill missing categorical values with the mode or an explicit `"Unknown"` category;
5. encode categorical columns; and
6. split data before learning transformations to avoid data leakage.

Tree-based models normally do **not** require feature scaling, unlike KNN, SVM, or many linear models. Categorical values must still be encoded before being passed to standard scikit-learn Random Forest models.

### Minimal classifier example

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("OOB accuracy:", model.oob_score_)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

For an imbalanced target—such as far fewer customers buying a package than not buying it—do not depend on accuracy alone. Also inspect precision, recall, F1-score, and potentially ROC-AUC. Consider `class_weight="balanced"` if missing minority-class cases is important.

**Remember:** *For classification, prepare clean encoded features, train many trees, and evaluate more than accuracy when classes are imbalanced.*

---

## 8. Random Forest Regression — Example

### Problem example: used-car price prediction

A seller wants to estimate a used car's selling price from features such as vehicle age, kilometers driven, fuel type, transmission type, engine size, power, seats, and model.

The target is continuous, so this is a regression problem.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("OOB R2:", model.oob_score_)
print("Test R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
```

### Regression metrics in simple words

- **MAE (Mean Absolute Error):** average absolute size of the mistakes. If MAE is ₹40,000, predictions are off by about ₹40,000 on average.
- **RMSE (Root Mean Squared Error):** like MAE, but penalizes large mistakes more strongly.
- **$R^2$ score:** how much of the target's variation the model explains. Closer to 1 is generally better; a negative test $R^2$ means the model is worse than simply predicting the target mean.

**Remember:** *Random Forest Regressor averages many tree predictions. Use MAE/RMSE to measure error size and $R^2$ to measure explained variation.*

---

## 9. Training More Than One Model

Random Forest is powerful, but it should not be assumed to be best for every dataset. Train several reasonable models and compare them on the **same train/test split** and the **same metric**.

For a classification project, compare Random Forest with models such as Logistic Regression and a Decision Tree. For a regression project, compare it with Linear Regression, Ridge/Lasso, KNN Regressor, and Decision Tree Regressor.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(random_state=42, n_jobs=-1),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, "accuracy:", model.score(X_test, y_test))
```

Do not choose solely from training performance. A model with nearly 100% training accuracy but much lower test accuracy is likely overfitting. Cross-validation and a held-out test set make comparisons more reliable.

**Remember:** *Use Random Forest as a strong baseline, not an automatic winner. Compare models fairly using the same split and the metric that matters to the problem.*

---

## 10. Important Hyperparameters

A Random Forest has many settings. Start with defaults, then tune only the settings that matter most.

| Hyperparameter | What it controls | Practical effect |
|---|---|---|
| `n_estimators` | Number of trees | More trees usually make predictions more stable, but increase runtime and memory |
| `max_depth` | Maximum depth of each tree | Lower values restrict complexity and can reduce overfitting |
| `min_samples_split` | Minimum rows needed to split a node | Larger values prevent very specific splits |
| `min_samples_leaf` | Minimum rows allowed in a leaf | Larger values make predictions smoother and less noisy |
| `max_features` | Features considered at each split | Smaller values add diversity between trees |
| `bootstrap` | Whether to use bootstrap samples | Must be `True` when using OOB scoring |
| `class_weight` | Relative class importance for classification | Useful when the target classes are imbalanced |
| `criterion` | Split-quality measure | Classifier: `"gini"`, `"entropy"`, `"log_loss"`; regressor: `"squared_error"`, `"absolute_error"`, `"poisson"` |

Important defaults in current scikit-learn:

- both classifier and regressor use `n_estimators=100`;
- classifier default: `max_features="sqrt"`;
- regressor default: `max_features=1.0`;
- `max_depth=None` allows trees to grow until leaves are pure or cannot be split further;
- `bootstrap=True`.

Use `random_state` in examples and experiments. Random Forest deliberately uses random sampling and random feature order, so a fixed seed makes your result reproducible.

**Remember:** *The most useful tuning controls are the number of trees, tree depth, minimum samples per split/leaf, and number of features considered per split.*

---

## 11. Hyperparameter Tuning with RandomizedSearchCV

`GridSearchCV` tries every combination in a parameter grid. It is thorough but can become slow very quickly. `RandomizedSearchCV` samples a selected number of combinations, making it a good first choice for Random Forests with several parameters.

```python
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    "n_estimators": randint(100, 500),
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", 0.3],
}

search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
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

After the search, evaluate `search.best_estimator_` once on the untouched test set. Do not tune repeatedly on the test set; that slowly turns the test set into training information.

> **Note:** old tutorials may show `max_features="auto"`. It is no longer a valid Random Forest option in modern scikit-learn. Use `"sqrt"`, `"log2"`, `None`, an integer, or a fraction such as `0.3` instead.

**Remember:** *RandomizedSearchCV explores a broad hyperparameter space faster than an exhaustive grid. Tune with cross-validation, then report final performance on untouched test data.*

---

## 12. Feature Importance — Useful, but Interpret Carefully

Random Forest exposes `feature_importances_`, which measures how much each feature reduced impurity across all trees. It is fast and useful for a first view of which features influenced the model most.

The following snippets continue the Iris classifier example from Section 7, where `model`, `X_test`, and `y_test` are already defined.

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

However, impurity-based feature importance can be misleading for features with many unique values, such as customer IDs or high-cardinality categories. Such features can appear artificially important because they offer many possible split points.

Use **permutation importance** as a stronger check. It measures how much the model's test performance drops after shuffling one feature at a time.

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

Feature importance describes association with the model's predictions. It does **not** prove that changing a feature causes the outcome to change.

**Remember:** *`feature_importances_` is quick but can favor high-cardinality features. Validate important findings with permutation importance on held-out data.*

---

## 13. Random Forest vs. a Single Decision Tree

| Topic | Single Decision Tree | Random Forest |
|---|---|---|
| Interpretability | Very high: draw and inspect one tree | Lower: hundreds of trees are difficult to explain individually |
| Overfitting | High risk when the tree is deep | Lower risk because many trees are combined |
| Accuracy / stability | Can change a lot when data changes slightly | Usually more stable and often more accurate |
| Training speed | Faster | Slower because many trees are trained |
| Prediction speed | Fast | Usually still fast, but slower than one tree |
| Best use | Explainable baseline, simple rule extraction | Strong general-purpose tabular-data model |

Use a **single Decision Tree** when explainability is the main requirement—for example, when you need to show decision rules clearly. Use a **Random Forest** when predictive performance and stability matter more than explaining every individual rule.

**Remember:** *A single tree is easier to explain; a forest is usually more stable and accurate. Choose based on the balance between interpretability and predictive performance.*

---

## Summary Table

| Topic | One-line takeaway |
|---|---|
| Ensemble learning | Combines multiple models to produce a stronger final prediction |
| Bagging | Trains independent models on bootstrap samples in parallel, then combines them |
| Bootstrap sampling | Samples rows with replacement, giving every tree a different view of the data |
| Feature randomness | Random subsets of features reduce similarity between trees |
| Classification | Every tree votes; the class with the most support wins |
| Regression | Every tree predicts a number; the forest averages the numbers |
| OOB score | Uses rows not seen by individual trees as a convenient training-time validation estimate |
| Evaluation | Use held-out test data; inspect appropriate metrics, not only training accuracy |
| Hyperparameter tuning | Tune depth, samples per node, number of trees, and feature subset with cross-validation |
| Feature importance | Start with impurity importance, then verify with permutation importance |
| Decision tree vs. forest | Tree = explainable; forest = typically more stable and accurate |

### Where to Go Next

- **Decision Trees** — review splitting, Gini/entropy, variance reduction, and pruning in `../decision_tree/`.
- **Classification metrics** — revisit precision, recall, F1-score, ROC curves, and AUC in `../logistic_regression/`.
- **Regularization and model workflow** — see Ridge, Lasso, Elastic Net, feature selection, and cross-validation in `../ridge_lasso_elasticnet/`.
- **Boosting** — continue with AdaBoost, Gradient Boosting, and XGBoost to learn the sequential ensemble family.
- **Model comparison** — compare Random Forest with KNN (`../k_nearest_neighbour/`) and SVM (`../support_vector_machine/`) on the same cleaned dataset.
