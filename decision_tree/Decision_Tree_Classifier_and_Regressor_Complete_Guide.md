# Decision Tree Classifier and Regressor — Complete Guide

A short, memorable reference for Decision Trees: the core intuition, splitting criteria, information gain, numerical feature splits, pruning, the regressor variant, and practical implementation. Technical but plain-spoken. Builds on the classification concepts in `../logistic_regression/`, `../support_vector_machine/`, `../naive_bayes_theorem/`, and `../k_nearest_neighbour/`.

**The one idea to hold onto:** a decision tree is just a nested chain of `if/elif/else` questions, automatically learned from data. At every node it asks "which yes/no question about my features splits the data into the purest possible groups?" — and keeps asking that question, recursively, until each group is pure (or good enough).

---

## 1. Introduction to Decision Tree

Decision Tree solves **both classification and regression**. It builds itself the same way you'd write nested `if/elif/else` code:

```
if age <= 15:      "in school"
elif age <= 21:    "in college"
else:              "passed college"
```

That's literally the shape of a tree: a **root node** asks a yes/no question about a feature, splitting the data into branches; each branch either becomes a **leaf node** (a final prediction) or gets split again by another question.

### Two flavors: ID3 vs. CART

- **ID3** — can split a node into more than 2 branches.
- **CART** (what sklearn's `DecisionTreeClassifier`/`Regressor` uses) — every split is strictly **binary** (yes/no).

### The two questions every split must answer

1. **Is this split "pure"?** — Does it cleanly separate the classes, or is it still mixed? Measured with **Entropy** or **Gini Impurity** (Section 2).
2. **Which feature should I split on next?** — Out of all available features, which one gives the best split? Measured with **Information Gain** (Section 3).

**Remember:** *A decision tree = automatically-learned nested if/elif/else. CART = binary splits only (sklearn's default). Purity check → Entropy/Gini. Feature choice → Information Gain.*

---

## 2. Entropy and Gini Impurity — Measuring "Purity"

A split is **pure** if every point in a resulting group shares the same class (e.g. all "yes"); **impure** if the group is a mix of classes.

### Entropy

$$H(S) = -p_+ \log_2(p_+) - p_- \log_2(p_-)$$

(For more than 2 classes, add one more `-p_i·log₂(p_i)` term per class.) `p_+` and `p_-` are the proportions of each class in that node.

- **Perfectly pure node** (all one class) → Entropy = **0**.
- **Worst possible mix** (50/50 split, for binary) → Entropy = **1** (the maximum).
- **Range: 0 to 1.**

### Gini Impurity

$$\text{Gini} = 1 - \sum_{i=1}^{n} p_i^2$$

Same idea, different formula (no logarithms — faster to compute).

- **Perfectly pure node** → Gini = **0**.
- **Worst possible mix** (50/50) → Gini = **0.5** (the maximum).
- **Range: 0 to 0.5.**

### Which to use?

| | Entropy | Gini Impurity |
|---|---|---|
| Formula | Uses log₂ (slower) | No logs (faster) |
| Best for | Smaller datasets | Larger datasets |
| sklearn default | — | **Gini** (most common choice in practice) |

**Remember:** *Entropy ranges 0–1, uses logs, fine for small data. Gini ranges 0–0.5, faster, sklearn's default — use it for larger datasets.*

---

## 3. Information Gain — Choosing Which Feature to Split On

With multiple features to choose from (Outlook, Temperature, Humidity, Wind…), **which one should the tree split on first?** Answer: whichever gives the **biggest reduction in impurity** — that reduction is called **Information Gain**.

$$\text{Gain}(S, \text{feature}) = H(S) - \sum_{v \in \text{values}} \frac{|S_v|}{|S|} \times H(S_v)$$

In words: *(entropy of the parent node)* minus *(the weighted-average entropy of the child nodes after the split)*. The weight for each child is its share of the total samples.

**How to use it:** compute Information Gain for **every candidate feature**, and split on whichever feature produces the **highest** gain (i.e. the biggest drop in impurity). Repeat this process at every new node, recursively, using whichever features/thresholds remain relevant.

**Remember:** *Information Gain = parent's impurity − weighted-average impurity of the children. Split on the feature that maximizes this value; the tree builds itself node by node this way.*

---

## 4. Decision Tree Split for Numerical Features

Categorical features (Sunny/Overcast/Rain) split naturally into their existing categories. **Continuous numerical features** (e.g. 2.3, 3.6, 4.0…) need a different approach:

1. **Sort** the feature's values in ascending order.
2. **Try each value as a candidate threshold** — for every value, test the binary split "≤ this value" vs. "> this value" (building, conceptually, one mini decision tree per candidate threshold).
3. **Compute Information Gain for each candidate threshold's split.**
4. **Pick the threshold with the highest Information Gain** as the actual split point for that node.

> **Example:** trying threshold 2.3 gives one Information Gain; trying 3.6 gives another. Whichever threshold (say, 4) yields the best gain becomes `feature ≤ 4` at that node.

**The tradeoff:** this means testing *many* candidate thresholds per numerical feature — computationally expensive on large datasets, since you're essentially comparing many mini-splits before picking the best one.

**Remember:** *Numerical features get split by trying every sorted value as a candidate threshold, computing Information Gain for each, and keeping the best one. More thorough, but slower than categorical splits.*

---

## 5. Post-Pruning and Pre-Pruning — Fighting Overfitting

Left unchecked, a decision tree splits **all the way down** until every leaf is 100% pure — memorizing the training data. Result: **overfitting** — very high train accuracy (low bias), poor test accuracy (high variance).

### Post-Pruning

**Build the full tree first, then cut back.** After construction, look at deep nodes where the split barely helped (e.g. 9 "yes" vs. 2 "no" — overwhelmingly one class already) and **remove the unnecessary further splits**, letting that node become a leaf based on the majority class.

- ✅ Simple concept, done after the fact.
- ❌ Only practical for **smaller datasets** — building the entire tree first, then pruning, is expensive on large data.

### Pre-Pruning

**Constrain the tree while it's being built**, via hyperparameters — never let it grow past sensible limits in the first place:

| Hyperparameter | Controls |
|---|---|
| `max_depth` | Maximum depth of the tree |
| `max_features` | Max features considered per split |
| `min_samples_split` | Minimum samples required to split a node further |
| `min_samples_leaf` | Minimum samples required in a leaf |
| `criterion` | `'gini'`, `'entropy'`, or `'log_loss'` |
| `splitter` | `'best'` or `'random'` |

This is essentially **hyperparameter tuning** (via GridSearchCV/RandomizedSearchCV) applied *during* tree construction.

**Remember:** *Post-pruning = build fully, then trim (good for small data). Pre-pruning = constrain via hyperparameters (max_depth, min_samples_split, etc.) while building — the standard, scalable approach.*

---

## 6. Decision Tree Regressor — Variance Reduction

For **regression** (continuous output), Entropy/Gini/Information Gain don't apply — there's no fixed set of classes to measure purity over. Instead, use **Variance Reduction**.

### The variance at a node

$$\text{Variance} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2 \quad \text{(this is just Mean Squared Error against the node's own average)}$$

### Variance Reduction — comparing candidate splits

$$\text{Variance Reduction} = \text{Var(parent)} - \sum_{i} w_i \times \text{Var(child}_i\text{)}$$

Where $w_i$ is each child's share of the total samples (same weighting idea as Information Gain).

**How to use it:** just like numerical feature splitting for classification, try multiple candidate thresholds, compute the Variance Reduction for each, and **pick the split with the highest Variance Reduction**.

> **Example:** comparing threshold ≤2 (Variance Reduction = 0, no improvement) vs. ≤2.5 (Variance Reduction > 0) → the ≤2.5 split wins since it reduces variance more.

### Making a prediction

At a leaf node, the regressor's output is simply the **average** (mean) of all the target values that landed in that leaf.

**Remember:** *Regression trees use Variance Reduction (essentially MSE-based) instead of Information Gain. A leaf's prediction is the average of its training targets.*

---

## 7 & 8. Practical Implementation — Classifier (with Pre-Pruning & Hyperparameter Tuning)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# No scaling/normalization needed for decision trees
tree_clf = DecisionTreeClassifier()   # default: fully grown tree (risk of overfitting)
tree_clf.fit(X_train, y_train)

plt.figure(figsize=(15, 10))
plot_tree(tree_clf, filled=True)   # visualize the splits, Gini values, and leaves
```

**Pre-pruning via GridSearchCV** — sweep the key hyperparameters together:

```python
params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2, 3, 4, 5],
    'max_features': ['sqrt', 'log2', None],
}
grid = GridSearchCV(DecisionTreeClassifier(), params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
grid.best_params_, grid.best_score_

y_pred = grid.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**The practical lesson:** an unconstrained tree splits fully and can overfit; pre-pruning through GridSearchCV over `max_depth`, `criterion`, `splitter`, and `max_features` together consistently finds a better-generalizing tree than eyeballing one parameter at a time.

---

## 9. Practical Implementation — Regressor (Diabetes Prediction)

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = load_diabetes()
X, y = data.data, data.target   # 10 features (age, sex, bmi, bp, s1-s6); y = disease progression after 1 year
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

params = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2, 3, 4, 5, 6, 10, 15, 20, 25],
    'max_features': ['sqrt', 'log2', None],
}
grid = GridSearchCV(DecisionTreeRegressor(), params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
grid.best_params_

y_pred = grid.predict(X_test)
print(r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred))
```

**Note the regressor-specific `criterion` options:** `squared_error` and `friedman_mse` both implement the variance-reduction idea from Section 6 (mean-squared-error-based splitting), while classification's `criterion` options were Gini/Entropy — same tree-building logic, different math because the output is continuous.

**Remember:** *Same GridSearchCV workflow for classifier and regressor — only the criterion options and scoring metric change to match continuous vs. categorical output.*

---

## Summary Table

| Topic | One-line takeaway |
|---|---|
| Core idea | A tree of learned yes/no questions; splits recursively until leaves are pure (or pruned) |
| Entropy vs Gini | Entropy (0–1, log-based, small data) vs Gini (0–0.5, faster, sklearn default) — both measure impurity |
| Information Gain | Parent impurity − weighted child impurity; pick the feature that maximizes it |
| Numerical splits | Sort values, try every value as a threshold, keep the one with highest Information Gain |
| Post-pruning | Build fully, then trim back — good for small datasets |
| Pre-pruning | Constrain growth via hyperparameters (max_depth, min_samples_split, etc.) — the scalable default |
| Regressor (variance reduction) | Same idea as Information Gain, but using variance/MSE instead of entropy/Gini |
| Implementation | No scaling needed; tune via GridSearchCV; `plot_tree()` to visualize the learned splits |

### Where to Go Next
- **Ensemble methods (Random Forest, Gradient Boosting, AdaBoost)** — combine many decision trees to reduce overfitting and boost accuracy; this is the natural next step after mastering a single tree.
- **Feature importance** — decision trees naturally expose which features drove the most splits.
- **Precision/Recall/ROC-AUC** — the same classification metrics from `../logistic_regression/` apply to Decision Tree Classifier predictions.
- **Comparing models** — benchmark Decision Trees against Logistic Regression, SVM (`../support_vector_machine/`), Naive Bayes (`../naive_bayes_theorem/`), and KNN (`../k_nearest_neighbour/`) on the same dataset.
