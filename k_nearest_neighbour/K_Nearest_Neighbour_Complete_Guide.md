# K Nearest Neighbour (KNN) — Complete Guide

A short, memorable reference for KNN: the core intuition, how it does both classification and regression, the tree-based optimizations that make it fast, and practical implementation. Technical but plain-spoken. Builds on the classification concepts in `../logistic_regression/`, `../support_vector_machine/`, and `../naive_bayes_theorem/`.

**The one idea to hold onto:** KNN doesn't "learn" a formula at all — it just remembers all the training data, and for any new point, looks at its **K closest neighbors** and copies their answer (majority vote for classification, average for regression). It's the "ask the people standing nearest to you" algorithm.

---

## 1. KNN Classification and Regression — In-Depth Intuition

### Classification — vote with your neighbors

Given a training set with features and a category label (e.g. 0 or 1, or multiple classes), predicting a new point's class takes 3 steps:

1. **Pick K** — a hyperparameter, any integer > 0 (K=1, K=3, K=5, …). Found via hyperparameter tuning (try several K values, keep whichever gives the best accuracy).
2. **Find the K nearest neighbors** of the new point, using a distance formula (see below).
3. **Majority vote** — count how many of those K neighbors belong to each class; the new point is assigned the **class with the most votes**.

> **Example (K=5):** of the 5 nearest neighbors to a new point, 2 belong to class 0 and 3 belong to class 1 → predict **class 1** (majority wins).

### Regression — average with your neighbors

Same first two steps (pick K, find the K nearest neighbors) — but instead of voting, **average their target values** to get the prediction. If there are heavy outliers among the neighbors, use the **median** instead of the mean for a more robust estimate.

**Remember:** *KNN classification = majority vote among K nearest neighbors. KNN regression = average (or median) of K nearest neighbors' values. No formula is learned — the training data itself IS the model.*

### The two distance formulas

**Euclidean distance** — straight-line ("as the crow flies") distance, the hypotenuse of a right triangle between two points:

$$d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$$

(extends naturally to more dimensions by adding more squared-difference terms). Used when movement between points is unconstrained — e.g. flight paths between cities.

**Manhattan distance** — sum of the distances along each axis separately (like navigating city blocks, only moving horizontally or vertically, never diagonally):

$$d = |x_2-x_1| + |y_2-y_1|$$

Used when movement is constrained to a grid — e.g. a car/Uber navigating city streets, which can't cut diagonally through buildings.

**Remember:** *Euclidean = straight-line distance (unrestricted movement). Manhattan = grid/block distance (axis-constrained movement). Which one's better is itself a hyperparameter — try both, keep whichever gives higher accuracy.*

### The catch: it's slow at scale

To find the K nearest neighbors, naively you must calculate the distance from the query point to **every single training point** — that's **O(n)** time complexity per prediction. With millions of data points, this becomes painfully slow. The fix: **KD-Tree** and **Ball Tree** — see Section 2.

---

## 2. Optimizing KNN — KD-Tree and Ball Tree

Both structures avoid computing distance to *every* point by organizing the training data into a **binary tree**, so a search only has to walk down one branch instead of scanning everything.

### KD-Tree

Built by **repeatedly splitting the data on the median**, alternating between feature dimensions each level down:

1. Take feature 1 (e.g. F1/x-axis). Find its **median** value among all points. Draw a splitting line there — this creates two regions (left/below vs. right/above the median).
2. Within each region, now split on feature 2 (F2/y-axis) using **its** median.
3. Keep alternating dimensions (F1, F2, F1, F2, …) at each deeper level, each split creating a new branch in the binary tree.

The result is a tree where each node is a training point, and its left/right children are points that fell below/above the median split at that level.

**Searching a new query point:** walk down the tree (compare against each split's dimension and median) to quickly locate the region it falls in — this gives the *first* nearest neighbor almost immediately. To find the *next* nearest neighbors, the algorithm **backtracks** up the tree, checking whether nearby branches might contain closer points.

**Remember:** *KD-Tree = binary tree built by alternately splitting on the median of each feature dimension. Fast to find the first neighbor; needs backtracking for subsequent ones.*

### Ball Tree — no backtracking needed

Ball Tree groups points differently: it repeatedly **clusters the nearest points together**, bottom-up, forming a hierarchy of groups (a group of 2 points, merged into a group of 4, merged into a group of 8, and so on) until everything is one top-level group.

**Searching a new query point:** identify which bottom-level cluster it's closest to, then only calculate distances to the members of *that* cluster (and neighboring clusters) — no need to backtrack through the whole structure like KD-Tree does.

**Remember:** *Ball Tree = bottom-up clustering of nearby points into nested groups. Generally preferred over KD-Tree because it skips the backtracking step.*

### Why this matters

| Approach | How it finds neighbors | Time complexity |
|---|---|---|
| Brute force | Compute distance to every single point | O(n) — slow at scale |
| KD-Tree | Binary search down alternating-dimension splits, then backtrack | Much faster, still needs backtracking |
| Ball Tree | Binary search down nested distance-based clusters | Fastest, no backtracking needed |

---

## 3. KNN Classifier and Regressor — Practical Implementation

### Key hyperparameters (same for both classifier and regressor)

| Parameter | What it controls |
|---|---|
| `n_neighbors` | K — the number of neighbors to vote/average over. Default = 5. Tune via GridSearchCV. |
| `p` | Distance metric: **p=2** → Euclidean (default), **p=1** → Manhattan. |
| `algorithm` | `'auto'`, `'ball_tree'`, `'kd_tree'`, or `'brute'`. `'auto'` lets sklearn pick the best one based on your data at fit time. |
| `weights` | `'uniform'` (every neighbor's vote counts equally) or `'distance'` (closer neighbors count more). |

### Classifier

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

X, y = make_classification(n_samples=1000, n_features=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y)

knn_clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto', p=2)  # p=2 -> Euclidean
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Regressor

```python
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

X, y = make_regression(n_samples=1000, n_features=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y)

knn_reg = KNeighborsRegressor(n_neighbors=6, algorithm='auto')
knn_reg.fit(X_train, y_train)
y_pred = knn_reg.predict(X_test)

print(r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred))
```

### Tuning K with GridSearchCV

```python
param_grid = {'n_neighbors': [3, 5, 6, 7, 9, 11], 'p': [1, 2]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
grid.best_params_
```

**The practical lesson:** don't just accept the default K=5 — small changes in K (5 vs. 6) can noticeably shift accuracy (e.g. ~90% vs ~89% in practice). Always sweep several K values (and both `p` values) via GridSearchCV rather than guessing.

**Remember:** *`n_neighbors` (K) and `p` (distance metric) are the hyperparameters to tune. `algorithm='auto'` is a safe default — it picks Ball Tree, KD-Tree, or brute force based on your data automatically.*

---

## Summary Table

| Topic | One-line takeaway |
|---|---|
| Classification | Majority vote among the K nearest neighbors |
| Regression | Average (or median, if outliers) of the K nearest neighbors' values |
| Euclidean distance | Straight-line distance — unrestricted movement |
| Manhattan distance | Sum of axis-wise distances — grid/block-constrained movement |
| The scaling problem | Naive search = O(n) distance checks per prediction — too slow at scale |
| KD-Tree | Binary tree split on alternating feature medians; needs backtracking |
| Ball Tree | Binary tree of nested nearest-point clusters; no backtracking needed |
| Implementation | Tune `n_neighbors` (K) and `p` (distance metric) via GridSearchCV; `algorithm='auto'` picks the fastest search structure |

### Where to Go Next
- **Feature scaling** — KNN is distance-based, so unscaled features (e.g. one ranging 0–1, another 0–100,000) will dominate distance calculations; always standardize/normalize first.
- **Curse of dimensionality** — KNN's distance metric becomes less meaningful as feature count grows very large; consider dimensionality reduction first.
- **Precision/Recall/ROC-AUC** — the same classification metrics from `../logistic_regression/` apply to KNN classifier predictions.
- **Comparing classifiers** — benchmark KNN against Logistic Regression, SVM (`../support_vector_machine/`), and Naive Bayes (`../naive_bayes_theorem/`) on the same dataset.
