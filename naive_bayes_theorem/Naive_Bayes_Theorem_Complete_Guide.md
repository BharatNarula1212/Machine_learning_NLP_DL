# Naive Bayes Theorem — Complete Guide

A short, memorable reference for Naive Bayes: the probability foundations, Bayes' Theorem, how it becomes a classifier, its three variants, and practical implementation. Technical but plain-spoken. Builds on the classification concepts in `../logistic_regression/` and `../support_vector_machine/`.

**The one idea to hold onto:** Naive Bayes answers "given what I observed (the features), what's the most likely class?" by flipping the question around with Bayes' Theorem — it's easier to know "how likely are these features, *given* the class" (from training data) than the reverse, so Bayes' Theorem lets you compute one from the other.

---

## 1. Understanding Bayes' Theorem

### Independent vs. dependent events

- **Independent events** — one outcome doesn't affect another. Rolling a die: P(1) = P(2) = P(3) = 1/6 every single time, no matter what was rolled before.
- **Dependent events** — one outcome changes the probability of the next. Drawing marbles from a bag *without replacement*: removing an orange marble changes the odds for the next draw, since the bag's contents shrank.

### Conditional probability — the building block

For dependent events, the probability of event B happening *given that A already happened* is written **P(B|A)** — read "probability of B given A." Combining two dependent events in sequence:

$$P(A \text{ and } B) = P(A) \times P(B|A)$$

> **Example — bag of 3 orange + 2 yellow marbles.** P(orange first) = 3/5. After removing it, 4 marbles remain (2 orange, 2 yellow), so P(yellow given orange already drawn) = 2/4 = 1/2. Combined: P(orange, then yellow) = 3/5 × 1/2 = **3/10**.

### Deriving Bayes' Theorem

Since "A and B" is symmetric — $P(A \text{ and } B) = P(B \text{ and } A)$ — we can expand both sides using conditional probability and set them equal:

$$P(A) \times P(B|A) = P(B) \times P(A|B)$$

Rearranging to solve for $P(A|B)$ gives **Bayes' Theorem**:

$$P(A|B) = \frac{P(A) \times P(B|A)}{P(B)}$$

**Remember:** *Dependent events chain as P(A)·P(B|A). Bayes' Theorem just rearranges that chain to flip the direction: turn P(B|A) — often easy to know — into P(A|B) — often what you actually want.*

---

## 2. From Bayes' Theorem to a Classifier

In a classification problem, you have independent features $x_1, x_2, x_3, \ldots$ and want to predict the output class **y**. Plug that into Bayes' Theorem:

$$P(y \mid x_1, x_2, x_3) = \frac{P(y) \times P(x_1, x_2, x_3 \mid y)}{P(x_1, x_2, x_3)}$$

### The "naive" assumption

Computing $P(x_1, x_2, x_3 \mid y)$ jointly is hard — features usually interact. Naive Bayes makes a simplifying (and yes, "naive") assumption: **treat every feature as independent of the others, given the class.** This lets the joint term break apart into a simple product:

$$P(y \mid x_1, x_2, x_3) = \frac{P(y) \times P(x_1|y) \times P(x_2|y) \times P(x_3|y)}{P(x_1) \times P(x_2) \times P(x_3)}$$

### Dropping the denominator

The denominator $P(x_1)\times P(x_2)\times P(x_3)$ is the **same constant** no matter which class you're evaluating (yes vs. no, or class A vs. B vs. C) — so for the purpose of *comparing* classes, it can be dropped entirely:

$$P(y \mid x_1, x_2, x_3) \propto P(y) \times P(x_1|y) \times P(x_2|y) \times P(x_3|y)$$

**How to predict:** compute this score for **every possible class**, then pick the class with the **highest** score. To turn the raw scores into clean percentages, normalize by dividing each score by the sum of all scores.

> **Worked example — classic "Play Tennis" dataset.** New data point: Outlook = Sunny, Temperature = Hot.
> - P(Yes) = 9/14. P(Sunny|Yes) = 2/9. P(Hot|Yes) = 2/9. → score = 9/14 × 2/9 × 2/9 ≈ **0.031**
> - P(No) = 5/14. P(Sunny|No) = 3/5. P(Hot|No) = 2/5. → score = 5/14 × 3/5 × 2/5 ≈ **0.085**
> - Normalize: P(Yes) = 0.031/(0.031+0.085) ≈ **27%**, P(No) ≈ **73%**.
> - **Prediction: No** — the model says don't expect them to play tennis (higher probability wins).

**Remember:** *Naive Bayes assumes features are conditionally independent given the class — multiply P(class) × P(each feature|class), skip the shared denominator, pick the class with the highest score.*

---

## 3. The Three Variants of Naive Bayes

The core Bayes' Theorem math is identical across all three — what changes is **how P(feature|class) is calculated**, based on what kind of data the feature is.

### Bernoulli Naive Bayes

**Use when:** features follow a **Bernoulli distribution** — i.e. every feature is strictly **binary** (0/1, yes/no, pass/fail, male/female). Common after converting categorical or text data into a **sparse matrix** of 0s and 1s.

### Multinomial Naive Bayes

**Use when:** the input is **text data** for a classification problem (e.g. spam vs. ham email classification). Raw text first gets converted into numeric vectors via NLP techniques — **Bag of Words**, **TF-IDF**, or **Word2Vec** — then Multinomial NB works on the resulting word-count-style features.

### Gaussian Naive Bayes

**Use when:** features are **continuous numeric values** that roughly follow a **Gaussian (bell-curve/normal) distribution** — e.g. age, height, weight, sepal/petal measurements. If a feature follows a different distribution (e.g. skewed or exponential), transform it toward normal first (log transform, etc.) before using this variant.

### Choosing between them

| Your features look like… | Use |
|---|---|
| All binary (0/1) | **Bernoulli** NB |
| Text converted to word vectors | **Multinomial** NB (or Bernoulli, for text too) |
| Continuous numeric, bell-shaped | **Gaussian** NB |
| Mixed — go with whichever dominates | Pick the variant matching the **majority** of your features |

**Remember:** *Bernoulli = binary features. Multinomial = text/word-count features. Gaussian = continuous bell-curve features. Same theorem underneath — only the P(feature\|class) calculation changes.*

---

## 4. Practical Implementation

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Iris features (sepal/petal length & width) are continuous → GaussianNB is the right variant
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

For the other variants, the API is identical — only the import and class name change:

```python
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
# BernoulliNB()   — for binary/sparse 0-1 features
# MultinomialNB() — for text-derived word-count/TF-IDF features
```

**The workflow lesson:** the *implementation* is trivially short (import, `.fit()`, `.predict()`) — the real work is (1) correctly identifying which distribution your features follow, and (2) doing the right preprocessing first (one-hot/label encoding for categoricals, BoW/TF-IDF for text) so the chosen variant's assumptions actually hold.

**Remember:** *Match the variant to the feature type BEFORE fitting — encoding/vectorizing correctly matters far more than the one-line `.fit()` call itself.*

---

## Summary Table

| Topic | One-line takeaway |
|---|---|
| Independent vs dependent events | Independent: unaffected by prior outcomes. Dependent: P(A∩B) = P(A)·P(B\|A) |
| Bayes' Theorem | P(A\|B) = P(A)·P(B\|A) / P(B) — flips a hard-to-know probability into an easy-to-compute one |
| Naive assumption | Treat all features as independent given the class → joint probability becomes a simple product |
| Making a prediction | Score every class with P(y)·∏P(xᵢ\|y), drop the shared denominator, pick the highest score |
| Bernoulli NB | Binary (0/1) features |
| Multinomial NB | Text data converted to word-count vectors (BoW, TF-IDF) |
| Gaussian NB | Continuous, bell-curve-shaped numeric features |
| Implementation | Same 3-line sklearn API for all variants — the hard part is correct feature preprocessing |

### Where to Go Next
- **NLP text preprocessing** — Bag of Words, TF-IDF, Word2Vec (needed before Multinomial/Bernoulli NB on text).
- **Precision/Recall/F1/ROC-AUC** — the same classification metrics from `../logistic_regression/` apply to Naive Bayes predictions.
- **Laplace smoothing** — handles the zero-probability problem when a feature value never appeared with a class in training data.
- **Comparing classifiers** — benchmark Naive Bayes against Logistic Regression and SVM (see `../support_vector_machine/`) on the same dataset.
