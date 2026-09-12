# ============================================================
# EXPERIMENT 5
# Binary Classification using Naive Bayes and KNN
# Dataset: Spambase
# ============================================================

# -------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("spambase_csv.csv")

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum().sum())


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("class", axis=1)
y = df["class"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

print("\nMissing values after preprocessing:")
print(X.isnull().sum().sum())


# ============================================================
# 6. CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(6, 4))

sns.countplot(x=y)

plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


# ============================================================
# 7. FEATURE DISTRIBUTION
# ============================================================

# Display distributions of first 6 features

X.iloc[:, :6].hist(figsize=(12, 8), bins=20)

plt.suptitle("Feature Distributions")
plt.show()


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 9. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 10. NAIVE BAYES MODELS
# ============================================================

# ------------------------------------------------------------
# Gaussian Naive Bayes
# ------------------------------------------------------------

start = time.time()

gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)

gnb_train_time = time.time() - start

start = time.time()

gnb_pred = gnb.predict(X_test_scaled)
gnb_prob = gnb.predict_proba(X_test_scaled)[:, 1]

gnb_prediction_time = time.time() - start


# ------------------------------------------------------------
# Multinomial Naive Bayes
# ------------------------------------------------------------
# Multinomial NB requires non-negative values.
# Therefore, use the original non-negative features.

start = time.time()

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

mnb_train_time = time.time() - start

start = time.time()

mnb_pred = mnb.predict(X_test)
mnb_prob = mnb.predict_proba(X_test)[:, 1]

mnb_prediction_time = time.time() - start


# ------------------------------------------------------------
# Bernoulli Naive Bayes
# ------------------------------------------------------------
# Convert features into binary values

X_train_binary = (X_train > 0).astype(int)
X_test_binary = (X_test > 0).astype(int)

start = time.time()

bnb = BernoulliNB()
bnb.fit(X_train_binary, y_train)

bnb_train_time = time.time() - start

start = time.time()

bnb_pred = bnb.predict(X_test_binary)
bnb_prob = bnb.predict_proba(X_test_binary)[:, 1]

bnb_prediction_time = time.time() - start


# ============================================================
# 11. FUNCTION TO CALCULATE METRICS
# ============================================================

def calculate_metrics(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)

    return accuracy, precision, recall, f1, specificity, fpr


# ============================================================
# 12. NAIVE BAYES METRICS
# ============================================================

gnb_metrics = calculate_metrics(y_test, gnb_pred)
mnb_metrics = calculate_metrics(y_test, mnb_pred)
bnb_metrics = calculate_metrics(y_test, bnb_pred)


# ============================================================
# 13. NAIVE BAYES PERFORMANCE TABLE
# ============================================================

nb_results = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Specificity",
        "False Positive Rate",
        "Training Time (s)",
        "Prediction Time (s)"
    ],

    "Gaussian NB": [
        gnb_metrics[0],
        gnb_metrics[1],
        gnb_metrics[2],
        gnb_metrics[3],
        gnb_metrics[4],
        gnb_metrics[5],
        gnb_train_time,
        gnb_prediction_time
    ],

    "Multinomial NB": [
        mnb_metrics[0],
        mnb_metrics[1],
        mnb_metrics[2],
        mnb_metrics[3],
        mnb_metrics[4],
        mnb_metrics[5],
        mnb_train_time,
        mnb_prediction_time
    ],

    "Bernoulli NB": [
        bnb_metrics[0],
        bnb_metrics[1],
        bnb_metrics[2],
        bnb_metrics[3],
        bnb_metrics[4],
        bnb_metrics[5],
        bnb_train_time,
        bnb_prediction_time
    ]
})

print("\nNaive Bayes Performance:")
print(nb_results)


# ============================================================
# 14. CONFUSION MATRICES FOR NAIVE BAYES
# ============================================================

models_nb = {
    "Gaussian NB": gnb_pred,
    "Multinomial NB": mnb_pred,
    "Bernoulli NB": bnb_pred
}

for name, prediction in models_nb.items():

    cm = confusion_matrix(y_test, prediction)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix - " + name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# ============================================================
# 15. ROC CURVES FOR NAIVE BAYES
# ============================================================

fpr_g, tpr_g, _ = roc_curve(y_test, gnb_prob)
fpr_m, tpr_m, _ = roc_curve(y_test, mnb_prob)
fpr_b, tpr_b, _ = roc_curve(y_test, bnb_prob)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr_g,
    tpr_g,
    label="Gaussian NB (AUC = %.2f)" % auc(fpr_g, tpr_g)
)

plt.plot(
    fpr_m,
    tpr_m,
    label="Multinomial NB (AUC = %.2f)" % auc(fpr_m, tpr_m)
)

plt.plot(
    fpr_b,
    tpr_b,
    label="Bernoulli NB (AUC = %.2f)" % auc(fpr_b, tpr_b)
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Naive Bayes")
plt.legend()
plt.show()


# ============================================================
# 16. BASELINE KNN
# ============================================================

start = time.time()

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

knn_train_time = time.time() - start


start = time.time()

knn_pred = knn.predict(X_test_scaled)
knn_prob = knn.predict_proba(X_test_scaled)[:, 1]

knn_prediction_time = time.time() - start


knn_metrics = calculate_metrics(y_test, knn_pred)

print("\nBaseline KNN Performance")

print("Accuracy:", knn_metrics[0])
print("Precision:", knn_metrics[1])
print("Recall:", knn_metrics[2])
print("F1 Score:", knn_metrics[3])
print("Specificity:", knn_metrics[4])
print("False Positive Rate:", knn_metrics[5])
print("Training Time:", knn_train_time)
print("Prediction Time:", knn_prediction_time)


# ============================================================
# 17. ACCURACY VS K
# ============================================================

k_values = range(1, 21)

train_accuracy = []
test_accuracy = []

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train_scaled, y_train)

    train_accuracy.append(
        model.score(X_train_scaled, y_train)
    )

    test_accuracy.append(
        model.score(X_test_scaled, y_test)
    )


plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    train_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    k_values,
    test_accuracy,
    marker="o",
    label="Validation/Test Accuracy"
)

plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.legend()
plt.grid()
plt.show()


# ============================================================
# 18. GRID SEARCH FOR KNN
# ============================================================

param_grid = {
    "n_neighbors": list(range(1, 21)),
    "weights": ["uniform", "distance"],
    "p": [1, 2]
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

start = time.time()

grid_search.fit(X_train_scaled, y_train)

grid_time = time.time() - start

print("\nGrid Search Results")

print("Best Parameters:")
print(grid_search.best_params_)

print("Best K:", grid_search.best_params_["n_neighbors"])

print("Best CV Accuracy:",
      grid_search.best_score_)

print("Grid Search Time:",
      grid_time)


# ============================================================
# 19. RANDOMIZED SEARCH FOR KNN
# ============================================================

param_distribution = {
    "n_neighbors": list(range(1, 21)),
    "weights": ["uniform", "distance"],
    "p": [1, 2]
}

random_search = RandomizedSearchCV(
    KNeighborsClassifier(),
    param_distributions=param_distribution,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

start = time.time()

random_search.fit(X_train_scaled, y_train)

random_time = time.time() - start

print("\nRandomized Search Results")

print("Best Parameters:")
print(random_search.best_params_)

print("Best K:",
      random_search.best_params_["n_neighbors"])

print("Best CV Accuracy:",
      random_search.best_score_)

print("Randomized Search Time:",
      random_time)


# ============================================================
# 20. HYPERPARAMETER SEARCH COMPARISON
# ============================================================

search_results = pd.DataFrame({

    "Search Method": [
        "Grid Search",
        "Randomized Search"
    ],

    "Best K": [
        grid_search.best_params_["n_neighbors"],
        random_search.best_params_["n_neighbors"]
    ],

    "Best CV Accuracy": [
        grid_search.best_score_,
        random_search.best_score_
    ],

    "Best Parameters": [
        grid_search.best_params_,
        random_search.best_params_
    ]
})

print("\nKNN Hyperparameter Tuning Results")
print(search_results)


# ============================================================
# 21. BEST K FROM GRID SEARCH
# ============================================================

best_k = grid_search.best_params_["n_neighbors"]

print("\nOptimal K:", best_k)


# ============================================================
# 22. KNN USING KD TREE
# ============================================================

kd_model = KNeighborsClassifier(
    n_neighbors=best_k,
    weights=grid_search.best_params_["weights"],
    p=grid_search.best_params_["p"],
    algorithm="kd_tree"
)

start = time.time()

kd_model.fit(X_train_scaled, y_train)

kd_train_time = time.time() - start


start = time.time()

kd_pred = kd_model.predict(X_test_scaled)

kd_prediction_time = time.time() - start


kd_metrics = calculate_metrics(
    y_test,
    kd_pred
)

print("\nKDTree KNN")

print("Accuracy:", kd_metrics[0])
print("Precision:", kd_metrics[1])
print("Recall:", kd_metrics[2])
print("F1 Score:", kd_metrics[3])
print("Training Time:", kd_train_time)
print("Prediction Time:", kd_prediction_time)


# ============================================================
# 23. KNN USING BALL TREE
# ============================================================

ball_model = KNeighborsClassifier(
    n_neighbors=best_k,
    weights=grid_search.best_params_["weights"],
    p=grid_search.best_params_["p"],
    algorithm="ball_tree"
)

start = time.time()

ball_model.fit(X_train_scaled, y_train)

ball_train_time = time.time() - start


start = time.time()

ball_pred = ball_model.predict(X_test_scaled)

ball_prediction_time = time.time() - start


ball_metrics = calculate_metrics(
    y_test,
    ball_pred
)

print("\nBallTree KNN")

print("Accuracy:", ball_metrics[0])
print("Precision:", ball_metrics[1])
print("Recall:", ball_metrics[2])
print("F1 Score:", ball_metrics[3])
print("Training Time:", ball_train_time)
print("Prediction Time:", ball_prediction_time)


# ============================================================
# 24. KD TREE PERFORMANCE TABLE
# ============================================================

kd_table = pd.DataFrame({

    "Metric": [
        "Optimal k",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Training Time (s)",
        "Prediction Time (s)"
    ],

    "Value": [
        best_k,
        kd_metrics[0],
        kd_metrics[1],
        kd_metrics[2],
        kd_metrics[3],
        kd_train_time,
        kd_prediction_time
    ]
})

print("\nKDTree Performance")
print(kd_table)


# ============================================================
# 25. BALL TREE PERFORMANCE TABLE
# ============================================================

ball_table = pd.DataFrame({

    "Metric": [
        "Optimal k",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Training Time (s)",
        "Prediction Time (s)"
    ],

    "Value": [
        best_k,
        ball_metrics[0],
        ball_metrics[1],
        ball_metrics[2],
        ball_metrics[3],
        ball_train_time,
        ball_prediction_time
    ]
})

print("\nBallTree Performance")
print(ball_table)


# ============================================================
# 26. KD TREE VS BALL TREE
# ============================================================

tree_comparison = pd.DataFrame({

    "Criterion": [
        "Accuracy",
        "Training Time (s)",
        "Prediction Time (s)"
    ],

    "KDTree": [
        kd_metrics[0],
        kd_train_time,
        kd_prediction_time
    ],

    "BallTree": [
        ball_metrics[0],
        ball_train_time,
        ball_prediction_time
    ]
})

print("\nKDTree vs BallTree")
print(tree_comparison)


# ============================================================
# 27. CONFUSION MATRIX - OPTIMIZED KNN
# ============================================================

plt.figure(figsize=(5, 4))

sns.heatmap(
    confusion_matrix(y_test, kd_pred),
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix - Optimized KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# 28. ROC CURVE - OPTIMIZED KNN
# ============================================================

kd_prob = kd_model.predict_proba(X_test_scaled)[:, 1]

fpr_k, tpr_k, _ = roc_curve(y_test, kd_prob)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr_k,
    tpr_k,
    label="KNN (AUC = %.2f)" % auc(fpr_k, tpr_k)
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Optimized KNN")

plt.legend()
plt.show()


# ============================================================
# 29. TRAINING VS VALIDATION ACCURACY
# ============================================================

best_knn = KNeighborsClassifier(
    n_neighbors=best_k,
    weights=grid_search.best_params_["weights"],
    p=grid_search.best_params_["p"]
)

best_knn.fit(X_train_scaled, y_train)

training_accuracy = best_knn.score(
    X_train_scaled,
    y_train
)

validation_accuracy = best_knn.score(
    X_test_scaled,
    y_test
)

print("\nTraining Accuracy:", training_accuracy)
print("Validation Accuracy:", validation_accuracy)


plt.figure(figsize=(6, 5))

plt.bar(
    ["Training", "Validation"],
    [training_accuracy, validation_accuracy]
)

plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.show()


# ============================================================
# 30. FINAL MODEL COMPARISON
# ============================================================

final_results = pd.DataFrame({

    "Model": [
        "Gaussian NB",
        "Multinomial NB",
        "Bernoulli NB",
        "Baseline KNN",
        "KNN KDTree",
        "KNN BallTree"
    ],

    "Accuracy": [
        gnb_metrics[0],
        mnb_metrics[0],
        bnb_metrics[0],
        knn_metrics[0],
        kd_metrics[0],
        ball_metrics[0]
    ],

    "Precision": [
        gnb_metrics[1],
        mnb_metrics[1],
        bnb_metrics[1],
        knn_metrics[1],
        kd_metrics[1],
        ball_metrics[1]
    ],

    "Recall": [
        gnb_metrics[2],
        mnb_metrics[2],
        bnb_metrics[2],
        knn_metrics[2],
        kd_metrics[2],
        ball_metrics[2]
    ],

    "F1 Score": [
        gnb_metrics[3],
        mnb_metrics[3],
        bnb_metrics[3],
        knn_metrics[3],
        kd_metrics[3],
        ball_metrics[3]
    ]
})

print("FINAL MODEL COMPARISON")

print(final_results)


# ============================================================
# 31. ACCURACY COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.bar(
    final_results["Model"],
    final_results["Accuracy"]
)

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.xticks(rotation=30)

plt.show()


# ============================================================
# 32. BEST MODEL
# ============================================================

best_model_index = final_results["Accuracy"].idxmax()

print("\nBest Model:")
print(final_results.loc[best_model_index])