
# ASSIGNMENT
# Experiment 3: Breast Cancer Classification using SVM

# # Experiment 3: Breast Cancer Classification Using Support Vector Machine

# ## 1. Aim and Objective
# To implement a Support Vector Machine (SVM) classifier for predicting breast cancer diagnosis using the Breast Cancer Prediction dataset. Data preprocessing and exploratory data analysis are performed, different kernel functions (Linear, Polynomial, RBF, Sigmoid) are compared, classification results are visualized, and classifier performance is analysed using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC score.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, roc_auc_score)
import warnings
warnings.filterwarnings("ignore")

print("Libraries imported successfully.")

# ## 2. Dataset Description
# The Breast Cancer Prediction dataset consists of 10,000 patient records with clinical and diagnostic attributes. The target variable is `Biopsy_Result` (Benign / Malignant), a binary classification problem. `Patient_ID` is an identifier column.

file_path = "breast_cancer_prediction.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
df.head()

# ## 3. Data Preprocessing
# This section identifies and handles missing values, removes irrelevant features, encodes categorical variables, and standardizes numerical features. Each step is justified below.

# ### 3a. Identify and Handle Missing Values
# First, the missing values in each column are counted.

print("Missing values per column:")
print(df.isnull().sum())

print("\nPercentage missing per column:")
print((df.isnull().sum() / len(df) * 100).round(2))

# **Method used and justification:** Missing numerical values are imputed using the **median** of each column, computed **only from the training set** (after the train-test split) and then applied to both the training and test sets. Median imputation is used instead of mean because it is more robust to outliers and skewed clinical measurements (e.g. tumor size, blood pressure). Fitting the imputation statistics on the training set only (rather than the whole dataset) avoids data leakage from the test set into the training process. This is implemented later in Section 3d, after the train-test split is performed.

# ### 3b. Remove Irrelevant Features
# Features that carry no predictive information are identified and removed.

# Check for identifier-like / near-constant columns
print("Unique values per column:")
print(df.nunique()) # no of unique values in each column

print("\nIs Patient_ID unique for every row?", df["Patient_ID"].is_unique)

# **Justification:** `Patient_ID` is a unique identifier assigned to each record. It has no relationship with the biological or clinical outcome (`Biopsy_Result`) and would only add noise or cause overfitting if used as a model feature, so it is removed. All other columns are retained since `nunique()` confirms none of them are constant (zero-variance) or purely identifier-like, meaning they may carry genuine predictive signal.

# ### 3c. Encode Categorical Variables
# Categorical (non-numeric) columns are identified below.

categorical_cols_preview = df.drop(columns=["Biopsy_Result", "Patient_ID"]).select_dtypes(include=["object"]).columns
print("Categorical columns:", list(categorical_cols_preview))

for col in categorical_cols_preview:
    print("\n" + col + " - unique categories:", df[col].nunique())
    print(df[col].value_counts())

# **Technique used and justification:** Categorical features are encoded using **one-hot encoding** (`pd.get_dummies`, with `drop_first=True` to avoid the dummy variable trap). One-hot encoding is chosen over label/ordinal encoding because these categorical fields (e.g. gender, lifestyle/history categories) are **nominal** — their categories have no natural order or ranking. Assigning arbitrary integer labels (as in label encoding) would falsely imply an ordinal relationship, which could mislead a distance-based classifier like SVM. Since the number of unique categories per column is small, one-hot encoding does not cause excessive dimensionality.

# ### 3d. Feature/Target Split, Train-Test Split, Missing Value Imputation, and Standardization
# The target variable is separated, irrelevant columns are dropped, categorical variables are one-hot encoded, the data is split into training and test sets, missing values are imputed using training-set medians, and numerical features are standardized.

# Separate features and target, drop irrelevant identifier column
X = df.drop(columns=["Biopsy_Result", "Patient_ID"])
y = df["Biopsy_Result"].map({"Benign": 0, "Malignant": 1})

# One-hot encode categorical features
categorical_columns = X.select_dtypes(include=["object"]).columns
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True).astype(float)

# Train-test split (80-20, stratified on target)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Impute missing values using training-set medians only (avoids data leakage)
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Missing values in training set after imputation:", X_train.isnull().sum().sum())
print("Missing values in testing set after imputation:", X_test.isnull().sum().sum())

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features standardized. Mean of scaled tra ining features (~0):", round(X_train_scaled.mean(), 4))
print("Std of scaled training features (~1):", round(X_train_scaled.std(), 4))

# **Justification for standardization:** SVM is a distance/margin-based algorithm — it computes dot products and distances between data points to find the optimal separating hyperplane. Features on larger numeric scales (e.g. `Blood_Pressure` vs a 0/1 flag) would dominate this distance calculation if left unscaled. `StandardScaler` transforms each feature to zero mean and unit variance, fitted on the training set only and then applied to the test set, so all features contribute proportionally and no test-set information leaks into training.

# ## 4. Exploratory Data Analysis (EDA) and Visualization
# The dataset is examined using summary statistics, followed by histograms, bar charts, scatter plots, box plots, a correlation heatmap, and a pair plot.

# ### 4a. Summary Statistics

df.info()
print()
df.describe()

print("Target class distribution:")
print(df["Biopsy_Result"].value_counts())
print("\nTarget class proportion:")
print(df["Biopsy_Result"].value_counts(normalize=True).round(3))

# ### 4b. Histograms — Distribution of Important Numerical Features

important_features = ["Age", "BMI", "Tumor_Size_cm", "Blood_Pressure", "Cholesterol"]
important_features = [f for f in important_features if f in df.columns]

for feature in important_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=feature, kde=True)
    plt.title("Distribution of " + feature)
    plt.show()

# ### 4c. Bar Charts — Class Distribution and Categorical Features

# Bar chart: target class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Biopsy_Result")
plt.title("Class Distribution: Benign vs Malignant")
plt.xlabel("Biopsy Result")
plt.ylabel("Number of Records")
plt.show()

# Bar charts: categorical feature counts
categorical_cols_eda = df.drop(columns=["Biopsy_Result", "Patient_ID"]).select_dtypes(include=["object"]).columns

for col in categorical_cols_eda:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col)
    plt.title("Count of Records by " + col)
    plt.xticks(rotation=30)
    plt.show()

# ### 4d. Scatter Plots — Relationships Between Numerical Features and Target

# Scatter plot: Tumor Size vs Age, colored by diagnosis
if "Tumor_Size_cm" in df.columns and "Age" in df.columns:
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="Age", y="Tumor_Size_cm", hue="Biopsy_Result", alpha=0.5)
    plt.title("Tumor Size vs Age by Biopsy Result")
    plt.show()

# Scatter plot: BMI vs Cholesterol, colored by diagnosis
if "BMI" in df.columns and "Cholesterol" in df.columns:
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="BMI", y="Cholesterol", hue="Biopsy_Result", alpha=0.5)
    plt.title("Cholesterol vs BMI by Biopsy Result")
    plt.show()

# ### 4e. Box Plots — Spread and Outliers in Important Numerical Features

for feature in important_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, y=feature)
    plt.title("Box Plot of " + feature)
    plt.show()

# Box plots split by target class, to see if the feature separates Benign vs Malignant
for feature in important_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Biopsy_Result", y=feature)
    plt.title(feature + " by Biopsy Result")
    plt.show()

# ### 4f. Correlation Heatmap of Numerical Features

numeric_df = df.select_dtypes(include=["int64", "float64"])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# ### 4g. Pair Plot of Important Numerical Features

# A random sample is used to keep the pair plot readable and fast to render
pairplot_sample = df.sample(n=min(500, len(df)), random_state=42)

sns.pairplot(pairplot_sample, vars=important_features, hue="Biopsy_Result", diag_kind="hist")
plt.show()

# ## 5. SVM Model Development
# Four SVM kernels — Linear, Polynomial, RBF, and Sigmoid — are trained with default parameters (C=1, gamma="scale") as a baseline before hyperparameter tuning.

kernels = ["linear", "poly", "rbf", "sigmoid"]
basic_results = []
basic_models = {}

for kernel in kernels:
    start = time.time()
    model = SVC(kernel=kernel, C=1, gamma="scale", degree=3, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    start = time.time()
    pred = model.predict(X_test_scaled)
    pred_time = time.time() - start

    basic_models[kernel] = model
    basic_results.append({
        "Kernel": kernel.capitalize(),
        "Training Accuracy": accuracy_score(y_train, model.predict(X_train_scaled)),
        "Testing Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "Training Time (sec)": train_time,
        "Prediction Time (sec)": pred_time
    })

basic_results_df = pd.DataFrame(basic_results)
basic_results_df.round(4)

# ## 6. Hyperparameter Tuning
# Search space: `C = {1, 10}`, `gamma = {scale, 0.01}`, `degree = {2, 3}` (Polynomial only). This is a reduced version of the full assignment search space (`C = {0.1, 1, 10, 100}`, `gamma = {scale, auto, 0.001, 0.01, 0.1}`, `degree = {2, 3, 4}`), chosen for computational feasibility: the full grid requires ~520 SVM fits on 8,000 training samples across 5-fold CV, and non-linear kernel SVMs scale poorly with sample count, making the full search impractically slow on a laptop. 5-fold stratified cross-validation is used with `GridSearchCV`, optimizing for accuracy.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Reduced search space for computational feasibility (see markdown above).
C_values = [1, 10]
gamma_values = ["scale", 0.01]
degree_values = [2, 3]

# Note: probability=True is intentionally left out during the grid search.
# It is not needed here (GridSearchCV only needs predict()/accuracy), and enabling
# it triggers an extra internal 5-fold Platt-scaling calibration inside every single
# SVM fit, which makes the search several times slower without changing which
# hyperparameters are found to be best. Probability estimates are added afterwards,
# once, only for the four final chosen models.
linear_search = GridSearchCV(
    SVC(kernel="linear", random_state=42),
    {"C": C_values}, cv=cv, scoring="accuracy", n_jobs=-1
)
linear_search.fit(X_train_scaled, y_train)

poly_search = GridSearchCV(
    SVC(kernel="poly", random_state=42),
    {"C": C_values, "gamma": gamma_values, "degree": degree_values},
    cv=cv, scoring="accuracy", n_jobs=-1
)
poly_search.fit(X_train_scaled, y_train)

rbf_search = GridSearchCV(
    SVC(kernel="rbf", random_state=42),
    {"C": C_values, "gamma": gamma_values},
    cv=cv, scoring="accuracy", n_jobs=-1
)
rbf_search.fit(X_train_scaled, y_train)

sigmoid_search = GridSearchCV(
    SVC(kernel="sigmoid", random_state=42),
    {"C": C_values, "gamma": gamma_values},
    cv=cv, scoring="accuracy", n_jobs=-1
)
sigmoid_search.fit(X_train_scaled, y_train)

searches = {
    "Linear": linear_search,
    "Polynomial": poly_search,
    "RBF": rbf_search,
    "Sigmoid": sigmoid_search
}

# Re-fit each best-parameter model once, this time with probability=True,
# so predict_proba() is available for the ROC curves and ROC-AUC later.
best_models = {}
for name, search in searches.items():
    best_params = search.best_params_
    kernel_name = search.estimator.kernel
    refit_model = SVC(kernel=kernel_name, probability=True, random_state=42, **best_params)
    refit_model.fit(X_train_scaled, y_train)
    best_models[name] = refit_model

# Table 1: Hyperparameter Tuning Summary
tuning_summary = []
for name, search in searches.items():
    tuning_summary.append({
        "Kernel": name,
        "Best C": search.best_params_.get("C", "NA"),
        "Best Gamma": search.best_params_.get("gamma", "NA"),
        "Best Degree": search.best_params_.get("degree", "NA"),
        "Best Cross Validation Accuracy": search.best_score_
    })

tuning_summary_df = pd.DataFrame(tuning_summary)
tuning_summary_df.round(4)

# ## 7. Visualizations — Model Results
# Confusion matrices, classification reports, and ROC curves are generated for the tuned models on the test set.

# Fit tuned models, generate predictions and probabilities on the test set
predictions = {}
probabilities = {}
train_times = {}
pred_times = {}

for name, model in best_models.items():
    start = time.time()
    model.fit(X_train_scaled, y_train)
    train_times[name] = time.time() - start

    start = time.time()
    predictions[name] = model.predict(X_test_scaled)
    pred_times[name] = time.time() - start

    probabilities[name] = model.predict_proba(X_test_scaled)[:, 1]

print("Predictions generated for:", list(predictions.keys()))

# Confusion matrices for each tuned kernel model
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for ax, (name, pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"], ax=ax)
    ax.set_title(name + " Kernel")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# Classification reports for each tuned kernel model
for name, pred in predictions.items():
    print("Classification Report -", name, "Kernel")
    print(classification_report(y_test, pred, target_names=["Benign", "Malignant"]))
    print("-" * 60)

# ROC curves for all tuned kernel models
plt.figure(figsize=(7, 6))

for name, prob in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_score = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=name + " (AUC = " + str(round(auc_score, 3)) + ")")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Accuracy comparison bar chart (Linear, Polynomial, RBF)
bar_kernels = ["Linear", "Polynomial", "RBF"]
bar_accuracies = [accuracy_score(y_test, predictions[k]) for k in bar_kernels]

plt.figure(figsize=(6, 4))
plt.bar(bar_kernels, bar_accuracies, color=["steelblue", "seagreen", "indianred"])
plt.title("Testing Accuracy Comparison")
plt.xlabel("Kernel")
plt.ylabel("Testing Accuracy")
plt.ylim(0, 1)
plt.show()

# Kernel comparison graph: Accuracy and F1-score for all four kernels
all_kernels = list(predictions.keys())
acc_scores = [accuracy_score(y_test, predictions[k]) for k in all_kernels]
f1_scores = [f1_score(y_test, predictions[k], zero_division=0) for k in all_kernels]

x = np.arange(len(all_kernels))
width = 0.35

plt.figure(figsize=(7, 5))
plt.bar(x - width/2, acc_scores, width, label="Accuracy")
plt.bar(x + width/2, f1_scores, width, label="F1 Score")
plt.xticks(x, all_kernels)
plt.xlabel("Kernel")
plt.ylabel("Score")
plt.title("Kernel Comparison: Accuracy vs F1 Score")
plt.legend()
plt.ylim(0, 1)
plt.show()

# ## 8. Performance Tables
# Table 2 reports 5-fold cross-validation performance on the training set. Table 3 reports final performance on the held-out test set.

# Table 2: Cross-Validation Performance (K = 5)
scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}

cv_results = []
for name, model in best_models.items():
    scores = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_results.append({
        "Kernel": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1 Score": scores["test_f1"].mean()
    })

cv_performance_df = pd.DataFrame(cv_results)
cv_performance_df.round(4)

# Table 3: Test Set Performance
test_results = []
for name, model in best_models.items():
    pred = predictions[name]
    prob = probabilities[name]
    test_results.append({
        "Kernel": name,
        "Training Accuracy": accuracy_score(y_train, model.predict(X_train_scaled)),
        "Testing Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "Training Time (sec)": train_times[name],
        "Prediction Time (sec)": pred_times[name]
    })

test_results_df = pd.DataFrame(test_results)
test_results_df.round(4)

# ## 9. Kernel Comparison
# The bar charts and tables above (Sections 7 and 8) compare all four kernels on accuracy, precision, recall, F1-score, and ROC-AUC. The kernel with the highest testing accuracy and F1-score in Table 3 is treated as the best-performing model for this dataset.

# Identify the best kernel based on testing accuracy
best_kernel_row = test_results_df.loc[test_results_df["Testing Accuracy"].idxmax()]
best_kernel_name = best_kernel_row["Kernel"]

print("Best performing kernel based on testing accuracy:", best_kernel_name)
print(best_kernel_row)

# ## 10. Analysis and Inference
# Misclassified samples for the best-performing kernel are shown below to help understand where the model makes errors.

# Misclassified samples table for the best kernel
best_pred = predictions[best_kernel_name]

misclassified_mask = (best_pred != y_test.values)
misclassified_indices = y_test.index[misclassified_mask]

misclassified_df = df.loc[misclassified_indices].copy()
misclassified_df["Predicted"] = pd.Series(best_pred, index=y_test.index)[misclassified_mask].map({0: "Benign", 1: "Malignant"})

print("Number of misclassified samples:", len(misclassified_df))
misclassified_df.head(20)

# ## 11. Observations
# 1. Missing values are handled using training-data medians only, to avoid data leakage.
# 2. `Patient_ID` is removed as an irrelevant identifier feature.
# 3. Categorical features are one-hot encoded before model training.
# 4. Features are standardized using `StandardScaler` before fitting the SVM.
# 5. Linear, Polynomial, RBF, and Sigmoid kernels are compared under identical preprocessing.
# 6. Hyperparameters (C, gamma, degree) are tuned using `GridSearchCV` with 5-fold stratified cross-validation.
# 7. Confusion matrices, ROC curves, and classification reports help analyze per-class classification errors.
# 8. The best kernel is selected using the final test-set accuracy and F1-score (see Table 3).

# ## 12. Conclusion
# An SVM classifier was implemented for breast cancer diagnosis using the Breast Cancer Prediction dataset. Data preprocessing — irrelevant feature removal, categorical encoding, training-based missing-value imputation, and feature standardization — was applied before training. Four kernels (Linear, Polynomial, RBF, Sigmoid) were compared before and after hyperparameter tuning. Based on the test-set results in Table 3, the kernel with the highest accuracy and F1-score is identified as the most suitable choice for this classification task.

# ## 13. References
# - Scikit-learn Documentation – Support Vector Machines: https://scikit-learn.org/stable/modules/svm.html
