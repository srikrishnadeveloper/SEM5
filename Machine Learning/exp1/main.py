# ASSIGNMENT - 1
# Machine Learning EDA on Diabetes dataset
# main.py - clean runnable version

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI windows
import matplotlib.pyplot as plt

# Load dataset

df = pd.read_csv("Diabetes_prediction(2).csv")
print("Dataset loaded successfully\n")

print("First Five Records")
print(df.head())

print("\nNumber of Rows")
print(df.shape[0])

print("\nNumber of Columns")
print(df.shape[1])

print("\nFeature Names")
print(df.columns.tolist())

# Identify target column

target = "Diagnosis"
print("\nTarget Column :", target)

# Missing values

print("\nMissing Values")
print(df.isnull().sum())

df = df.fillna(df.mean(numeric_only=True))

print("\nMissing Values After Handling")
print(df.isnull().sum())

# Remove irrelevant features

if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)
    print("\nID column removed")
else:
    print("\nNo irrelevant feature found")

# Remove duplicates

print("\nDuplicates Before Removal :", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicates After Removal :", df.duplicated().sum())

# Encode categorical variables

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
categorical = df.select_dtypes(include="object").columns

for col in categorical:
    df[col] = encoder.fit_transform(df[col])

print("\nCategorical Features Encoded")

# Standardize numerical input features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numerical_features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\nNumerical Features Standardized")
print(df.head())

# Correlation with target

correlation = df.corr(numeric_only=True)

print("\nCorrelation with Target")
print(correlation[target].sort_values(ascending=False))

# Mutual information

from sklearn.feature_selection import mutual_info_classif

X = df.drop(target, axis=1)
y = df[target]

mi = mutual_info_classif(X, y)
mi_scores = pd.Series(mi, index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)

print("\nMutual Information Scores")
print(mi_scores)

# Feature importance table

importance = pd.DataFrame({
    "Feature": X.columns,
    "Correlation": correlation[target].loc[X.columns],
    "Mutual Information": mi_scores.loc[X.columns]
}).sort_values(by="Mutual Information", ascending=False)

print("\nFeature Importance")
print(importance)

# EDA visualizations

print("\nSummary Statistics")
print(df.describe())

# Histograms

df.hist(figsize=(14, 10))
plt.suptitle("Histogram of Features")
plt.show()

# Bar chart of target distribution

class_count = df[target].value_counts()

plt.figure(figsize=(6, 5))
plt.bar(class_count.index.astype(str), class_count.values, color=["skyblue", "orange"])
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.title("Target Distribution")
plt.show()

# Scatter plot: Glucose vs BMI

plt.figure(figsize=(7, 5))

diabetic = df[df[target] == 1]
non_diabetic = df[df[target] == 0]

plt.scatter(non_diabetic["Glucose"], non_diabetic["BMI"], color="blue", label="Non-Diabetic")
plt.scatter(diabetic["Glucose"], diabetic["BMI"], color="red", label="Diabetic")

plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("Glucose vs BMI")
plt.legend()
plt.show()

# Box plot

plt.figure(figsize=(12, 6))
numeric_columns = df.select_dtypes(include=np.number).columns
plt.boxplot([df[col] for col in numeric_columns], tick_labels=numeric_columns)
plt.xticks(rotation=45)
plt.title("Box Plot of Numerical Features")
plt.show()

# Heatmap

corr = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()

# Pair plot

plt.figure(figsize=(6, 5))
plt.scatter(df["Glucose"], df["BMI"])
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("Glucose vs BMI")
plt.show()

# Class balance

print("\nClass Distribution")

class_count = df[target].value_counts()

plt.figure(figsize=(6, 5))
plt.bar(class_count.index.astype(str), class_count.values, color=["green", "red"])
plt.xlabel("Diagnosis")
plt.ylabel("Number of Samples")
plt.title("Class Distribution")
plt.show()

# Train / validation / test split

from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print("\nTraining Shape")
print(X_train.shape)

print("\nValidation Shape")
print(X_validation.shape)

print("\nTesting Shape")
print(X_test.shape)
