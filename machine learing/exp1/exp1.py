
# Import Libraries

import pandas as pd
import numpy as np
from google.colab import files

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import mutual_info_classif

from sklearn.model_selection import train_test_split

# ==========================================
# Step 1 : Load Dataset
# ==========================================

uploaded = files.upload()

df = pd.read_csv("Diabetes_prediction.csv")

print("Dataset Loaded Successfully\n")

# ==========================================
# Step 2 : Display First Five Records
# ==========================================

print("First Five Records")
print(df.head(5))

# ==========================================
# Step 3 : Dataset Information
# ==========================================

print("\nDataset Shape")

print("Rows :", df.shape[0])
print("Columns :", df.shape[1])

print("\nFeature Names")

print(df.columns.tolist())

target = "Diagnosis"

print("\nTarget Column :", target)

# ==========================================
# Step 4 : Check Missing Values
# ==========================================

print("\nMissing Values")

print(df.isnull().sum())

# Fill Missing Values

for col in df.columns:

    if df[col].dtype == "object":

        df[col] = df[col].fillna(df[col].mode()[0])
    else:

        df[col] = df[col].fillna(df[col].median())

print("\nMissing Values After Handling")

print(df.isnull().sum())

# ==========================================
# Step 5 : Remove Irrelevant Features
# ==========================================

# Example:
# If ID column exists remove it

if "id" in df.columns:

    df.drop("id", axis=1, inplace=True)

    print("\nID column removed")

else:

    print("\nNo irrelevant feature found")

# ==========================================
# Step 6 : Encode Categorical Variables
# ==========================================

encoder = LabelEncoder()

categorical = df.select_dtypes(include="object").columns

for col in categorical:

    df[col] = encoder.fit_transform(df[col])

print("\nCategorical Features Encoded")

# ==========================================
# Step 7 : Normalize / Standardize
# ==========================================

scaler = StandardScaler()

features = df.drop(target, axis=1)

scaled_features = scaler.fit_transform(features)

X = pd.DataFrame(scaled_features, columns=features.columns)

y = df[target]

print("\nNumerical Features Standardized")

# ==========================================
# Step 8 : Correlation Coefficient
# ==========================================

correlation = df.corr()

corr_target = correlation[target].sort_values(ascending=False)

print("\nCorrelation with Target")

print(corr_target)

# ==========================================
# Step 9 : Mutual Information
# ==========================================

mi = mutual_info_classif(X, y)

mi_scores = pd.Series(mi, index=X.columns)

mi_scores = mi_scores.sort_values(ascending=False)

print("\nMutual Information Scores")

print(mi_scores)

# ==========================================
# Step 10 : Important Features
# ==========================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Correlation": corr_target.drop(target).values,

    "Mutual Information": mi_scores.values

})

print("\nFeature Importance")

print(importance)

# Top 5 Features

print("\nTop 5 Important Features")

print(mi_scores.head())

# ==========================================
# Step 11 : Summary Statistics
# ==========================================

print("\nSummary Statistics")

print(df.describe())

# ==========================================
# Step 12 : Histogram
# ==========================================

df.hist(figsize=(14,10))

plt.suptitle("Histogram of Features")

plt.show()

# ==========================================
# Step 13 : Bar Chart
# ==========================================

class_count = df[target].value_counts()

plt.figure(figsize=(6,5))

plt.bar(class_count.index.astype(str),
        class_count.values,
        color=["skyblue","orange"])

plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.title("Target Distribution")

plt.show()

# ==========================================
# Step 14 : Scatter Plot
# ==========================================

plt.figure(figsize=(7,5))

diabetic = df[df[target]==1]
non_diabetic = df[df[target]==0]

plt.scatter(non_diabetic["Glucose"],
            non_diabetic["BMI"],
            color="blue",
            label="Non-Diabetic")

plt.scatter(diabetic["Glucose"],
            diabetic["BMI"],
            color="red",
            label="Diabetic")

plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("Glucose vs BMI")

plt.legend()

plt.show()

# ==========================================
# Step 15 : Box Plot
# ==========================================

plt.figure(figsize=(12,6))

numeric_columns = df.select_dtypes(include=np.number).columns

plt.boxplot([df[col] for col in numeric_columns],
            labels=numeric_columns)

plt.xticks(rotation=45)

plt.title("Box Plot of Numerical Features")

plt.show()

# ==========================================
# Step 16 : Heatmap
# ==========================================

corr = df.corr(numeric_only=True)

plt.figure(figsize=(10,8))

plt.imshow(corr, cmap="coolwarm")

plt.colorbar()

plt.xticks(range(len(corr.columns)),
           corr.columns,
           rotation=90)

plt.yticks(range(len(corr.columns)),
           corr.columns)

plt.title("Correlation Heatmap")

plt.show()
# ==========================================
# Step 17 : Pair Plot
# ==========================================

plt.figure(figsize=(6,5))

plt.scatter(df["Glucose"],
            df["BMI"])

plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("Glucose vs BMI")

plt.show()

# ==========================================
# Step 18 : Class Balance
# ==========================================

print("\nClass Distribution")

count = df[target].value_counts()

plt.figure(figsize=(6,5))

plt.bar(count.index.astype(str),
        count.values,
        color=["green","red"])

plt.xlabel("Diagnosis")
plt.ylabel("Number of Samples")
plt.title("Class Distribution")

plt.show()

# ==========================================
# Step 19 : Data Splitting
# ==========================================

# Train = 70%
# Validation = 15%
# Test = 15%

X_train, X_temp, y_train, y_temp = train_test_split(

    X,

    y,

    test_size=0.30,

    random_state=42

)

X_validation, X_test, y_validation, y_test = train_test_split(

    X_temp,

    y_temp,

    test_size=0.50,

    random_state=42

)

print("\nTraining Shape")

print(X_train.shape)

print("\nValidation Shape")

print(X_validation.shape)

print("\nTesting Shape")

print(X_test.shape)