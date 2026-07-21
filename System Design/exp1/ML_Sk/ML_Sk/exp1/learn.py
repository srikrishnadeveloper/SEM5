
# ASSIGNMENT 

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


#Load the dataset into Python using the Pandas library.

df = pd.read_csv("Diabetes_prediction(2).csv")
print("Dataset loaded succesfully\n")

#Display the first five records of the dataset.

print("displaying first five record")
print(df.head())

#Determine: Number of rows, Number of columns, Feature names, Target (label) column


print("Number of rows\n")
print(df.shape[0])

print("Number of colums\n")
print(df.shape[1])

print("Feature Name\n")
print(df.columns.tolist())  #to list to display it has list

print("Target(label) Column\n")
print(df.columns[-1])


#Data Preprocessing:

#Identify and handle missing values- Explain the method used to handle missing values.

print("Missing values")
print(df.isnull().sum()) #before

df.fillna(df.mean(numeric_only=True),inplace=True)
#or
# df = df.fillna(df.mean(numeric_only=True))

print(df.isnull().sum()) #after



#Remove irrelevant features - Justify whether they should be removed.

if "id" in df.columns:
    df.drop("id",axis=1,inplace=True)
    print(" ID column removed\n")
else:
    print("No irrelevent feture found\n")


print("Duplicates Before Removal :", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates After Removal :", df.duplicated().sum())


# Encode categorical variables - Justify the technique used for encoding

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder() #object


categorical = df.select_dtypes(include="object").columns

for col in categorical:

    df[col] = encoder.fit_transform(df[col])

print("\nCategorical Features Encoded")


from sklearn.preprocessing import StandardScaler

# Create scaler object
scaler = StandardScaler()

# Only scale input features
numerical_features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age" 
    #the target feture should not be mentioned here
]

df[numerical_features] = scaler.fit_transform(df[numerical_features])

print(df.head())

# Correlation Matrix
correlation = df.corr(numeric_only=True)

print(correlation)

print("\nCorrelation with Target Variable:")
print(correlation["Diagnosis"].sort_values(ascending=False))


#Mutual Information
from sklearn.feature_selection import mutual_info_classif
X = df.drop("Diagnosis",axis=1)
y = df["Diagnosis"]

mi = mutual_info_classif(X,y)
mi_scores = pd.Series(mi,index=X.columns)
print("Mutual Infromation Scores")
print(mi_scores.sort_values(ascending=False))



#Important Features
importance = pd.DataFrame({
    "Feature": X.columns,
    "Correlation": correlation["Diagnosis"].drop("Diagnosis").values,
    "Mutual Information": mi_scores.values
})

print("Important Features")
print(importance)

print(mi_scores.head(10))


#Exploratory Data Analysis (EDA) and Visualization: 

#Summary statistics
print("Summary statistics")
print(df.describe())


#Histograms,
df.hist(figsize=(14,10))
plt.suptitle("Histogram of Fetures")
plt.show()

#Bar Chart
# class_count = df["Diagnosis"].value_counts()

# plt.figure(figsize=(7,5))

# plt.bar(
#     class_count.index.astype(str),
#     class_count.values,
#     color=["skyblue", "darkblue"]
# )

# plt.xlabel("Diagnosis")
# plt.ylabel("Count")
# plt.title("Target Distribution")

# plt.show()

class_count = df["Diagnosis"].value_counts()

plt.figure(figsize=(7,5))

plt.bar = (
    class_count.index.astype(str),
    Class_count.values,
    color=["skyblue","orange"]
)
plt.xlabel("Diagnosis")
plt.ylabel("count")
plt.title("Target Distribution")

plt.show()

#Scatter Plot

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


# Box Plot

plt.figure(figsize=(12,6))

numeric_columns = df.select_dtypes(include=np.number).columns

plt.boxplot([df[col] for col in numeric_columns],
            labels=numeric_columns)

plt.xticks(rotation=45)

plt.title("Box Plot of Numerical Features")

plt.show()

#Heatmap

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

#Pair Plot


plt.figure(figsize=(6,5))

plt.scatter(df["Glucose"],
            df["BMI"])

plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("Glucose vs BMI")

plt.show()

# Class Balance

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

# Data Splitting



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