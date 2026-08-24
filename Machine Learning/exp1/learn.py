import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df =  pd.read_csv("Diabetes_prediction(2).csv")

print(df.head()) #display the first five records

print(df.shape[0]) #row
print(df.shape[1]) #column
print(df.columns.tolist()) # to show as to list
print(df.columns[-1]) # to show target
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())


# idetify and handle missing values
print(df.isnull().sum())
# df = df.fillna(df.mean(numeric_only=True))
df = df.ffill().bfill()
df.fillna(df.mean(numeric_only=True),inplace=True)

#encoder
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# for col in df.columns:
#     if df[col].dtype =="object":
#         df[col]=encoder.fit_transform(df[col])


from sklearn.preprocessing import OneHotEncoder

catgorical = df.select_dtypes(include="object").columns

#intlize encoder object
encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(df[catgorical])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(catgorical),
    index=df.index
)

df = df.drop(columns=catgorical)
df = pd.concat([df,encoded_df],axis=1)
print(df.head())


#Standard Scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
target= df.columns[-1]

X=df.drop(columns=[target])
y=df[target]
X = scaler.fit_transform(X)


#Correlation Matrix:
correlation = df.corr(numeric_only=True)
print(correlation)
print(correlation["Diagnosis"].sort_values(ascending=False))

#Mutual info
from sklearn.feature_selection import mutual_info_classif
target = df.columns[-1]
X=df.drop(columns=[target])
y= df[target]
mi = mutual_info_classif(X,y)
mi_scores = pd.Series(mi,index=X.columns)
print(mi_scores.sort_values(ascending=False))

#Histograms
# df.hist(figsize=(14,10))
# plt.suptitle("Histogram of Fetures")
# plt.show()

# #plt histogram
# plt.hist(df["Age"],age=10)
# plt.hist(data)


plt.figure(figsize=(8,5))
plt.hist(
    df["Age"],
    bins=10,
    density=True,
    edgecolor="black",
    alpha=0.7, #trasnperancy
)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distrubtions")

plt.show()

#Bar chart

classcount = df["Diagnosis"].value_counts()
plt.bar(
    classcount.index.astype(str),
    classcount.values,
    color=["skyblue","darkblue"]
)
plt.xlabel("Diagnosis")
plt.ylabel("count")
plt.title("Target Distribution")


#scatter plot

diabetic = df[df["Diagnosis"]==1]
non_diabetic = df[df["Diagnosis"]==0]
plt.scatter(non_diabetic["Glucose"],
            non_diabetic["BMI"],
            color="blue",
            label="Non-Diabetic"
            )
plt.scatter(diabetic["Glucose"],
            diabetic["BMI"],
            color="blue",
            label="Non-Diabetic"
            )
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("Glucose vs BMI")
plt.show()

plt.boxplot(
    [df["Age"], df["BMI"], df["Glucose"]],
    label=["Age", "BMI", "Glucose"]
)
plt.show()

sns.boxplot(
    data=df,
    x="Diagnosis",
    y="BMI"
)


#heatmap
sns.heatmap(correlation
,annot=True,cmap="coolwarm")
plt.show()


#pair plot
# sns.pairplot(df,hue="Diagnosis")
# plt.show()

from sklearn.model_selection import train_test_split

X_train,X_temp,y_train,y_temp=train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


X_valdiation,X_test,y_valdiation,y_test=train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print(X_train.shape())