import pandas as pd
print(pd.__version__)

#to get label encoder from sckit
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler # used to standardize

column_names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]


df = pd.read_csv(
    "iris/iris.data",
    names=column_names
)

#to display first five record head() default =5
print(df.head())


#to display the no of rows
print(df.shape[0])

#to display the no of columns
print(df.shape[1])

#to display the feture name

print(df.dtypes)

#or using the df colums

print("Feature Name:",df.columns)

# to display the target label value
print("Target Label Value",df['species'].head(5))
#or 
print("Target Label\n\n",df.columns[-1])

#Data preprocessing


# Identify and handle missing values- Explain the method used to handle missing values

print("Dropping Null Value\n",df.dropna())


#Remove irrelevant features - Justify whether they should be removed.

print("Dropping Petal Length And Petal_Width\n\n")
# df = df.drop(columns=['petal_length','petal_width'])
# to verify the feture have been deleted
print(df.head())


#removing duplicates 
print(df.duplicated().sum()) #before
df = df.drop_duplicates()
print(df.duplicated().sum()) #after



#Encode categorical variables - Justify the technique used for encoding

encoder = LabelEncoder()

df["species"]= encoder.fit_transform(df["species"])


#Normalize or standardize numerical features.

# standarization

scaler = StandardScaler()

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

df[features] = scaler.fit_transform(df[features])
print(df.head())


# Compute the Correlation Coefficient and Mutual Information scores for all input features with respect to
# the target variable. Tabulate the results and identify the top ‘n’ important features


correlation = df.corr(numeric_only=True)
print("confusion Matrix")
print(correlation)

#pip install seaborn