import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("credit_risk_dataset.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())


# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])


# Encode categorical columns
le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])


# EDA - Target distribution
plt.hist(df["loan_amnt"], bins=30)
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.title("Loan Amount Distribution")
plt.show()


# Feature vs Target
plt.scatter(df["person_income"], df["loan_amnt"], alpha=0.3)
plt.xlabel("Income")
plt.ylabel("Loan Amount")
plt.title("Income vs Loan Amount")
plt.show()


# Correlation
print("\nCorrelation:")
print(df.corr())


# X and y
X = df.drop(columns=["loan_amnt"])
y = df["loan_amnt"]


# Train - Validation - Test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)


# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# Model
model = LinearRegression()


# Training time
start = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start

print("\nTraining Time:", training_time, "seconds")


# Predictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)


# Evaluation function
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("\n", name)
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)


# Validation performance
evaluate(y_val, y_val_pred, "Validation Performance")


# Test performance
evaluate(y_test, y_test_pred, "Test Performance")


# Actual vs Predicted
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.xlabel("Actual Loan Amount")
plt.ylabel("Predicted Loan Amount")
plt.title("Actual vs Predicted")
plt.show()


# Residual Plot
residuals = y_test - y_test_pred

plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(0)
plt.xlabel("Predicted Loan Amount")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()


# Training Error vs Validation Error
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)

plt.bar(["Training", "Validation"], [train_mse, val_mse])
plt.ylabel("MSE")
plt.title("Training Error vs Validation Error")
plt.show()


# Coefficient Comparison
plt.bar(X.columns, model.coef_)
plt.xticks(rotation=90)  
plt.xlabel("Features")
plt.ylabel("Coefficient")
plt.title("Linear Regression Coefficients")
plt.tight_layout()
plt.show()