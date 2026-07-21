
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_validate

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("./train.csv")   

print(df.head())
print(df.info())
print(df.describe())


# 3a. Handle missing values
df = df.ffill().bfill()
print("Missing values after imputation:\n", df.isnull().sum())

# 3b. Encode categorical variables
encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == "object" or pd.api.types.is_string_dtype(df[column]):
        df[column] = encoder.fit_transform(df[column])

TARGET = "Loan Sanction Amount (USD)"

X = df.drop(TARGET, axis=1)
y = df[TARGET]
feature_names = X.columns

# 3c. Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# 4. Train / Validation / Test Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)
print("Training Samples:", len(X_train))
print("Validation Samples:", len(X_validation))
print("Testing Samples :", len(X_test))


# 5. Exploratory Data Analysis
print("summary staticis")
# print(df.describe())

# Visualize feature distributions and target distribution --------------

#using the histogram
plt.figure(figsize=(6, 4))
plt.hist(y, bins=20)
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.show()


# Feature distribution plots
plt.figure(figsize=(15, 10))                                                                                                                                         
grid_rows = int(np.ceil(len(feature_names) / 4))
for i, column in enumerate(feature_names, 1):
    plt.subplot(grid_rows, 4, i)   # 3 rows, 4 columns
    plt.hist(df[column], bins=20)
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Feature vs Target scatter plots

plt.figure(figsize=(15, 10))
for i, column in enumerate(feature_names, 1):
    plt.subplot(grid_rows, 4, i)
    plt.scatter(df[column], y)
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel("Loan Amount")

plt.tight_layout()
plt.show()

# Split the dataset into training, validation and testing sets ----------

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)


X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)


# Helper: compute MAE, MSE, RMSE, R2 in one call
#y_true contains the actual values.
# y_pred contains the predicted values from the model.
def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred) # actual - predicated = error take average of this errors
    mse = mean_squared_error(y_true, y_pred) # saqure the errors usefull in the big value
    rmse = np.sqrt(mse) # root of the mse
    r2 = r2_score(y_true, y_pred)  # tell the varation of the model preidction and the actual value 
    return mae, mse, rmse, r2



# Linear Regression tries to use all features.
# Ridge Regression also keeps all features but reduces their influence.
# Lasso Regression Hair Color and Favorite Food don't help predict the loan amount. I'll remove them.
# Elastic Regression Combines both advantages of the Ridge and Lasso:
                # Ridge → Keeps all features but reduces their importance.
                # Lasso → Can completely remove unimportant features.
                # Elastic Net → Does bot




from sklearn.linear_model import LinearRegression
# 6. Baseline Linear Regression
print("--------------- Linear Regression ---------------")
# equation =  y=mx+c
#used whenever the op is number
#dataset is regression

linear = LinearRegression()
start = time.time()
linear.fit(X_train, y_train)
linear_time = time.time() - start

# Predict using validation and test data
val_pred_linear = linear.predict(X_validation)
test_pred_linear = linear.predict(X_test)

# Calculate validation metrics
lin_val_mae, lin_val_mse, lin_val_rmse, lin_val_r2 = calc_metrics(y_validation, val_pred_linear)

# Calculate test metrics
lin_mae, lin_mse, lin_rmse, lin_r2 = calc_metrics(y_test, test_pred_linear)

print("Validation -> MAE:", lin_val_mae, "MSE:", lin_val_mse, "RMSE:", lin_val_rmse, "R2:", lin_val_r2)
print("Test       -> MAE:", lin_mae, "MSE:", lin_mse, "RMSE:", lin_rmse, "R2:", lin_r2)
print("Training Time :", linear_time)


# 7. Ridge Regression
#adds penalty if the model try to learn by memorizing   
# alpha is the penatly here and we don't know which alpha is best so we try some bunch of alpha value


ridge_model = Ridge()

parameters = {"alpha": [0.01, 0.1, 1, 10, 100]} #creating dict containig all the alpha values

grid_ridge = GridSearchCV(ridge_model, parameters, cv=5) # ridge model = model to train on, paramentes = the value to test,cv=5 the method to use
start = time.time()
grid_ridge.fit(X_train, y_train)
ridge_time = time.time() - start

print("Best Parameters:", grid_ridge.best_params_)
print("Best Cross Validation R2:", grid_ridge.best_score_)

val_pred_ridge = grid_ridge.predict(X_validation)
test_pred_ridge = grid_ridge.predict(X_test)

ridge_val_mae, ridge_val_mse, ridge_val_rmse, ridge_val_r2 = calc_metrics(y_validation, val_pred_ridge)
ridge_mae, ridge_mse, ridge_rmse, ridge_r2 = calc_metrics(y_test, test_pred_ridge)

print("Validation -> MAE:", ridge_val_mae, "MSE:", ridge_val_mse, "RMSE:", ridge_val_rmse, "R2:", ridge_val_r2)
print("Test       -> MAE:", ridge_mae, "MSE:", ridge_mse, "RMSE:", ridge_rmse, "R2:", ridge_r2)




# 8. Lasso Regression
print("--------------- Lasso Regression ---------------")
#use the regulirzed model and then just like the ridgemodel
#It can completely remove unimportant features from the model.
#good when the fetures are unesscary

lasso = Lasso(max_iter=5000) #no of iterations
lasso_params = {"alpha": [0.001, 0.01, 0.1, 1, 10]} #different parameter to work on

grid_lasso = GridSearchCV(lasso, lasso_params, cv=5)
start = time.time()
grid_lasso.fit(X_train, y_train)
lasso_time = time.time() - start


print("Best Parameters :", grid_lasso.best_params_)
print("Best CV R2      :", grid_lasso.best_score_)

val_pred_lasso = grid_lasso.predict(X_validation)
test_pred_lasso = grid_lasso.predict(X_test)

lasso_val_mae, lasso_val_mse, lasso_val_rmse, lasso_val_r2 = calc_metrics(y_validation, val_pred_lasso)
lasso_mae, lasso_mse, lasso_rmse, lasso_r2 = calc_metrics(y_test, test_pred_lasso)

print("Validation -> MAE:", lasso_val_mae, "MSE:", lasso_val_mse, "RMSE:", lasso_val_rmse, "R2:", lasso_val_r2)
print("Test       -> MAE:", lasso_mae, "MSE:", lasso_mse, "RMSE:", lasso_rmse, "R2:", lasso_r2)






# 9. Elastic Net Regression
#combines both the lasso and then ridge and then remove unesscary fetures and then also minmize there importance
print("--------------- Elastic Net Regression ---------------")

elastic = ElasticNet(max_iter=5000)
elastic_params = {
    "alpha": [0.01, 0.1, 1, 10],
    "l1_ratio": [0.2, 0.5, 0.8] # how much the ridge and the lasso are mixed why three value because we don't which mix is best that's why we give three value
}

grid_elastic = GridSearchCV(elastic, elastic_params, cv=5)
start = time.time()
grid_elastic.fit(X_train, y_train)
elastic_time = time.time() - start

print("Best Parameters :", grid_elastic.best_params_)
print("Best CV R2      :", grid_elastic.best_score_)

val_pred_elastic = grid_elastic.predict(X_validation)
test_pred_elastic = grid_elastic.predict(X_test)

elastic_val_mae, elastic_val_mse, elastic_val_rmse, elastic_val_r2 = calc_metrics(y_validation, val_pred_elastic)
elastic_mae, elastic_mse, elastic_rmse, elastic_r2 = calc_metrics(y_test, test_pred_elastic)

print("Validation -> MAE:", elastic_val_mae, "MSE:", elastic_val_mse, "RMSE:", elastic_val_rmse, "R2:", elastic_val_r2)
print("Test       -> MAE:", elastic_mae, "MSE:", elastic_mse, "RMSE:", elastic_rmse, "R2:", elastic_r2)




#hypermeter is the value we set before the training such as alpha the model does't know we give it  
#5 folde or 5cv is the process of the spiltting the dataset into the five set and using the four set for the training and then the last set for the 


# 10. Table 1 - Hyperparameter Tuning Summary
table1 = pd.DataFrame({
    "Model": ["Ridge Regression", "Lasso Regression", "Elastic Net Regression"],
    "Search Method": ["Grid Search", "Grid Search", "Grid Search"],
    "Best Parameters": [grid_ridge.best_params_, grid_lasso.best_params_, grid_elastic.best_params_],
    "Best Cross Validation R2": [grid_ridge.best_score_, grid_lasso.best_score_, grid_elastic.best_score_]
})
print("\nTable 1: Hyperparameter Tuning Summary")
print(table1)


# 11. Table 2 - Validation Performance Summary
table2 = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net"],
    "MAE": [lin_val_mae, ridge_val_mae, lasso_val_mae, elastic_val_mae],
    "MSE": [lin_val_mse, ridge_val_mse, lasso_val_mse, elastic_val_mse],
    "RMSE": [lin_val_rmse, ridge_val_rmse, lasso_val_rmse, elastic_val_rmse],
    "R2 Score": [lin_val_r2, ridge_val_r2, lasso_val_r2, elastic_val_r2]
})
print("\nTable 2: Validation Performance Summary")
print(table2)

# ------------------------------------------------------------

# ---------------------------------------------------
# 12. Table 3 - Test Set Performance
# ---------------------------------------------------
table3 = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net"],
    "MAE": [lin_mae, ridge_mae, lasso_mae, elastic_mae],
    "MSE": [lin_mse, ridge_mse, lasso_mse, elastic_mse],
    "RMSE": [lin_rmse, ridge_rmse, lasso_rmse, elastic_rmse],
    "R2 Score": [lin_r2, ridge_r2, lasso_r2, elastic_r2],
    "Training Time (s)": [linear_time, ridge_time, lasso_time, elastic_time]
})
print("\nTable 3: Test Set Performance")
print(table3)

# ------------------------------------------------------------

# ---------------------------------------------------
# 13. Table 4 - Coefficient Comparison (per feature)
# ---------------------------------------------------
table4 = pd.DataFrame({
    "Feature": feature_names,
    "Linear": linear.coef_,
    "Ridge": grid_ridge.best_estimator_.coef_,
    "Lasso": grid_lasso.best_estimator_.coef_,
    "Elastic Net": grid_elastic.best_estimator_.coef_
})
print("\nTable 4: Coefficient Comparison")
print(table4)


# 14. Visualizations - Predicted vs Actual & Residuals (all models)


#sns is built on over the matplotlib

model_preds = {    #stores model name and its predications
    "Linear Regression": test_pred_linear,
    "Ridge Regression": test_pred_ridge,
    "Lasso Regression": test_pred_lasso,
    "Elastic Net": test_pred_elastic
}

for name, preds in model_preds.items():
    # Predicted vs Actual
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted - {name}")
    plt.show()

    # Residual plot
    #residual = actual - predicated
    # here y-test is actual and the pred is predicted 
    residual = y_test - preds
    plt.figure(figsize=(6, 4))
    plt.scatter(preds, residual)
    plt.axhline(y=0)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot - {name}")
    plt.show()




# 15. Training vs Validation Error across Ridge alphas
#why we do this is because to find the overfitting and the underfitting

#created two empty list to store train and the validation error
train_error = []
validation_error = []
alphas = [0.01, 0.1, 1, 10, 100] #range of the alphas
 
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    train_prediction = model.predict(X_train)
    validation_prediction = model.predict(X_validation)

    #here the mean_squared_error is a function form the sckit learn
    train_error.append(mean_squared_error(y_train, train_prediction))
    validation_error.append(mean_squared_error(y_validation, validation_prediction))


plt.figure(figsize=(6, 4))
plt.plot(alphas, train_error, marker="o", label="Training Error")
plt.plot(alphas, validation_error, marker="o", label="Validation Error")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.title("Training vs Validation Error (Ridge)")
plt.legend()
plt.show()

# 

# 16. Coefficient Comparison Bar Plot (average magnitude)

# Suppose your model is

# Loan Amount =
# 5000
# + 120 × Income
# + 80 × Credit Score
# − 30 × Age
# Then the coefficients are 120,80,30

# Linear Regression → Uses the original coefficients.
# Ridge Regression → Shrinks coefficients slightly.
# Lasso Regression → Shrinks some coefficients to zero (feature selection).
# Elastic Net → Combination of Ridge and Lasso.

#this graph is used to know how each model is compressed coefficents

models_list = ["Linear", "Ridge", "Lasso", "ElasticNet"]
coefficients = [
    #np.abs = abosulate value ie negative values become postive
    #best_estimator_ returns the best model
    #.coef =  to get its coeffcients
    np.mean(np.abs(linear.coef_)),
    np.mean(np.abs(grid_ridge.best_estimator_.coef_)), 
    np.mean(np.abs(grid_lasso.best_estimator_.coef_)),
    np.mean(np.abs(grid_elastic.best_estimator_.coef_))
]

plt.figure(figsize=(6, 4))
plt.bar(models_list, coefficients)
plt.title("Coefficient Comparison (Average Magnitude)")
plt.ylabel("Average Coefficient Magnitude")
plt.show()


# 17. Table 5 - Training vs Validation Error (for Overfitting/Underfitting analysis)
train_pred_linear = linear.predict(X_train)
train_pred_ridge = grid_ridge.predict(X_train)
train_pred_lasso = grid_lasso.predict(X_train)
train_pred_elastic = grid_elastic.predict(X_train)

train_mse_linear = mean_squared_error(y_train, train_pred_linear)
train_mse_ridge = mean_squared_error(y_train, train_pred_ridge)
train_mse_lasso = mean_squared_error(y_train, train_pred_lasso)
train_mse_elastic = mean_squared_error(y_train, train_pred_elastic)

table5 = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net"],
    "Training MSE": [train_mse_linear, train_mse_ridge, train_mse_lasso, train_mse_elastic],
    "Validation MSE": [lin_val_mse, ridge_val_mse, lasso_val_mse, elastic_val_mse],
    "Gap (Val - Train)": [
        lin_val_mse - train_mse_linear,
        ridge_val_mse - train_mse_ridge,
        lasso_val_mse - train_mse_lasso,
        elastic_val_mse - train_mse_elastic
    ]
})
print("\nTable 5: Training vs Validation MSE (use this for the Overfitting/Underfitting write-up)")
print(table5)

print("\nAll tables generated: table1, table2, table3, table4, table5")





