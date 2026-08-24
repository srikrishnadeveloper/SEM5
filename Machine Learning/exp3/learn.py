# Patient_ID,Age,Gender,BMI,Family_History,Smoking,Alcohol_Consumption,Physical_Activity,Hormone_Therapy,Menopause_Status,Genetic_Mutation,Tumor_Size_cm,Lymph_Node_Involvement,Mammogram_Result,Biopsy_Result,Blood_Pressure,Cholesterol,Diabetes,Exercise_Days_Per_Week,Breastfeeding_History,Annual_Income_USD


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("breast_cancer_prediction.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.shape[0])
print(df.shape[1])

#handle missing values

cat_col  = df.select_dtypes(include=['object','str','category']).columns
num_col  = df.select_dtypes(include=['number']).columns

for col in num_col:
    df[col]= df[col].fillna(df[col].median())

for col in cat_col:
    df[col]=df[col].fillna(df[col].mode().iloc[0])

#encode:

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for col in cat_col:
    df[col] = encoder.fit_transform(df[col])

#standlize

target = "Biopsy_Result"
X = df.drop(columns=[target])
y = df[target]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)

from sklearn.model_selection import train_test_split

X_train,X_temp,y_train,y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)
X_validation,X_test,y_validation,y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.30,
    random_state=42
)

# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score,classification_report

# kernels = ["linear","poly","rbf","sigmoid"]
# svmresult = {}

# for kernel in kernels:
#     print(f"usnig:{kernel}")
#     model = SVC(kernel=kernel,C=0.1,random_state=42)
#     model.fit(X_train,y_train)

#     valpre = model.predict(X_validation)
#     testpre = model.predict(X_test)

#     valacc = accuracy_score(y_validation,valpre)
#     testacc = accuracy_score(y_test,testpre)

#     svmresult[kernel] = {
#         "model":kernel,
#         "valacc":valacc,
#         "testacc":testacc
#     }
#     print(f"Valdiation Accuracy{valacc:.4f}")
#     print(f"Test Accuracy{testacc:.4f}")
#     print(classification_report(y_test,testpre))

# #compare kernel perfomance
# summary = pd.DataFrame([

# ])

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

model = SVC()

parameter = {
    "kernel":["linear","poly","rbf","sigmoid"],
    "C":[0.1,1,10,100],
    "gamma":["scale","auto",0.001,0.01,0.1],
    "degree":[2,3,4]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=parameter,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)

grid.fit(X_train,y_train)
valpre = grid.predict(X_validation)
testpre = grid.predict(X_test)

print("best model",grid.best_estimator_)
print("best parameter",grid.best_params_)
print("best score",grid.best_score_)

bestmodel = grid.best_estimator_
print("validation accuracy",bestmodel.score(X_validation,valpre))
print("test accuracy",bestmodel.score(X_test,testpre))

#Predict the class labels for the test dataset.

y_pred = grid.best_estimator_.predict(X_test)

print("prediecated class labelf for the test data")
print(y_pred)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print("accuracy score",accuracy_score(y_test,y_pred))
print("classifcation report",classification_report(y_test,y_pred))
print("confusion matrix",confusion_matrix(y_test,y_pred))
 

# Evaluate the classifier using accuracy, precision, recall, F1-score, confusion 
# matrix, and ROC-AUC score. 


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    classification_report,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
y_pred  = grid.best_estimator_.predict(X_test)

y_scores = grid.best_estimator_.decision_function(X_test)
print("accuracy score",accuracy_score(y_test,y_pred))
print("classifcation report",classification_report(y_test,y_pred))
print("confusion matrix",confusion_matrix(y_test,y_pred))
print("precision_score ",precision_score(y_test,y_pred))
print("recall_score ",recall_score(y_test,y_pred))
print("f1_score ",f1_score(y_test,y_pred))
print("roc_auc_score ",roc_auc_score(y_test,y_scores))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

fig,axes = plt.subplot(1,2,figsize=(12,5))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,cmpa="Blues",ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicated")
axes[0].set_ylabel("Acutal")

RocCurveDisplay.from_predictions(y_test,y_scores,ax=axes[1])
axes[1].set_title("Roc Curve")
axes[1].grid(True)

plt.tight_layout()
plt.show()


# • Compare the performance of different SVM kernel functions. 

cv_results = pd.DataFrame(grid.cv_results_)

kernel_comparsion= (
    cv_results.groupby("param_kernel")["mean_test_score"]
    .max()
    .reset_index()
    .rename(columns={"param_kernel":"Kernel","mean_test_score":"best cv accuracy"})
    .sort_values(by="best cv accuracy",ascending=False)
)

print("svm kernel comparsion")
print(kernel_comparsion.to_string(index=False))
