import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#load the dataset 

df = pd.read_csv("test.csv")

#handle missing values

#seprate as catgorical and numerical

cat_col = df.select_dtypes(include = ["object","str","category"]).columns
num_col = df.select_dtypes(include = ["number"]).columns

for col in cat_col:
    df[col] = df[col].fillna(df[col].mode().iloc[0])

for col in num_col:
    df[col] = df[col].fillna(df[col].median())

# encode
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_scaled = scaler.fit_transform(X)

#eda 
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)
print(df.shape[1])
print(df.shape[0])

#5. Split the dataset into training, validation and testing sets  
from sklearn.model_selection import train_test_split

X_train,X_temp,y_train,y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
) 
X_val,X_test,y_val,y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.30,
    random_state=42
) 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)



from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score

)
def cal_metric(y_acutal,y_predicated):
    mae = mean_absolute_error(y_acutal,y_predicated)
    mse = mean_squared_error(y_acutal,y_predicated)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_acutal,y_predicated)
    return mae,mse,rmse,r2
    
# baseline liner regression

from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(X_train,y_train)
lvalpre = linear.predict(X_val) 
ltestpre = linear.predict(X_test)

vmae,vmse,vrmse,vr2_score = cal_metric(y_val,lvalpre)
tmae,tmse,trmse,tr2_score = cal_metric(y_test,ltestpre)

print(vmae,vmse,vrmse,vr2_score)
print(tmae,tmse,trmse,tr2_score)

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge

ridge = Ridge()

paramter = {"alpha":[0.01,0.1,1,10,100]}

gridridge = GridSearchCV(ridge,paramter,cv=5)

gridridge.fit(X_train,y_train)
rvalpre = gridridge.predict(X_val) 
rtestpre = gridridge.predict(X_test)
print("bestparamter",gridridge.best_params_),
print("bestscore",gridridge.best_score_)

vmae,vmse,vrmse,vr2_score = cal_metric(y_val,rvalpre)
tmae,tmse,trmse,tr2_score = cal_metric(y_test,rtestpre)

print(vmae,vmse,vrmse,vr2_score)
print(tmae,tmse,trmse,tr2_score)


import seaborn as sns

plt.figure(figsize = (10,7))
sns.heatmap(df.corr(),annot=True,fmt=".2g")
plt.show()

y_pre = gridridge.predict(X_test)
resdulis = y_test - y_pre
plt.figure(figsize=(12,10))
plt.scatter(y_test,resdulis,alpha=0.6,color="Blue")
plt.show()

plt.figure(figsize=(12,10))
plt.bar(y_test,resdulis,color="blue")
plt.show()
from sklearn.metrics import confusion_matrix


thresold = np.median(y_test)

y_test_value = (y_test>thresold).astype(int)
y_pre_value = (y_pre>thresold).astype(int)

cm  = confusion_matrix(y_test_value,y_pre_value)
sns.heatmap(cm,annot=True,cmap="Blues",xticklabels=["below Threshold","above threshold"],yticklabels=["below threshold","above threshold"],fmt="d")
plt.show()




# from sklearn.linear_model import Lasso

# lasso = Lasso(max_iter=10000,tol=1e-3)

# paramter = {"alpha":[0.01,0.1,1,10,100]}

# gridlasso = GridSearchCV(lasso,paramter,cv=5,n_jobs=-1)

# gridlasso.fit(X_train,y_train)
# rvalpre = gridlasso.predict(X_val) 
# rtestpre = gridlasso.predict(X_test)
# print("bestparamter",gridlasso.best_params_),
# print("bestscore",gridlasso.best_score_)

# vmae,vmse,vrmse,vr2_score = cal_metric(y_val,rvalpre)
# tmae,tmse,trmse,tr2_score = cal_metric(y_test,rtestpre)

# print(vmae,vmse,vrmse,vr2_score)
# print(tmae,tmse,trmse,tr2_score)



# from sklearn.linear_model import ElasticNet

# elastic = ElasticNet(max_iter=10000,tol=1e-3)

# paramter = {"alpha":[0.01,0.1,1,10,100],"l1_ratio":[0.2,0.5,0.8]}

# gridelastic = GridSearchCV(elastic,paramter,cv=5,n_jobs=-1)

# gridelastic.fit(X_train,y_train)
# rvalpre = gridelastic.predict(X_val) 
# rtestpre = gridelastic.predict(X_test)
# print("bestparamter",gridelastic.best_params_),
# print("bestscore",gridelastic.best_score_)

# vmae,vmse,vrmse,vr2_score = cal_metric(y_val,rvalpre)
# tmae,tmse,trmse,tr2_score = cal_metric(y_test,rtestpre)

# print(vmae,vmse,vrmse,vr2_score)
# print(tmae,tmse,trmse,tr2_score)


# correlation  = df.corr(numeric_only=True)

# # from sklearn.feature_selection import mutual_info_regression
# # mi  =mutual_info_regression(X_test,y_test)
# # mi_scores = pd.Series(mi,index=df.columns)


