
# import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset

df = pd.read_csv("test.csv")
# print(df.head())

#perform data preprocessing

#handle missing values
print(df.isnull().sum())
cat_col = df.select_dtypes(include=["str","object","category"]).columns
num_col = df.select_dtypes(include=["number"]).columns
for col in num_col:
    df[col]= df[col].fillna(df[col].median())
for col in cat_col:
    df[col] = df[col].fillna(df[col].mode()[0])


#encode catgorical values:

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for col in df.columns:
    df[col] = encoder.fit_transform(df[col])





target = df.columns[-1]
feture_name = df.drop(columns=target).columns
X = df.drop(columns=target)
y=df[target]

#standlize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)


#corrlection matrix
correlation = df.corr(numeric_only=True)
print(correlation[target].sort_values(ascending=True))

#mutual info classify
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X_Scaled,y)
mi_scores=pd.Series(mi,index=feture_name)
print(mi_scores.sort_values(ascending=False))

#visiulize feture and target distributions

# df.hist(
#     bins=10,
#     edgecolor="black",
#     alpha=0.7,
#     figsize=(7,4)
# )

# plt.suptitle("Distribution")
# plt.tight_layout()
# plt.show()
print("Current last column:", df.columns[-1])
# plt.figure(figsize=(12,10))
# plt.hist(
#     df[df.columns[-1]], #pass the whole column
#     edgecolor="black"
# )
# plt.title("Target dist")
# plt.show()

#splitting the dataset in the test and train

from sklearn.model_selection import train_test_split

X_train,X_temp,y_train,y_temp=train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)
X_validation,X_test,y_validation,y_test=train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score



def cal_metrics(y_actual,y_predicted):
    mae = mean_absolute_error(y_actual,y_predicted)
    mse = mean_squared_error(y_actual,y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual,y_predicted)
    return mae,mse,rmse,r2



from sklearn.linear_model import LinearRegression
liner_model = LinearRegression()
liner_model.fit(X_train,y_train)
lvalpre=liner_model.predict(X_validation) 
lvaltest = liner_model.predict(X_test)
vmae,vmse,vrmse,vr2 = cal_metrics(y_validation,lvalpre)
tmae,tmse,trmse,tr2 = cal_metrics(y_test,lvaltest)
print(vmae,vmse,vrmse,vr2)
print(tmae,tmse,trmse,tr2)


from sklearn.model_selection import GridSearchCV,cross_validate

from sklearn.linear_model import Ridge

ridge_model = Ridge()

parameter = {"alpha":[0.01,0.1,1,10,100]}

grid_ridge = GridSearchCV(ridge_model,parameter,cv=5)
grid_ridge.fit(X_train,y_train)
print("best parameter",grid_ridge.best_params_)
print("best parameter",grid_ridge.best_score_)
lvalpre=grid_ridge.predict(X_validation) 
lvaltest = grid_ridge.predict(X_test)
vmae,vmse,vrmse,vr2 = cal_metrics(y_validation,lvalpre)
tmae,tmse,trmse,tr2 = cal_metrics(y_test,lvaltest)
print("Ridge:")
print(vmae,vmse,vrmse,vr2)
print(tmae,tmse,trmse,tr2)



# from sklearn.linear_model import Lasso

# lasso_model = Lasso(max_iter=10000,tol=1e-3)
# parameter = {"alpha":[0.01,0.1,1,10,100]}

# grid_lasso = GridSearchCV(lasso_model,parameter,cv=5)
# grid_lasso.fit(X_train,y_train)

# print("best paramter",grid_lasso.best_params_)
# print("best paramter",grid_lasso.best_score_)

# valpre = grid_lasso.predict(X_validation)
# testpre = grid_lasso.predict(X_test)

# vmae,vmse,vrmse,vr2 = cal_metrics(y_validation,valpre)
# tmae,tmse,trmse,tr2= cal_metrics(X_test,testpre)

# print(vmae,vmse,vrmse,vr2)
# print(tmae,tmse,trmse,tr2)



#elstic regression

from sklearn.linear_model import ElasticNet

elastic = ElasticNet(max_iter=10000,tol=1e-3)

elastic_param = {
    "alpha":[0.01,0.1,1,10],
    "l1_ratio":[0.2,0.5,0.8]
}


grid_elastic = GridSearchCV(elastic,parameter,cv=5)
grid_elastic.fit(X_train,y_train)

print("best paramter",grid_elastic.best_params_)
print("best paramter",grid_elastic.best_score_)

valpre = grid_elastic.predict(X_validation)
testpre = grid_elastic.predict(X_test)

vmae,vmse,vrmse,vr2 = cal_metrics(y_validation,valpre)
tmae,tmse,trmse,tr2= cal_metrics(X_test,testpre)

print(vmae,vmse,vrmse,vr2)
print(tmae,tmse,trmse,tr2)


