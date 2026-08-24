import pandas as pd
import numpy as no
import matplotlib.pyplot as plt
import time


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("spambase.data",header=None)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train,X_temp,y_train,y_temp=train_test_split(X,y,random_state=42,test_size=0.30)

X_validation,X_test,y_validation,y_test=train_test_split(X,y,random_state=42,test_size=0.30)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

from sklearn.preprocessing import MinMaxScaler

scaler_mm = MinMaxScaler()

X_train_mm = scaler_mm.fit_transform(X_train)
X_test_mm = scaler_mm.fit_transform(X_test)

from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score
)

gnb,bnb,mnb=GaussianNB(),BernoulliNB(),MultinomialNB()


start = time.time();
gnb.fit(X_train_scaled,y_train)
gnbtime = time.time() - start

start = time.time();
bnb.fit(X_train_scaled,y_train)
bnbtime = time.time() - start

start = time.time();
mnb.fit(X_train_mm,y_train)
mnbtime = time.time() - start

def get_metrics(model,X_test,train_time):
    y_pred = model.predict(X_test)
    tn,fp,fn,tp = confusion_matrix(y_test,y_pred).ravel()
    return[
        accuracy_score(y_test,y_pred),
        precision_score(y_test,y_pred),
        recall_score(y_test,y_pred),
        f1_score(y_test,y_pred),
        tn/(tn+fp),
        train_time
    ]

metrics =["accuracy_score","precision_score","recall_score","f1_score","Specificity","Train_time"]
table1 = pd.DataFrame({
"GaussianNB":get_metrics(gnb,X_test_scaled,gnbtime),
    "BernoulliNB":get_metrics(bnb,X_test_mm,bnbtime),
    "MultinomialNB":get_metrics(mnb,X_test_scaled,mnbtime)

},index=metrics)

print("table1")
print(table1)

from sklearn.neighbors import KNeighborsClassifier

knnbaseline = KNeighborsClassifier(n_neighbors=5)

start = time.time()
knnbaseline.fit(X_train_scaled,y_train)
knntime = time.time() - start

#predict
start = time.time()
knnbaseline.predict(X_test_scaled)
knnbaseline.predict_proba(X_test_scaled)[:,1]
knnpretime = time.time() -start

knn_baseline_metric = get_metrics(knnbaseline,X_test_scaled,knntime)

print("knnbaselinepreduction metrics")
print(f"")
print("accuracy_score:",knn_baseline_metric[0])
print("precision_score:",knn_baseline_metric[1])
print("recall_score:",knn_baseline_metric[2])
print("f1_score:",knn_baseline_metric[3])
print("Specificity:",knn_baseline_metric[4])
print("Train_time:",knn_baseline_metric[5])
print("knnpretime:",knnpretime)


# knn using the grid search and random search

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

parameter = {
    "n_neighbors":list(range(1,21)),
    "weights":["uniform","distance"],
    "p":[1,2]# 1 mahatten distance and 2 eucliden distance
}



gridknn = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=parameter,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)

start =time.time()
gridknn.fit(X_train_scaled,y_train)
gridknntime = time.time() -start

randomknn = RandomizedSearchCV(
    estimator=KNeighborsClassifier(),
    param_distributions=parameter,
    scoring="accuracy",
    n_iter=10,
    n_jobs=1
)
 
start =time.time()
randomknn.fit(X_train_scaled,y_train)
randomknntime = time.time() -start

table2 = pd.DataFrame({
    "SearchMethod":["GridSearch","RandomizedSearch"],
    "Best K":[gridknn.best_params_["n_neighbors"],randomknn.best_params_["n_neighbors"]],
    "Best Cv accuracy":[gridknn.best_score_,randomknn.best_score_],
    "Best Parameter":[str(gridknn.best_params_),str(randomknn.best_params_)]
})
print("knn clasfier in the gridsearch and randomsized search")
print(table2.to_string(index=False))


# knn using the kdtree and ball tree

bestk = gridknn.best_params_["n_neighbors"]
best_weight = gridknn.best_params_["weights"]
best_p = gridknn.best_params_["p"]

knnkdtree = KNeighborsClassifier(
    n_neighbors=bestk,
    weights=best_weight,
    p=best_p,
    algorithm='kd_tree'
)


knnkdtree.fit(X_train_scaled,y_train)

start= time.time()
knnkdtree.predict(X_test_scaled)
knnkdtreetime = time.time() - start

knnkdtreemetric = get_metrics(knnkdtree,X_test_scaled,knnkdtreetime)

knnballtree = KNeighborsClassifier(
    n_neighbors=bestk,
    weights=best_weight,
    p=best_p,
    algorithm="ball_tree"
)

knnballtree.fit(X_train_scaled,y_train)

start = time.time()
knnballtree.predict(X_test_scaled)
knnballtreetime = time.time()-start

knnballtreemetric = get_metrics(knnballtree,X_test_scaled,knnballtreetime)

table3 = pd.DataFrame({
    "metric":["accuracy_score","precision_score","recall_score","f1_score","Specificity","Train_time"],
    "result of kd tree":[knnkdtreemetric[0],knnkdtreemetric[1],knnkdtreemetric[2],knnkdtreemetric[3],knnkdtreemetric[4],knnkdtreemetric[5],],
    "result of ball tree":[knnballtreemetric[0],knnballtreemetric[1],knnballtreemetric[2],knnballtreemetric[3],knnballtreemetric[4],knnballtreemetric[5],]
})

print(table3)

