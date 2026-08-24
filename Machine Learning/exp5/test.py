import time 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# load the dataset
df = pd.read_csv("spambase.data",header=None)

# best apporach is to split before doing anyother things

#target value separation
target = df.columns[-1]

X = df.iloc[:,:-1]
y = df.iloc[:,-1]


from sklearn.model_selection import train_test_split

X_train,X_temp,y_train,y_temp =  train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.30
)

X_validation,X_test,y_validation,y_test =  train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.30
)

# handle missing values

# this exp has only numerical values
X = X.fillna(df.median())

# Scaling

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
scaler_mm = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_mm = scaler_mm.fit_transform(X_train)
X_test_scaled_mm = scaler_mm.transform(X_test)

# eda

print(df.head())
print(df.info())
print(df.describe())
print(df.shape)
print(df.shape[0])
print(df.shape[1])


# feature distributions
# df.hist(figsize=(12,10),edgecolor="black",alpha=0.6)
# plt.tight_layout()
# plt.suptitle("distriubtions")
# plt.show()

# #target distributions
# plt.figure(figsize=(12,10))
# plt.hist(df,edgecolor="black",alpha=0.6)

# # other plots

# sns.boxplot(data=X)
# plt.show()

# #sns hist plot

# sns.histplot(data = X)
# plt.show()

# sns.heatmap(X.corr(),annot=True)

# sns.pairplot(X,hue=None)


#import statement for the all metric
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    r2_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    f1_score

)

#helper metric calucator
def get_metric(model,X_test,traintime):
    y_pred = model.predict(X_test)
    tp,fn,fp,tn = confusion_matrix(y_test,y_pred).ravel()
    return(
        accuracy_score(y_test,y_pred),
        precision_score(y_test,y_pred),
        recall_score(y_test,y_pred),
        f1_score(y_test,y_pred),
        tn/(tn+fp), 
        traintime
    )



# 6. Train Na¨ıve Bayes variants 


from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB

gnb,bnb,mnb =GaussianNB(),BernoulliNB(),MultinomialNB()

gnb.fit(X_train_scaled,y_train)
bnb.fit(X_train_scaled,y_train)
mnb.fit(X_train_scaled_mm,y_train)

#for gnb
start = time.time()
gnb.predict(X_test_scaled)
gnbtime = time.time()-start

#for bnb
start = time.time()
bnb.predict(X_test_scaled)
bnbtime = time.time()-start

#for mnb
start = time.time()
mnb.predict(X_test_scaled_mm)
mnbtime = time.time()-start

Metrics = ["accuracy_score","precision_score","recall_score","f1_score","Specificity","traintime"]
table1 = pd.DataFrame({
    "GaussianNB":get_metric(gnb,X_test_scaled,gnbtime),
    "BernoulliNB":get_metric(bnb,X_test_scaled,bnbtime),
    "MultinomialNB":get_metric(mnb,X_test_scaled_mm,mnbtime)
},index=Metrics)
print(table1)


# kernel

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled,y_train)

start = time.time()
knn.predict(X_test_scaled)
knn.predict_proba(X_test_scaled)# ================
knntime = time.time()-start


print("knn metrics")
table2 = pd.DataFrame({
    "Knn model":get_metric(knn,X_test_scaled,knntime)
},index=Metrics)


print(table2)

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


parameters = {
    "n_neighbors":list(range(1,21)),
    "weights":["uniform","distance"],
    "p":[1,2]
}



gridknn = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid = parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)

start  = time.time()
gridknn.fit(X_train_scaled,y_train)
gridknntime = time.time()-start

randomknn = RandomizedSearchCV(
    estimator=KNeighborsClassifier(),
    param_distributions=parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)

start=time.time()
randomknn.fit(X_train_scaled,y_train)
randomknntime = time.time() -start

print("knn metrics with the grid and random")
table3 = pd.DataFrame({
    "Knn model with grid search":get_metric(gridknn,X_test_scaled,gridknntime),
    "Knn model with random search":get_metric(randomknn,X_test_scaled,randomknntime)
},index=Metrics)

print(table3)


# knn using the balltree and kd tree

bestk=gridknn.best_params_["n_neighbors"]
bestw=gridknn.best_params_["weights"]
bestp=gridknn.best_params_["p"]

kdtree = KNeighborsClassifier(
    n_neighbors=bestk,
    weights=bestw,
    p=bestp,
    algorithm="kd_tree"
)
balltree = KNeighborsClassifier(
    n_neighbors=bestk,
    weights=bestw,
    p=bestp,
    algorithm="ball_tree"
)

kdtree.fit(X_train_scaled,y_train)
balltree.fit(X_train_scaled,y_train)

start = time.time()
kdtree.predict(X_test_scaled)
kdtreetime = time.time() -start


start = time.time()
balltree.predict(X_test_scaled)
balltreetime = time.time() -start


print("knn metrics with the kdtree and balltree")
table4 = pd.DataFrame({
    "Knn model with kdtree":get_metric(kdtree,X_test_scaled,gridknntime),
    "Knn model with balltree":get_metric(balltree,X_test_scaled,randomknntime)
},index=Metrics)
print(table4)


from sklearn.cluster import KMeans

k = 3
kmeans = KMeans(n_clusters=4,n_init=10,random_state=42)
cluster_label = kmeans.fit_predict(X_test_scaled)

centroids = kmeans.cluster_centers_
wcss = kmeans.inertia_

print("centorids",np.unique(centroids))
print("wcss",wcss)
