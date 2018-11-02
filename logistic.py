from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("car_evaluation.csv",header= None)
# print(data.head())
# print(data[0])
# for i in range(len(data[0])):
#     if data[0][i] == "vhigh":
#         data[0][i] = 0
#     elif data[0][i] == 'high':
#         data[0][i] = 1
#     elif data[0][i] == 'med':
#         data[0][i] = 2
#     elif data[0][i] == 'low':
#         data[0][i] = 3
for i in range(7):
    data[i] = LabelEncoder().fit_transform(data[i])
# print(data.head())

x = data.iloc[:,0:6]
y = data.iloc[:,6]
# print(data[:,0:6])
# x = np.array(x)
# y = np.array(y)
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=1, stratify = y)
print("Xtrain shape: ",xtrain.shape)
print("Ytrain shape: ",ytrain.shape)
print("Xtest shape: ",xtest.shape)
print("Ytest shape: ",ytest.shape)
print(np.bincount(ytrain))
print(np.bincount(ytest))
model = LogisticRegression(max_iter=1000)
model.fit(xtrain,ytrain)
print("Logistic Regression",model.score(xtest,ytest))

dt = DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
print("Decision Tree",dt.score(xtest,ytest))

mlp = MLPClassifier(max_iter=1000)
mlp.fit(xtrain,ytrain)
print("Neural Netowrk",mlp.score(xtest,ytest))


from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
d=load_iris()
x=d.data
y=d.target
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(x,y)
print("SVM",svc.score(xtest,ytest))