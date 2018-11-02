import pandas as pd
from sklearn.tree import tree,DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("mushrooms.csv",header= None)
for i in range(22):
    data[i] = LabelEncoder().fit_transform(data[i])

x = data.iloc[:,0:22]
y = data.iloc[:,0]


xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=42, stratify = y)


###logistic regression

model = LogisticRegression()
model.fit(xtrain,ytrain)
# test=[['b','s','y','t','a','f','c','b','g','e','c','s','s','w','w','p','w','o','p','k','s','m']]
print('Logistic Regression prediction: ',model.predict(xtest))
print("Logistic Regression score: ",model.score(xtest,ytest))

print("")

##decision tree classifier
DT=DecisionTreeClassifier()
DT.fit(xtrain,ytrain)
print('prediction tree prediction: ',model.predict(xtest))
print("Decision tree score: ",model.score(xtest,ytest))

print("")

##neural network
from sklearn.neural_network import MLPClassifier
NN=MLPClassifier()
NN.fit(xtrain,ytrain)
print("Nueral network prediction: ",NN.predict(xtest))
print("Nueral network score: ",NN.score(xtest,ytest))