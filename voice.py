import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df=pd.read_csv('voice.csv')
# print(df.shape)
#
# print("Number of male: {}".format(df[df.label == 'male'].shape[0]))
# print('Number of Female: {}'.format(df[df.label=='female'].shape[0]))

x=df.iloc[:,:-1]
y=df.iloc[:,-1]


gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
scaler=StandardScaler()
scaler.fit(x)
X = scaler.transform(x)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
scc=SVC()
scc.fit(x_train,y_train)
y_predict=scc.predict(x_test)
print("SVM accuracy is:",scc.score(x_test,y_test))

###Nueral network
from sklearn.neural_network import MLPClassifier
NN=MLPClassifier()
NN.fit(x_train,y_train)
print("Nueral network score: ",NN.score(x_test,y_test))

###Decision tree
from sklearn.tree import tree
DT=tree.DecisionTreeClassifier()
DT.fit(x_train,y_train)
print("Decision tree score: ",DT.score(x_test,y_test))

###Logistic regression
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(x_train,y_train)
print("Logistic regression score: ",LR.score(x_test,y_test))