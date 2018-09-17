import pandas as pd
from sklearn import tree

file=r'DT.csv'
data=pd.read_csv(file)


###Prepare data for training
x_train=data.loc[:,'Height':'Voice-pitch']###Give data to model from first attribute
y_train=data.loc[:,'Gender']##Give value to model from data

###Train the model

DT=tree.DecisionTreeClassifier()
DT.fit(x_train,y_train)
Test=[[138,26,1]]
prediction=DT.predict(Test)
print(prediction)