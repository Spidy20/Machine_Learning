import quandl
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import tree

Data=pd.read_csv('infibeam.csv')
x=Data.loc[:,'High':'Turnover (Lacs)']
y=Data.loc[:,'Open']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# DT=tree.DecisionTreeRegressor(random_state=42)
MLP=MLPRegressor (random_state=42)
MLP.fit(x_train,y_train)
Test=[[2239.65,230.35,235.15,234.9,3357625.0,7898.64]]
Prediction=MLP.predict(Test)
print(Prediction)
print(MLP.score(x_test,y_test))

