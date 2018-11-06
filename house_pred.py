import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

###Let see which regression algorithm can predict perfect price of House

data=pd.read_csv('house.csv')
x=data.loc[:,'bedrooms':'sqft_lot15']
y=data.loc[:,'price']

###Split data into train & test model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
Test=[[5,2,1810,4850,1.5,0,0,3,7,1810,0	,1900,0,98107,47.67,-122.394,1360,4850]]

####Linear regression
LR=LinearRegression()
LR.fit(x_train,y_train)
print("Linear regression prediction for House price =",LR.predict(Test))
print("Linear regression score is:",LR.score(x_test,y_test))
print(' ')


####neural network
from sklearn.neural_network import MLPRegressor
NN=MLPRegressor()
NN.fit(x_train,y_train)
print("Neural network prediction for House price =",NN.predict(Test))
print("Neural network score is:",NN.score(x_test,y_test))

print(' ')
###decision tree regression
from sklearn.tree import tree
DT=tree.DecisionTreeRegressor()
DT.fit(x_train,y_train)
print("Decision tree prediction for House price =",DT.predict(Test))
print("Decision tree regression score is:",DT.score(x_test,y_test))

print(' ')
print('and Decision tree won this Game , its shows accurate price for particular house details')