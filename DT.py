import pandas as pd
from sklearn import tree
X = [ [180, 15,0],
      [167, 42,1],
      [136, 35,1],
      [174, 15,0],
      [141, 28,1]]

Y = ['man', 'woman', 'woman', 'man', 'woman']

DT=tree.DecisionTreeClassifier()
DT.fit(X,Y)
Test=[[183,26,1]]
prediction=DT.predict(Test)
print(prediction)


