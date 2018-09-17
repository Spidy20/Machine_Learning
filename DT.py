import pandas as pd
from sklearn import tree

# file=r'DT.csv'
# DF=pd.read_csv(file)

# Ht=DF['Height'].values
# HL=DF['Hair-length'].values
# VP=DF['Voice-pitch'].values
# G=DF['Gender'].values
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