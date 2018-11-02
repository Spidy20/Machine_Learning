from sklearn import tree
from sklearn import datasets


DB=datasets.load_diabetes()
print(DB.get('Column'))
# x=DB.loc[:,'Target']
# y=DB.loc[:,'test']
# print(x)
impo