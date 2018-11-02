import  pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


csv=pd.read_csv('sales.csv')

encoder = LabelEncoder()
csv["Transaction_date"] = encoder.fit_transform(csv["Transaction_date"].fillna('0'))
csv["Product"] = encoder.fit_transform(csv["Product"].fillna('0'))
csv["Price"] = encoder.fit_transform(csv["Price"].fillna('0'))
csv["Payment"] = encoder.fit_transform(csv["Payment"].fillna('0'))


# x=csv.loc[:,'Transaction_date':'Payment']
# print(x.shape)
# y=csv.loc[:,'Product']


# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Model=DecisionTreeClassifier()
# print(Model)
# Model.fit(x_train,y_train)
# print(Model.score(x_test,y_test))
