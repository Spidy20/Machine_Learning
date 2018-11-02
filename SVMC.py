# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# from sklearn.datasets import load_iris
#
# Data=load_iris()
# x=Data.data
# y=Data.target
# print(x.shape)
#
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# SVM=SVC()
# SVM.fit(x_train,y_train)
# print(SVM.score(x_test,y_test))

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import lightfm

iris = datasets.load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
sorted(clf.cv_results_.keys())
print('Best score for data:', clf.best_score_)
print('Best C:',clf.best_estimator_.C)
print('Best Kernel:',clf.best_estimator_.kernel)
print('Best Gamma:',clf.best_estimator_.gamma)