from sklearn import  linear_model
import  matplotlib.pyplot as plt

workdays=[[365],[300],[250],[200]]
salary=[150000,130000,110000,100000]

plt.scatter(workdays,salary,color='blue')
plt.xlabel('workdays of employe')
plt.ylabel('salary of workdays')


LR=linear_model.LinearRegression()
LR.fit(workdays,salary)
prediction=LR.predict([[315],[100],[150]])
plt.plot([[315],[100],[150]],prediction,color='green',linewidth=3)
print(prediction)
plt.show()