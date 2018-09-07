import numpy as np
import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt

file=r'car.csv'
df=pd.read_csv(file)

Car_years=df['Car'].values[:,np.newaxis]
Car_price=df['Price'].values

LR=linear_model.LinearRegression()
LR.fit(Car_years,Car_price)
Test_age=[[8],[12],[17],[19]]
prediction=LR.predict(Test_age)
print(prediction)

plt.scatter(Car_years,Car_price)
plt.scatter(Test_age,prediction,color='green',linewidth='2')
plt.plot(Test_age,prediction,color='blue',linewidth='2')
plt.xlabel('Age of car')
plt.ylabel('Price Depend on Car age')
plt.show()