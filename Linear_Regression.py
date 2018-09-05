import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

file=r'example.csv'
df=pd.read_csv(file)


x_train=df['Father'].values[:,np.newaxis]
y_train=df['Son'].values

LR=LinearRegression()
LR.fit(x_train,y_train)

x_test=[[72.8],[61.1],[67.4],[70.2],[75.6],[60.2],[65.3],[59.2]]
prediction=LR.predict(x_test)
print(prediction)

plt.scatter(x_train,y_train,color='b')
plt.scatter(x_test,prediction,color='black',linewidths=3)
plt.xlabel("Father Height in inches")
plt.ylabel("Son height in inches")
plt.show()

