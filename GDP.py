import  numpy as np
import  pandas as pd
from matplotlib import  pyplot as plt
from sklearn import  linear_model

file=r'GDP.csv'
df=pd.read_csv(file)


year=df['Year'].values[:,np.newaxis]
gdp=df['GDP'].values

LR=linear_model.LinearRegression()
LR.fit(year,gdp)
Future_gdp=[[2019],[2020],[2021],[2022]]
prediction=LR.predict(Future_gdp)
print(prediction)
plt.scatter(year,gdp)
plt.scatter(Future_gdp,prediction,color='green',linewidth='2')
plt.plot(Future_gdp,prediction,color='orange',linewidth='2')
plt.xlabel("Growth of GDP in years")
plt.ylabel("GDP in %")
plt.show()