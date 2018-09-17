import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  linear_model
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

##Import data
File=r'GDP.csv'
data=pd.read_csv(File)

year=data['Year'].values[:,np.newaxis]
gdp=data['GDP'].values

LR=linear_model.LogisticRegression()
LR.fit(year,gdp)
LR.score(year,gdp)
Future_gdp=[[2019],[2020],[2021],[2022]]
prediction=LR.predict(Future_gdp)
print(prediction)
