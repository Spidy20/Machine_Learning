import  pandas as pd

url='http://samplecsvs.s3.amazonaws.com/SalesJan2009.csv'

data=pd.read_csv(url)
print(data)
