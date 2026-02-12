import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('supply_chain_data.csv')
#print(df.columns.tolist())


features=['Price','Availability','Stock levels','Lead times','Order quantities']
features2=['Number of products sold','Revenue generated']
#scale of features 1 and features 2 are so different, so we will plot them separately for better visualisation in the original
#you will realise once you see charts

plt.figure(figsize=(10,6))   #figsize is for size of chart in inches (width, height)

for f in features:
    df[f].plot(kind='kde',label=f)
#in this pandas internally call matplotlib which internally calls scipy
#kde is kernel density estimation which is a way to estimate the probability density function of a random variable. It is a non-parametric way to estimate the distribution of data. It is used to visualize the distribution of data and to identify patterns in the data.

plt.title('Original Feature Distributions 1')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.show()

plt.figure(figsize=(10,6)) 
for f in features2: 
    df[f].plot(kind='kde',label=f) 
plt.title('Original Feature Distributions 2')
plt.xlabel('Values') 
plt.ylabel('Density') 
plt.legend() 
plt.show()