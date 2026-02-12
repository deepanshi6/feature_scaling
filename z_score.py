#we will do in 3 ways - forrmula , scipy , sklearn
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
df=pd.read_csv('supply_chain_data.csv')
features=['Price','Availability','Stock levels','Lead times','Order quantities','Number of products sold','Revenue generated']
#here we used in one because problem solved by z score is same for all features so we can plot them together

z_score_df=(df[features]-df[features].mean())/df[features].std()
#every feature in features would have different mean and different standard deviation and pandas manage them separately for each feature

plt.figure(figsize=(10,6))
for f in features:
     z_score_df[f].plot(kind='kde',label=f)
plt.title('Z Score Normalised Feature Distributions')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.show()

#############
#using scipy
from scipy.stats import zscore

zscore_df = df[features].apply(zscore)

plt.figure(figsize=(10,6)) 
for f in features: 
     zscore_df[f].plot(kind='kde',label=f)
plt.title('Z Score Normalised Feature Distributions (Scipy)')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.show()

######################
#using Scikit-Learn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
zscore_df_sklearn = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
plt.figure(figsize=(10,6))
for f in features:
     zscore_df_sklearn[f].plot(kind='kde',label=f)     
plt.title('Z Score Normalised Feature Distributions (Scikit-Learn)')
plt.xlabel('Values') 
plt.ylabel('Density')
plt.legend()
plt.show()