import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('supply_chain_data.csv')
features=['Price','Availability','Stock levels','Lead times','Order quantities','Number of products sold','Revenue generated']

min_max_df=(df[features]-df[features].min())/(df[features].max()-df[features].min())
plt.figure(figsize=(10,6))
for f in features:
     min_max_df[f].plot(kind='kde',label=f)
plt.title('Min-Max Normalised Feature Distributions')
plt.xlabel('Values')        
plt.ylabel('Density')
plt.legend()
plt.show()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
min_max_df_sk=pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
plt.figure(figsize=(10,6))
for f in features:
     min_max_df_sk[f].plot(kind='kde',label=f)
plt.title('Min-Max Normalised Feature Distributions (Scikit-Learn)')
plt.xlabel('Values')    
plt.ylabel('Density')
plt.legend()
plt.show()