import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('supply_chain_data.csv')
features=['Price','Availability','Stock levels','Lead times','Order quantities','Number of products sold','Revenue generated']

Q1=df[features].quantile(0.25)
Q3=df[features].quantile(0.75)
IQR=Q3-Q1
r_s_df=(df[features]-Q1)/IQR

plt.figure(figsize=(10,6))
for f in features:
     r_s_df[f].plot(kind='kde',label=f) 
plt.title('Robust Scaled Feature Distributions')
plt.xlabel('Values')    
plt.ylabel('Density')
plt.legend()
plt.show()

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
r_s_df_sk=pd.DataFrame(scaler.fit_transform(df[features]),columns=features)
plt.figure(figsize=(10,6))
for f in features:
     r_s_df_sk[f].plot(kind='kde',label=f)
plt.title('Robust Scaled Feature Distributions (Scikit-Learn)')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.show()