import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
df=pd.read_csv('supply_chain_data.csv')
features=['Price','Availability','Stock levels','Lead times','Order quantities','Number of products sold','Revenue generated']

original_df=df[features]

scaler_mm=MinMaxScaler()
mm_df=pd.DataFrame(scaler_mm.fit_transform(df[features]),columns=features)

z_Scaler=StandardScaler()
z_score_df=pd.DataFrame(z_Scaler.fit_transform(df[features]),columns=features)

scaler_rs=RobustScaler()
r_s_df=pd.DataFrame(scaler_rs.fit_transform(df[features]),columns=features)

fig,axes=plt.subplots(2,2,figsize=(14,10))
#divides chart into 4 parts , 2 rows and 2 columns
axes[0,0].set_title('Original Feature Distributions')
for f in features:
     original_df[f].plot(kind='kde',label=f,ax=axes[0,0])
axes[0,0].legend()
axes[0,1].set_title('Min-Max Normalised Feature Distributions')
for f in features:     
     mm_df[f].plot(kind='kde',label=f,ax=axes[0,1])
axes[0,1].legend()
axes[1,0].set_title('Z Score Normalised Feature Distributions')
for f in features:
     z_score_df[f].plot(kind='kde',label=f,ax=axes[1,0])
axes[1,0].legend()
axes[1,1].set_title('Robust Scaled Feature Distributions')  
for f in features:
     r_s_df[f].plot(kind='kde',label=f,ax=axes[1,1])
axes[1,1].legend()
plt.tight_layout()   #it automatically adjusts the spacing between subplots
plt.show()