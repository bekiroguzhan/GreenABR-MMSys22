import pandas as pd 
import numpy as np 
import os 
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 


mos_data=pd.read_csv('data.csv')[['streaming_log','mos']]

data=[]
for l in os.listdir('./streaming_logs/'):
    d_=pd.read_csv('./streaming_logs/'+l)
    vmaf_change=0
    smooth_count=0
    rebuffer_count=0
    for i in range(len(d_)):    
        if i>0:
            if d_['rebuffering_duration'].iloc[i]>0:
                rebuffer_count+=1
            vmaf_change=vmaf_change+np.abs(d_['vmaf'].iloc[i]-d_['vmaf'].iloc[i-1])
            smooth_count=smooth_count+(np.abs(d_['vmaf'].iloc[i]-d_['vmaf'].iloc[i-1]))//20
#             if vmaf_change > 20:
#                 smooth_count+=1
    data.append([l,d_['vmaf'].sum(), d_['rebuffering_duration'].sum(),rebuffer_count, vmaf_change, smooth_count])

agg_data=pd.DataFrame(data, columns=['streaming_log','total_vmaf','total_rebuffer','total_rebuffer_count','total_smooth_change','total_smooth_count'])

merged_data=pd.merge(mos_data, agg_data, on='streaming_log')


X=merged_data.drop(columns=['mos','streaming_log'])
y=merged_data['mos']



scores=[]
rmses=[]
for i in range(1000):
    ridge = Ridge(normalize=True, alpha=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
    ridge.fit(X_train,y_train)

    print(ridge.score(X_test,y_test))

    y_pred=ridge.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmses.append(rmse)
    print("Root Mean Squared Error: {}".format(rmse))
    cr=pd.DataFrame(zip(y_test,y_pred)).corr(method='spearman')
    print('Spearman correlation score: ', cr[0][1])
    scores.append(cr[0][1])
    print("Coefficients: ",ridge.coef_)

print('Median and mean corrrelation scores: ',np.median(scores),np.mean(scores))


