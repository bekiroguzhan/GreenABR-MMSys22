
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from joblib import dump, load
import numpy as np




# Scales the features and targets based on the corresponding max value
def NormalizeDataSet(filename):
    originalData=pd.read_csv(filename)
    originalData=originalData[originalData['Power']<1700]
    n_dataset=originalData[['Bitrate','FileSize','Quality','Motion']]
    n_dataset['PixelRate']=originalData['Height']*originalData['Width']
    n_dataset['Power']=originalData['Power']
    data_val = n_dataset.values #returns a numpy array
    max_scaler = preprocessing.MaxAbsScaler()
    data_val_scaled = max_scaler.fit_transform(data_val)
    n_dataset = pd.DataFrame(data_val_scaled)
    return n_dataset
    
    

# base NN to use in the regression model 
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(7, input_dim=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# evaluate the model for an external device which was not used for training
# measurements are done for this device for the same videos 
def evaluateForOtherDevice(dataset_path):
    dataset2=NormalizeDataSet(dataset_path).values
    x_other=dataset2[:,0:5]
    y_other=dataset2[:,5]
    return pipeline.score(x_other,y_other)


n_dataset= NormalizeDataSet('./dataset/dataset_all_videos_galaxy.csv')  # streaming dataset collected with real streaming sessions with power meter
dataset = n_dataset.values
# split into input (X) and output (Y) variables
X = dataset[:,0:5]
Y = dataset[:,5]


random_state = 42
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=random_state)



estimators = []
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=10, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
print('********Model training started')
pipeline.fit(x_train,y_train)
results = cross_val_score(pipeline, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
print("Average cross validation scores over 10 fold Mean: %.4f  Std: %.4f" % (results.mean(), results.std()))



print("Model evaluation results over test split:  rmse = ", pipeline.score(x_test,y_test))


print('******** Saving model')
dump(pipeline, 'power_model.joblib')

print('Model evaluation results over unseen device(Not used for training) :  rmse = ', evaluateForOtherDevice('./dataset/dataset_all_videos_xcover.csv'))







