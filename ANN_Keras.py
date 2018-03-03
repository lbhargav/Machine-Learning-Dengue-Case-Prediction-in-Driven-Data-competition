'''
CS6375.004: Machine Learning
Project: DengAI: Predicting Disease Spread

Team Members : Nandish Muniswamappa (nxm160630)
               Bhargav Lenka (bxl171030)
               Madhupriya Pal (mxp162030)
               Masoud Shahshahani (mxs161831)

File name: ANN_Keras.py
Input argument format : <Training Data set> <Test Data set>

'''

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import random
random.seed(0)

import sys
import numpy as np
import pandas
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error,explained_variance_score
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
# load dataset
dataframe = pandas.read_csv(sys.argv[1],  header=None)
dataset = dataframe.values
X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,-1]
X_train, X_valid, Y_train, Y_valid = cross_validation.train_test_split(X, Y, test_size=0.2)
nFeatures = X_train.shape[1]

# load Test DataSet
Testdataframe = pandas.read_csv(sys.argv[2],  header=None)
Testdataset = Testdataframe.values
X_test = Testdataset[:,0:Testdataset.shape[1]]


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
cvscores=[]

R2Validation_list = []
R2Training_list = []

for train, test in kfold.split(X, Y):
    model = Sequential()
    model.add(Dense(80, input_dim=nFeatures, kernel_initializer='normal', activation='relu'))
    # you can comment below line in order to reduce over fiting
    model.add(Dropout(0.2))
    model.add(Dense(80, kernel_initializer='normal', activation='relu'))
    # you can comment below line in order to reduce over fiting
    model.add(Dropout(0.2))
    # last layer without any activation function
    model.add(Dense(1, kernel_initializer='normal' ))
    # use adam optimizer without decay for learning rate
    adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(loss='mean_squared_error', optimizer=adam)

    # checkpoint
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    model.fit(X_train,Y_train, nb_epoch=100, batch_size=16, validation_data=(X_valid,Y_valid),verbose=0, shuffle=True,callbacks=[checkpoint])
    # load the best trained model for calculate accuracy
    model.load_weights("weights.best.hdf5")
    Y_trained_predicted=model.predict(X_train)
    Y_test_predicted=model.predict(X_valid)
    # Calculate Mean Square Error
    MSE_train=np.mean((Y_train-Y_trained_predicted[:,0])**2)
    MSE_validation=np.mean((Y_valid-Y_test_predicted[:,0])**2)
    # Calculate Mean Absolute Error
    MAE_training = mean_absolute_error(Y_train,Y_trained_predicted[:,0])
    MAE_validation = mean_absolute_error(Y_test_predicted, Y_valid)
    # calculate R2 score
    R2Validation = round(explained_variance_score(Y_train, Y_trained_predicted),2)*100
    R2Training = round(explained_variance_score(Y_valid, Y_test_predicted),2)*100

    R2Validation_list.append(R2Validation)
    R2Training_list.append(R2Training)
    print("")
    #print("For training:   MSE is  = %3.3f"%round(MSE_train,3)," , MAE is  = %3.3f"%round(MAE_training,3)," and the score is =",R2Validation,"%")
    print("MAE = %3.3f"%round(MAE_validation,3)," Accuracy =",R2Training,"%")

print("=============================================================")
print("Average Accuracy:", np.mean(R2Validation_list))
print("=============================================================")

model.summary()
