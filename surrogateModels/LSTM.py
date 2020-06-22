import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


def timeTMap(z0, t0, iu, modelData):
    
    # prepare input data
    temp = z0.reshape((1, modelData.nDelay + 1, modelData.dimZ))
    tt = temp[:,::-1,:]
    tt = (tt-modelData.data_mean[iu])/modelData.data_std[iu];
    
    # model predicition
    y_pred = modelData.LSTM_model[iu].predict(tt,batch_size=1) 
    
    # transform output
    y_pred = y_pred * modelData.data_std[iu] + modelData.data_mean[iu] 
    z = np.concatenate((y_pred[0,:], z0[:-modelData.dimZ]))

    # add time step
    t = t0 + modelData.h
    
    return z, t, modelData


def createSurrogateModel(modelData, data):
    
    X = data['X']
    Y = data['Y']
     
    train_per = 0.9
    
    model = dict()
    mean = np.zeros([len(X),])
    std = np.zeros([len(X),])
    
    for i in range(len(X)):
        
        # Create LSTM-model for each control input
        
        dataX = X[i]
        dataY = Y[i]
        
        data_size = dataX.shape[0]
        
        train_size = int(np.ceil(train_per*data_size))
        val_size = data_size-train_size
    
        # Prepare data sets
        Xtrain,Ytrain,Xval,Yval,dimZ, mean[i], std[i] = LSTM_datasets(modelData, dataX, dataY, train_size, val_size)
    
        # Create and train model
        model[i] = create_and_train_LSTM_model(modelData,Xtrain,Ytrain,Xval,Yval,dimZ)
    
    
    # Save LSTM models and important parameters
    setattr(modelData, 'LSTM_model', model)
    setattr(modelData, 'dimZ', dimZ)
    setattr(modelData, 'data_mean', mean)
    setattr(modelData, 'data_std', std)

    return modelData


def LSTM_datasets(modelData, dataX, dataY, train_size,val_size):
    
    dimZ = int(dataX.shape[1] / (modelData.nDelay + 1))
    dataY = dataY[:,:dimZ]
    
    mean = np.mean(dataX[:,0:dimZ])
    std = np.std(dataX[:,0:dimZ])
    
    X = np.zeros(dataX.shape)
    
    # normalize data with train_data mean:
    for i in range(modelData.nDelay + 1):
        X[:, i * dimZ : (i+1) * dimZ] = (dataX[:, (modelData.nDelay -i) * dimZ : (modelData.nDelay+1 -i) * dimZ]-mean)/std;
    
    Y = (dataY - mean) / std
 
    # Devide data into training and validation set

    Xtrain = X[:train_size,:]
    Ytrain = Y[:train_size,:]

    Xval = X[train_size:,:]
    Yval = Y[train_size:,:]

    # reshape inputs to be 3D [samples, timesteps, features] for LSTM
    Xtrain = Xtrain.reshape((train_size, modelData.nDelay + 1, dimZ))
    Xval = Xval.reshape((val_size, modelData.nDelay + 1,dimZ))

    return Xtrain, Ytrain, Xval, Yval, dimZ, mean,std


def create_and_train_LSTM_model(modelData, Xtrain,Ytrain,Xval,Yval,dimZ):
    
    # create model
    model = Sequential()
    model.add(LSTM(modelData.nhidden, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
    model.add(Dense(dimZ))
    model.compile(loss='mse', optimizer='adam')

    # train model
    model.fit(Xtrain, Ytrain, epochs=10, validation_data=(Xval,Yval), batch_size=72, verbose=2, shuffle=True)
    
    return model

