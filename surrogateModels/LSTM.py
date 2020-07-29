import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

import pickle

import copy


def timeTMap(z0, t0, iu, modelData):
    
    # prepare input data
    temp = z0.reshape((1, modelData.nDelay + 1, modelData.dimZ))
    tt = temp[:,::-1,:]
    tt = (tt-modelData.data_mean[iu])/modelData.data_std[iu];
    
    # model predicition
    y_pred = modelData.LSTM_model[iu].predict_on_batch(tt)

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

    data_trainonline_x = list()
    data_trainonline_y = list()
    
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
        model[i]._experimental_run_tf_function = True

        data_trainonline_x.append([])
        data_trainonline_y.append([])
    
    
    # Save LSTM models and important parameters
    setattr(modelData, 'LSTM_model', model)
    setattr(modelData, 'dimZ', dimZ)
    setattr(modelData, 'data_mean', mean)
    setattr(modelData, 'data_std', std)

    setattr(modelData, 'data_trainonline_x', data_trainonline_x)
    setattr(modelData, 'data_trainonline_y', data_trainonline_y)

    return modelData


def LSTM_datasets(modelData, dataX, dataY, train_size,val_size,mean=None,std=None):
    
    dimZ = int(dataX.shape[1] / (modelData.nDelay + 1))
    dataY = dataY[:,:dimZ]

    if mean is None:
        mean = np.mean(dataX[:,0:dimZ])
    if std is None:
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
    model.fit(Xtrain, Ytrain, epochs=modelData.epochs, validation_data=(Xval,Yval), batch_size=modelData.batch_size, verbose=2, shuffle=True)



    return model


def loadSurrogateModel(loadPath):

    fIn = open(loadPath + '_LSTM.pkl', 'rb')
    modelData = pickle.load(fIn)

    model = dict()
    for iu in range(modelData.uGrid.shape[0]):
        model[iu] = load_model(loadPath + '_' + str(iu))

    setattr(modelData, 'LSTM_model', model)

    return modelData


def saveSurrogateModel(modelData,savePath):

    modelData_temp = copy.copy(modelData)
    modelData_temp.LSTM_model = []

    with open(savePath + '_LSTM.pkl', 'wb') as fOut:
        pickle.dump(modelData_temp, fOut)
    for iu in range(modelData.uGrid.shape[0]):
        modelData.LSTM_model[iu].save(savePath + '_' + str(iu))

def updateSurrogateModel(modelData, z_rawData, _, iu):

    current_control = int(iu[-1,0])

    if current_control == -1:
        print("uuubs")
    print("current_control:", current_control)

    test = z_rawData[::modelData.nLag, :]
    temp = test.reshape((1, modelData.nDelay + 2, modelData.dimZ))
    z = (temp - modelData.data_mean[current_control]) / modelData.data_std[current_control]

    z0 = z[:,:-1,:]
    z1 = z[:,-1,:]

    if modelData.data_trainonline_x[current_control] == []:
        modelData.data_trainonline_x[current_control] = z0
        modelData.data_trainonline_y[current_control] = z1
    else:
        modelData.data_trainonline_x[current_control] = np.concatenate((modelData.data_trainonline_x[current_control], z0))
        modelData.data_trainonline_y[current_control] = np.concatenate((modelData.data_trainonline_y[current_control], z1))

    updatePerformed = False

    if len(modelData.data_trainonline_x[current_control]) > modelData.batch_size: # modelData.batch_size:
        Xtrain = modelData.data_trainonline_x[current_control]
        Ytrain = modelData.data_trainonline_y[current_control]
        modelData.LSTM_model[current_control].fit(Xtrain, Ytrain, epochs=1, batch_size=modelData.batch_size, verbose=2, shuffle=True)

        #modelData.LSTM_model[current_control].fit(Xtrain, Ytrain, epochs=int(modelData.epochs/5), batch_size=modelData.batch_size, verbose=2,shuffle=True)
        modelData.data_trainonline_x[current_control] = []
        modelData.data_trainonline_y[current_control] = []

        updatePerformed = True


    return modelData, updatePerformed
