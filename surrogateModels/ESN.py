import numpy as np

from surrogateModels.ESN_Control import ESNControl


def timeTMap(z0, t0, iu, modelData):
    
    tt = z0.reshape([1,modelData.dimZ * (modelData.nDelay + 1)])

    state = modelData.ESN.eval_reservoir_layer(tt, 1, get_state(modelData, t0 - modelData.h))
    
    # eval output layer with current state 
    y_pred = modelData.ESN.eval_output_layer(state, iu)
    
    # transform output
    z = np.concatenate((y_pred.T[:,0], z0[:-modelData.dimZ]))
    # add time step
    t = t0 + modelData.h
    set_state(modelData, state, t)
    
    return z, t, modelData


def createSurrogateModel(modelData, data):
    
    dimZ = data.z[0].shape[1]
    n_control = modelData.uGrid.shape[0]
    
    
    X = []
    Y = []
    iuTrain = []
    
    for i in range(len(data.z)):
        dataPrep = data.z[i][::modelData.nLag]
        iuPrep = data.iu[i][::modelData.nLag]
        
        X.append(dataPrep[:-1,:])
        Y.append(dataPrep[1:,:])
    
        iuTrain.append(iuPrep[:-1,:])
    
    mean = np.mean(np.concatenate(X,axis=0),axis=0)
    std = np.std(np.concatenate(X,axis=0),axis=0)

    # create instance of ESN Class
    ESN = ESNControl(dimZ, n_control, dimZ, 
                     modelData.approx_res_size, modelData.spectral_radius, modelData.sparsity, 
                     data_shift=mean,data_scale=std)
        
    states = []
    
    # eval reservoire state (input + reservoir layer) for each trainingsequenze
    for i in range(len(data.z)):

        pred_len = X[i].shape[0]
        states.append(ESN.eval_reservoir_layer(X[i], pred_len))
      
        
    # concatenate all lists of trainingdata   
    Y = np.concatenate(Y,axis=0)    
    iuTrain = np.concatenate(iuTrain,axis=0)     
    states = np.concatenate(states,axis=1)    
    
    # train output layer of ESN with states and Y
    ESN.train(states, iuTrain, Y)

    state = np.zeros([states.shape[0],1])
    
    # Save ESN instance and important parameters
    setattr(modelData, 'ESN', ESN)
    setattr(modelData, 'dimZ', dimZ)
    setattr(modelData, 'state', state)

    return modelData
    

def set_state(modelData, state, t):
    if modelData.state.shape[1] < int(t / modelData.h) + 1:
        modelData.state = np.concatenate((modelData.state, np.zeros([modelData.ESN.n_reservoir,1])),axis=1)
    modelData.state[:,int(t / modelData.h)] = state[:,0]
    
def get_state(modelData, t):
    return np.reshape(modelData.state[:,int(t / modelData.h)],[modelData.ESN.n_reservoir,1])
