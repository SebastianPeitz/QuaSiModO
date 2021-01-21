import numpy as np

from surrogateModels.ESN_Control import ESNControl


def timeTMap(z0, t0, iu, modelData):
    
    tt = z0.reshape([1, modelData.dimZ* (modelData.nDelay + 1)])
    
    if modelData.nDelay > 0:
        
        temp = tt[:,:modelData.dimZ]
        for i in range(modelData.stepDelay,modelData.nDelay + 1, modelData.stepDelay):
            temp = np.concatenate((tt[:,(i)*modelData.dimZ:(i+1)*modelData.dimZ],temp),axis = 1)
        
        tt = temp
    
    state = modelData.ESN.eval_reservoir_layer(tt, 1, get_state(modelData, t0))
    
    # eval output layer with current state 
    y_pred = modelData.ESN.eval_output_layer(state, iu)
    
    # transform output
    z = np.concatenate((y_pred.T[:,0], z0[:-modelData.dimZ]))
    # add time step
    t = t0 + modelData.h
    set_state(modelData, state, t)
    
    return z, t, modelData


def createSurrogateModel(modelData, data):
    
    if not hasattr(modelData, 'stepDelay'):
        setattr(modelData, 'stepDelay', 1)        
    
    dimZ = data.z[0].shape[1] 
    dimZD = dimZ * int(np.ceil((modelData.nDelay + 1) / modelData.stepDelay))
    n_control = modelData.uGrid.shape[0]
    
    
    X = []
    Y = []
    iuTrain = []
    
    trans = int(modelData.h * 2.0)
    
    for i in range(len(data.z)):
        dataPrep = data.z[i][trans::modelData.nLag]
        iuPrep_temp = data.iu[i][trans ::modelData.nLag]
        test_t = data.t[i][trans ::modelData.nLag]
        
        temp = dataPrep[modelData.nDelay:-1,:]
        for i in range(modelData.stepDelay,modelData.nDelay + 1,modelData.stepDelay):
            temp = np.concatenate((dataPrep[modelData.nDelay - (i):-(i+1),:],temp),axis = 1)
        
        X.append(temp)
        iuPrep = iuPrep_temp[modelData.nDelay:-1]
        Y.append(dataPrep[modelData.nDelay + 1:, :])

        if iuPrep.shape[0] == dataPrep.shape[0]:
            iuTrain.append(iuPrep[:-1,:])
        else:
            iuTrain.append(iuPrep)
    
    mean = np.mean(np.concatenate(X, axis=0)[:,:dimZ], axis=0)
    std = np.std(np.concatenate(X, axis=0)[:,:dimZ], axis=0)
    

    # create instance of ESN Class
    ESN = ESNControl(dimZD, n_control, dimZ, 
                      modelData.approx_res_size, modelData.spectral_radius, modelData.sparsity, 
                      data_shift=mean,data_scale=std)
        
    states = []
    
    # eval reservoire state (input + reservoir layer) for each trainingsequenze
    for i in range(len(data.z)):

        pred_len = X[i].shape[0]
        states.append(ESN.eval_reservoir_layer(X[i], pred_len))
      
        
    # concatenate all lists of trainingdata   
    Y = np.concatenate(Y, axis=0)
    iuTrain = np.concatenate(iuTrain, axis=0)
    states = np.concatenate(states, axis=1)
    
    # train output layer of ESN with states and Y
    ESN.train(states, iuTrain, Y)

    state = np.reshape(states[:, -1], [states.shape[0], 1])   #np.zeros([states.shape[0],1])
    
    # Save ESN instance and important parameters
    setattr(modelData, 'ESN', ESN)
    setattr(modelData, 'dimZ', dimZ)
    setattr(modelData, 'nDelay', modelData.nDelay)
    setattr(modelData, 'state', state)

    return modelData
    

def set_state(modelData, state, t):
    step = int(np.round(t / modelData.h)) - modelData.nDelay
    if modelData.state.shape[1] < step + 1:
        modelData.state = np.concatenate((modelData.state, np.zeros([modelData.ESN.n_reservoir, 1])), axis=1)
    modelData.state[:, step] = state[:, 0]


def get_state(modelData, t):
    step = int(np.round(t / modelData.h)) - modelData.nDelay
    return np.reshape(modelData.state[:, step], [modelData.ESN.n_reservoir, 1])
