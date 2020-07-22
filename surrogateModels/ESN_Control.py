#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.sparse as sparse
#from scipy.sparse import linalg


class ESNControl():
    
    def __init__(self, n_inputs, n_control, n_outputs,
                 approx_res_size=500, spectral_radius=0.9, sparsity=0.99,
                 sigma=0.99, beta=0.0001, data_shift = 0.0,data_scale = 1.0):
        
    
        # check for proper dimensionality of all arguments and write them down.
        self.n_inputs = n_inputs
        self.n_control = n_control
        self.n_outputs = n_outputs
        
        self.n_reservoir = int(np.floor(approx_res_size/n_inputs) * n_inputs)
        
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.sigma = sigma
        self.beta = beta
            
        self.Wout = dict()
        
        # Initilize input and reservoir layer
        self.init_reservoir()
        
        self.data_shift = data_shift # normally mean
        self.data_scale = data_scale # normally std of trainingdata


    def init_reservoir(self):
        
        # Create Reservoir with size n_reservoir and given spectral radius 
        A = sparse.rand(self.n_reservoir,self.n_reservoir,density=self.sparsity)

        max_eigv,_ = sparse.linalg.eigs(A, k=1, which='LM')
        
        # Scale matrix according to desired spectral radius
        self.A = A / (np.abs(max_eigv) * self.spectral_radius)
        
        # Create input layer
        self.Win = self.sigma * (-1 + 2 * np.random.rand(self.n_reservoir,self.n_inputs))
            

    def eval_reservoir_layer(self, input, pred_len, state_init=None):
        
        # scale input
        input = (input-self.data_shift)/self.data_scale

        states = np.zeros([self.n_reservoir, pred_len + 1])
        
        if state_init is not None:
            states[:,0] = state_init[:,0]
        else:
            states[:,0] = np.ones([self.n_reservoir,])
    
        for i in range(states.shape[1]-1):
            states[:,i+1] = np.tanh(self.A.dot(states[:,i]) + np.dot(self.Win,input[i,:]))
        return states[:,1:]
    
    
    def eval_output_layer(self, x, iu):
        
        predict_length = x.shape[1]
        output = np.zeros((predict_length,self.n_outputs))
        for i in range(predict_length):
            x_aug = x.copy()
            for j in range(2,np.shape(x_aug)[0]-2):
                if (np.mod(j,2)==0):
                    x_aug[j] = (x[j-1]*x[j-2]).copy()
            out = np.squeeze(np.asarray(np.dot(self.Wout[iu],x_aug)))
            output[i,:] = out
        
        # unscale output
        output = output * self.data_scale + self.data_shift
        
        return output

 
    def train(self, states, iuTrain, dataY):

        # scale training output
        dataY = (dataY-self.data_shift)/self.data_scale
        
        for i in range(self.n_control):
            
            m_ctrl = (iuTrain == i)
            m_ctrl = m_ctrl.reshape([m_ctrl.shape[0],])
            
            idenmat = self.beta * sparse.identity(self.n_reservoir)
            states_ctrl = states[:,m_ctrl].copy()
            states2 = states_ctrl.copy()
            for j in range(2,np.shape(states2)[0]-2):
                if (np.mod(j,2)==0):
                    states2[j,:] = (states_ctrl[j-1,:]*states_ctrl[j-2,:]).copy()
            U = np.dot(states2,states2.transpose()) + idenmat
            Uinv = np.linalg.inv(U)
            
            # train output layer via linear regression
            Wout = np.dot(Uinv,np.dot(states2,dataY[m_ctrl,:]))
            self.Wout[i] = Wout.T   
        
        print("Finished ESN-Training")



