#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:41:58 2017

@author: corbi
"""

import scipy.io
import numpy as np
import pandas as pd

def _preprocessing(X,Y):
    """
    Prepare the dataset for the MTL algorithms
    """
    X_process=np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
    y_process=np.concatenate((Y[:,0].reshape(Y.shape[0],1),np.ones((Y.shape[0],1))),axis=1)
    for l in range(2,Y.shape[1]+1):
        X_l=np.concatenate((X,np.ones((X.shape[0],1))*l),axis=1)               
        X_process=np.append(X_process,X_l,axis=0)
        y_l = np.concatenate((Y[:,0].reshape(Y.shape[0],1),l*np.ones((Y.shape[0],1))),axis=1)
        y_process=np.append(y_process,y_l,axis=0)
    return X_process, y_process

def _make_true_W(d, m, r, v1=900, v2=16):
    """
    Return a weight matrix used in generating toy dataset
    """
    bws = [] # hold the base cluster
    for _ in range(r):
        bw = np.random.normal(0, np.sqrt(v1), int((d-2)/2.))
        bw = np.r_[bw, np.zeros_like(bw)]
        bws.append(bw)
    m_c = int(m / r) # nb tasks per cluster
    W = []
    E = np.zeros((m, r))
    i = 0 # i indice of tasks
    for c in range(r):
        Wc = np.empty((d-2, m_c))
        for t in range(m_c): # t indice of task within the cluster c
            w = np.random.normal(0, np.sqrt(v2), int((d-2)/2.))
            w = np.r_[w, np.zeros_like(w)]
            w += bws[c]
            Wc[:, t] = w
            E[i, c] = 1
            i += 1
        W.append(Wc)
    W = np.concatenate(W, axis=1)
    return np.r_[W, np.random.normal(0, v2, (2, m))], E

    

def load_toy_dataset(n=1000,d=12, m=9,r=3,v=150): 
    """
    Generate a toy dataset for a fix degree d, a fix number of sample n, a fix number of tasks m,
    a predifine number of clusters r and a variance from center v.
    OUTPOUT : X = featues
              y = labels
              E = residuals
    """        
    W, E = _make_true_W(d, m, r,)
    X = np.random.sample((n, d))
    Y = X.dot(W) + np.random.normal(0, np.sqrt(v), (n, m))
    X, y = _preprocessing(X,Y)
    return X,y,E
    
    
def load_school_dataset():
    """
    Load School dataset and select the first 27 tasks  for computing reasons
    """
    dataset = scipy.io.loadmat('data/school.mat')
    FEATURES_COLUMNS = ['Year_1985','Year_1986','Year_1987','FSM','VR1Percentage','Gender_Male','Gender_Female','VR_1','VR_2','VR_3',
                'Ethnic_ESWI','Ethnic_African','Ethnic_Arabe','Ethnic_Bangladeshi','Ethnic_Carribean','Ethnic_Greek','Ethnic_Indian',
                'Ethnic_Pakistani','Ethnic_Asian','Ethnic_Turkish','Ethnic_Others','SchoolGender_Mixed','SchoolGender_Male',
                'SchoolGender_Female','SchoolDenomination_Maintained','SchoolDenomination_Church','SchoolDenomination_Catholic',
                'Bias']
    
    # Dataframe representation
    X_df=pd.DataFrame(dataset['X'][:,0][0],columns=FEATURES_COLUMNS)
    y_df=pd.DataFrame(dataset['Y'][:,0][0],columns=['Exam_Score'])
    X_df['School'] = 1
    y_df['School'] = 1
        
    d = X_df.shape[1]-1
    for i in range(1,d):
        X_df_i=pd.DataFrame(dataset['X'][:,i][0],columns=FEATURES_COLUMNS)
        X_df_i['School'] = i+1  
        X_df = X_df.append(X_df_i,ignore_index=True)

        y_df_i=pd.DataFrame(dataset['Y'][:,i][0],columns=['Exam_Score'])
        y_df_i['School'] = i+1  
        y_df = y_df.append(y_df_i,ignore_index=True)  
        
    return X_df.values, y_df.values

    
def load_sarcos_dataset(set_size=1000):
    """
    Load SARCOS dataset and select the first 2000 samples for computing reasons
    """
    # Load training set
    sarcos_train = scipy.io.loadmat('data/sarcos_inv.mat')
    # Inputs  (7 joint positions, 7 joint velocities, 7 joint accelerations)
    Xtrain = sarcos_train["sarcos_inv"][:, :21]
    # Outputs (7 joint torques)
    Ytrain = sarcos_train["sarcos_inv"][:, 21:]

    # Load test set
    sarcos_test = scipy.io.loadmat("data/sarcos_inv_test.mat")
    Xtest = sarcos_test["sarcos_inv_test"][:, :21]
    Ytest = sarcos_test["sarcos_inv_test"][:, 21:]

    X = np.concatenate((Xtrain,Xtest),axis=0)
    Y = np.concatenate((Ytrain,Ytest),axis=0)

    return _preprocessing(X[:set_size,:],Y[:set_size,:])
