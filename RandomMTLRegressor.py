#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:40:09 2017

@author: corbi
"""

import numpy as np
from random import randint
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

class randompred(BaseEstimator):
    
    def __init__(self):
        self.m=139
        self.min=0
        self.max=0
        self.param_grid={}
    
    def fit(self, X, y):
        self.min=np.min(y[:,0])
        self.max=np.max(y[:,0])
    
    def predict(self,X):
        d=X.shape[1]
        y_pred = np.zeros((X.shape[0],2))
        for l in range(1,self.m):
            idx=np.where(X[:,d-1]==l)[0]
            y_pred[idx,0]=np.asarray([randint(0,100) for p in range(0,idx.shape[0])])  # Randomly generate labels between 0 and 100
            y_pred[idx,1]=l
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return 1. - np.sqrt(mean_squared_error(y[:,0], y_pred[:,0]))/(np.max(y[:,0])-np.min(y[:,0]))