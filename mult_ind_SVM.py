#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:51:59 2017

@author: corbi
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


class mult_ind_SVM(BaseEstimator):
    
    def __init__(self,m, C):
        self.m=m
        self.C=C
        self.dict_reg= {}
        for l in range(1,self.m):      
            self.dict_reg[l]= SVR(kernel='rbf', C=C, gamma=0.1,epsilon=0.01)

    def fit(self, X, y):
        d=X.shape[1]
        for l in range(1,self.m):
            print ("Fit task %d..." % l)
            idx=np.where(X[:,d-1]==l)[0]
            X_l = X[idx,:d-1]
            y_l = np.ravel(y[idx,:1])
            self.dict_reg[l].fit(X_l,y_l)          
    
    def predict(self, X):
        d=X.shape[1]
        y_pred = np.zeros((X.shape[0],2))
        for l in range(1,self.m):
            idx=np.where(X[:,d-1]==l)[0]
            X_l = X[idx,:d-1]
            y_pred[idx,0]=self.dict_reg[l].predict(X_l)
            y_pred[idx,1]=l
        return y_pred.astype(int)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return 1. - np.sqrt(mean_squared_error(y[:,0], y_pred[:,0]))/(np.max(y[:,0])-np.min(y[:,0]))