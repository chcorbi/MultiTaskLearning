#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 19:06:43 2017

@author: corbi
"""

import scipy.io
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from scipy.optimize import fmin_l_bfgs_b


class AlternatingStructureOptimization(BaseEstimator):
    
    def __init__(self, lbda, m, d, h, n_iter=5):
        self.m=m
        self.d=d
        self.h=h
        self.n_iter=n_iter
        self.lbda = lbda
        self.params={"h": np.arange(3,int(d/3))}
        
        self.U = np.zeros((self.d,self.m))
        self.U0 = np.ones((self.d,self.m))
        self.V = np.zeros((self.h,self.m))
        self.W = np.zeros((self.d,self.m))
        self.theta = np.ones((self.h,self.d))

    def fit(self, X, y):
        for it in range(self.n_iter):
            if it%10==0:
                print ("Iteration %d..." %(it+1))
                
            for l in range(1,self.m):
                idx=np.where(X[:,self.d]==l)[0]
                X_l = X[idx,:self.d]
                y_l = np.ravel(y[idx,:1])

                self.V[:,l] = np.dot(self.theta,self.W)[:,l]

                model = optim_ASO( X=X_l, y=y_l, theta=self.theta, v=self.V[:,l], 
                                    lbda=self.lbda[:,l])

                self.U[:,l] = l_bfgs_b(self.U0[:,l], model, n_iter=self.n_iter)
                self.W[:,l] =  self.U[:,l] + np.dot(self.theta.T,self.V[:,l])

            V1, D, V2 = scipy.linalg.svd(np.sqrt(self.lbda)*self.W)            
            self.theta = V1.T[np.arange(self.h),:]
            self.V = np.dot(self.theta,self.W)
               
    def predict(self, X):
        y_pred = np.zeros((X.shape[0],2))
        for l in range(1,self.m):
            idx=np.where(X[:,self.d]==l)[0]
            X_l = X[idx,:self.d]
            y_pred[idx,0]=np.dot(self.U[:,l] + np.dot(self.theta.T,self.V)[:,l],X_l.T)
            y_pred[idx,1]=l
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return 1.- np.sqrt(mean_squared_error(y[:,0], y_pred[:,0]))/(np.max(y[:,0])-np.min(y[:,0]))

def l_bfgs_b(x_init, model, n_iter=500, bounds=None, callback=None, **kwargs):
    """
    l-BFGS-b algorithm
    """
    x, _, _ = fmin_l_bfgs_b(model.loss, x_init, model.grad, bounds=bounds, pgtol=1e-20, callback=callback)
    return x

    
class optim_ASO():
    
    def __init__(self, X, y, theta, v, lbda):
        # model param
        self.X = X
        self.y = y
        self.theta = theta
        self.v = v
        self.n = X.shape[0]
        self.lbda=lbda

    def loss(self, u):
        """"
        loss of the optim problem
        """
        f = np.dot(u.T+np.dot(self.v.T,self.theta), self.X.T)
        return (1./self.n)*np.sum((f-self.y)**2)+self.lbda*np.linalg.norm(u)**2
    
    def grad(self, u):
        """
        gradient of the optim problem
        """
        f = np.dot(u.T+np.dot(self.v.T,self.theta), self.X.T)
        return ((2./self.n)*np.dot(self.X.T,(f-self.y))+(2./self.n)*self.lbda*u)

        
