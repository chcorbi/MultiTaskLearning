#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 23:47:18 2017

@author: corbi
"""

import scipy.io
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from AlternatingStructureOptimization import l_bfgs_b


class ConvexAlternatingStructureOptimization(BaseEstimator):
    
    def __init__(self, alpha,beta, m, d, h=3, n_iter=5, C=1., s=1):
        self.m=m
        self.d=d
        self.h=h
        self.n_iter=n_iter
        self.C=C
        self.s=s
        self.alpha=alpha
        self.eta=beta/alpha   
        
        self.M = np.eye(self.d)*self.h/self.d
        self.U = np.zeros((self.d,self.m))
        self.W0 = np.ones((self.d,self.m))
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
                model = optim_W_cASO( X=X_l, y=y_l, M=self.M, alpha=self.alpha, eta=self.eta, 
                                                        C=self.C, s=self.s)
            
                self.W[:,l] = l_bfgs_b(self.W0[:,l], model, n_iter=self.n_iter)

            P1, D, P2 = scipy.linalg.svd(self.W)
            q = np.linalg.matrix_rank(self.W)

            gammas_0 = np.ones(q)*self.h/q
            sigmas = D[:q]
    
            model_gammas = optim_M_cASO(sigmas=sigmas, eta=self.eta)         
            cons = ({'type': 'eq',
                     'fun' : lambda x: np.sum(x) - self.h,
                     'jac' : lambda x: np.array([1]*x.shape[0]) })
            bounds=[ (0,1)] * gammas_0.shape[0]
            res = scipy.optimize.minimize(model_gammas.loss, x0=gammas_0,
                            jac=model_gammas.grad, method='SLSQP', bounds=bounds, constraints=cons)
            gammas=res['x']
            Gamma = np.diag(np.append(gammas, np.zeros((self.d-len(gammas)))))                       
            self.M = np.dot(P1,np.dot(Gamma,P1.T))

            _, M_eigenvectors = np.linalg.eig(self.M)

            self.theta = M_eigenvectors[:,range(self.h)].T
            self.V = np.dot(self.theta,self.W)
            self.U = self.W - np.dot(self.theta.T,self.V)

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
        return 1. - np.sqrt(mean_squared_error(y[:,0], y_pred[:,0]))/(np.max(y[:,0])-np.min(y[:,0]))
        
        
        
class optim_W_cASO():
    
    def __init__(self, X, y, M, alpha, eta, C=1.0, s=1.0):
        # model param
        self.X = X
        self.y = y
        self.M = M
        self.alpha=alpha
        self.eta=eta        
        self.C=C
        self.s=s
        self.m = M.shape[0]
        self.d = X.shape[1]
        self.n = X.shape[0]
        
    def loss(self, W):
        """"loss of the optim problem"""
        inv = np.linalg.solve(self.eta*np.eye(self.d)+self.M,W.T)
        g = self.alpha*self.eta*(1.+self.eta)*(np.dot(W,inv)) 
        return 0.5*np.linalg.norm(self.y-np.dot(W.T,self.X.T))**2 +g
    
    def grad(self, W):
        """gradient of the optim problem"""
        inv = np.linalg.solve(self.eta*np.eye(self.d)+self.M,W)
        grad_g = 2. * self.alpha*self.eta*(1.+self.eta)*inv
        return np.dot(self.X.T,(np.dot(W.T,self.X.T)-self.y)) + np.dot(W,grad_g.T)

        
        
class optim_M_cASO():
    
    def __init__(self, sigmas, eta):
        # model param
        self.sigmas = sigmas
        self.eta=eta        
        self.q = sigmas.shape[0]
        
    def loss(self, gammas):
        """"loss of the optim problem"""
        loss = 0
        for i in range(self.q):
            loss += self.sigmas[i]**2/(self.eta+gammas[i])
        return loss 
    
    def grad(self, gammas):
        """"loss of the optim problem"""
        grad = np.zeros(self.q)
        for i in range(self.q):
            grad[i] = -self.sigmas[i]**2/((self.eta+gammas[i])**2)
        return grad