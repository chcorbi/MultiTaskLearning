#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 19:14:19 2017

@author: corbi
"""

import sys
import numpy as np
from time import time
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from loadData import load_toy_dataset, load_school_dataset, load_sarkos_dataset
from RandomMTLRegressor import randompred
from mult_ind_SVM import mult_ind_SVM
from AlternatingStructureOptimization import AlternatingStructureOptimization
from ConvexAlternatingStructureOptimization import ConvexAlternatingStructureOptimization
from ClusteredRegression import ClusteredLinearRegression


def compute_scores(X,y, model, n_splits=5, test_size=0.30, gridsearch=False, verbose=False):
    """
    Compute the nrMSE score for a given model and a given dataset (X,y)
    """        
    t0 = time() 
    nrMSE = []
   
    # Shuffle split
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    i=1
    for train_index, test_index in ss.split(X):
        t1 = time()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if verbose:
                print("Random shuffle split %d"%i)
                
        if gridsearch==True:
            grid = GridSearchCV(model, cv=3, param_grid=model.params,verbose=1)  
            grid.fit(X_train,y_train)
            print (grid.best_estimator_)
        else:
            model.fit(X_train,y_train)

        nrMSE.append(1. - model.score(X_test,y_test))
        i+=1
        
        if verbose:
                print ("....run in %fs" % (time() - t1) )
        
    print("Total run in %fs" % (time() - t0))
    if n_splits==1:
        return nrMSE[0]
    else:
        return [np.mean(nrMSE),np.var(nrMSE)]


if __name__=='__main__':
        # Get choices
        dataset = sys.argv[1]
        algo = sys.argv[2]
        splits = int(sys.argv[3])
        test_size = float(sys.argv[4])
        
        if test_size>=1:
                print("Test size > 1.")
                sys.exit()
                
        # Generate dataset     
        if dataset=="toy":
                X, y, E = load_toy_dataset()
        elif dataset=="school":
                X, y = load_school_dataset()  
        elif dataset=="sarkos":
                X, y = load_sarkos_dataset()
        else:
                print("Unkown dataset.")
                sys.exit()
        
        m=len(np.unique(X[:,-1]))       
        
        # Initialize chosen algorithm
        if algo=="random":
                modele = randompred()
        elif algo=="svm":
                modele = mult_ind_SVM(m=m)
        elif algo=="aso":
                lbda = np.ones((1,m))*0.225
                modele = AlternatingStructureOptimization(lbda=lbda,m=m, d=X.shape[1]-1, h=3)
        elif algo=="caso":
                alpha = 0.225
                beta = 0.15
                modele = ConvexAlternatingStructureOptimization(alpha=alpha, beta=beta,m=m, d=X.shape[1]-1, h=3)
        elif algo=="cmtl":
                epsilon = 0.5
                epsilon_m = 0.2*epsilon
                epsilon_b = 3.5*epsilon
                epsilon_w = 4.5*epsilon
                r=3
                modele = ClusteredLinearRegression(r, m, epsilon_m, epsilon_w, epsilon_b, mu=2.5)
        elif algo=="cmtl_e":
                epsilon = 0.5
                epsilon_m = 0.2*epsilon
                epsilon_b = 3.5*epsilon
                epsilon_w = 4.5*epsilon
                r=E.shape[1]               
                modele = ClusteredLinearRegression(r, m, epsilon_m, epsilon_w, epsilon_b, E,mu=2.5)
        
        # Compute score
        nrMSE = compute_scores(X,y, modele, n_splits=splits, test_size=test_size)
        
        if splits==1:
                print("nrMSE score: %f, +/- %f " % (nrMSE,0))
        else:
                print("nrMSE score: %f, +/- %f " % (nrMSE[0],nrMSE[1]))