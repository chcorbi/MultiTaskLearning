#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 22:33:42 2017

@author: corbi
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from loadData import load_toy_dataset, load_school_dataset, load_sarkos_dataset
from RandomMTLRegressor import randompred
from mult_ind_SVM import mult_ind_SVM
from AlternatingStructureOptimization import AlternatingStructureOptimization
from ConvexAlternatingStructureOptimization import ConvexAlternatingStructureOptimization
from ClusteredRegression import ClusteredLinearRegression
from computeScores import compute_scores

def plot_results(X,y, name, n_splits=5, gridsearch=False):
    m=len(np.unique(X[:,-1]))    
    test_size = [0.30, 0.40, 0.50, 0.60]

    nrMSE_SVM = []        
    nrMSE_CMTL = []   
    nrMSE_CMTLE = []
    nrMSE_ASO = []  
    nrMSE_cASO = []
              
    for i,size in enumerate(test_size):
        print ("======================= %d / %d : test_size=%f =======================" % (i+1,len(test_size),size))
        
        # Run ASO
        lbda = np.ones((1,m))*0.225
        ASO = AlternatingStructureOptimization(lbda=lbda,m=m, d=X.shape[1]-1, h=3)
        nrMSE_ASO.append(compute_scores(X,y, ASO, n_splits=splits, test_size=size)[0])

        # Run cASO
        alpha = 0.225
        beta = 0.15
        cASO = ConvexAlternatingStructureOptimization(alpha=alpha, beta=beta,m=m, d=X.shape[1]-1, h=3)
        nrMSE_cASO.append(compute_scores(X,y, cASO, n_splits=splits, test_size=size)[0])
        
        # Run CTML
        epsilon = 0.5
        epsilon_m = 0.2*epsilon
        epsilon_b = 3.5*epsilon
        epsilon_w = 4.5*epsilon
        r=3       
        CMTL = ClusteredLinearRegression(r, m, epsilon_m, epsilon_w, epsilon_b, mu=2.5)
        nrMSE_CMTL.append(compute_scores(X,y, CMTL, n_splits=splits, test_size=size)[0])
        
        if name=="toy":
           CMTLE = ClusteredLinearRegression(r, m, epsilon_m, epsilon_w, epsilon_b, mu=2.5)
           nrMSE_CMTLE.append(compute_scores(X,y, CMTLE, n_splits=splits, test_size=size)[0])                

        # Run SVM
        SVM = mult_ind_SVM(m=m)            
        nrMSE_SVM.append(compute_scores(X,y, SVM, n_splits=splits, test_size=size)[0])
   
    fig, ax = plt.subplots()
    ax.set_title("nrMSE for %s dataset, run %d times" % (name, n_splits))
    ax.plot(test_size,nrMSE_SVM, label='Multi SVM')
    ax.plot(test_size,nrMSE_CMTL, label='Clustered MTL')
    if name=='toy':
        ax.plot(test_size,nrMSE_CMTLE, label='Clustered MTL w. clusters')    
    ax.plot(test_size,nrMSE_ASO,label='ASO')
    ax.set_ylim([0, 0.35])
    ax.legend(loc='lower right', shadow=True)
    plt.show() 


if __name__=='__main__':
        # Get choices
        dataset = sys.argv[1]
        splits = int(sys.argv[2])
                
        if dataset=="toy":
                X, y, E = load_toy_dataset()
        elif dataset=="school":
                X, y = load_school_dataset()  
        elif dataset=="sarkos":
                X, y = load_sarkos_dataset()
        else:
                print("Unkown dataset.")
                sys.exit()

        plot_results(X,y,n_splits=splits, name=dataset)
