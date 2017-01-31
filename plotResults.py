#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 22:33:42 2017

@author: corbi
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from loadData import load_toy_dataset, load_school_dataset, load_sarcos_dataset
from mult_ind_SVM import mult_ind_SVM
from AlternatingStructureOptimization import AlternatingStructureOptimization
from ConvexAlternatingStructureOptimization import ConvexAlternatingStructureOptimization
from ClusteredRegression import ClusteredLinearRegression
from computeScores import compute_scores

def plot_results(X,y, name, C, r, h, n_splits=5, gridsearch=False):
    """
    Run each modele n_splits times and plot the average nrMSE score
    """  
    m=len(np.unique(X[:,-1]))    
    test_size = [0.30, 0.40, 0.50, 0.60]

    nrMSE_SVM = []        
    nrMSE_CMTL = []   
    nrMSE_CMTLE = []
    nrMSE_ASO = []  
    nrMSE_cASO = []
              
    for i,size in enumerate(test_size):
        print ("======================= %d / %d : test_size=%f =======================" % (i+1,len(test_size),size))
        
        print("------------Run ASO...")
        lbda = np.ones((1,m))*0.225
        ASO = AlternatingStructureOptimization(lbda=lbda,m=m, d=X.shape[1]-1, h=h)
        nrMSE_ASO.append(compute_scores(X,y, ASO, n_splits=splits, test_size=size)[0])

        print("------------Run cASO...")
        alpha = 0.225
        beta = 0.15
        cASO = ConvexAlternatingStructureOptimization(alpha=alpha, beta=beta,m=m, d=X.shape[1]-1, h=h)
        nrMSE_cASO.append(compute_scores(X,y, cASO, n_splits=splits, test_size=size)[0])
        
        print("------------Run CMTL...")
        epsilon = 0.5
        epsilon_m = 0.2*epsilon
        epsilon_b = 3.5*epsilon
        epsilon_w = 4.5*epsilon      
        CMTL = ClusteredLinearRegression(r, m, epsilon_m, epsilon_w, epsilon_b, mu=2.5)
        nrMSE_CMTL.append(compute_scores(X,y, CMTL, n_splits=splits, test_size=size)[0])
        
        if name=="toy":
           print("------------Run CMTLE...")
           CMTLE = ClusteredLinearRegression(r, m, epsilon_m, epsilon_w, epsilon_b, mu=2.5)
           nrMSE_CMTLE.append(compute_scores(X,y, CMTLE, n_splits=splits, test_size=size)[0])                

        print("------------Run SVM...")
        SVM = mult_ind_SVM(m=m, C=C)            
        nrMSE_SVM.append(compute_scores(X,y, SVM, n_splits=splits, test_size=size)[0])
        
        print ("SVM: %f,   CMTL: %f,   ASO: %f,   cASO: %f" % (nrMSE_SVM[i], nrMSE_CMTL[i], nrMSE_ASO[i], nrMSE_cASO[i]))
   
    fig, ax = plt.subplots()
    ax.set_title("nrMSE for %s dataset, run %d times" % (name, n_splits))
    ax.plot(test_size,nrMSE_SVM, label='M-SVM')
    ax.plot(test_size,nrMSE_CMTL, label='CMTL')
    if name=='toy':
        ax.plot(test_size,nrMSE_CMTLE, label='CMTL_E')    
    ax.plot(test_size,nrMSE_ASO,label='ASO')
    ax.plot(test_size,nrMSE_cASO,label='cASO')
    #ax.set_ylim([0, 0.35])
    ax.set_xlabel('% Test size')
    ax.set_ylabel('nrMSE')
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.show() 


if __name__=='__main__':
        # Get choices
        dataset = sys.argv[1]
        splits = int(sys.argv[2])
                
        # Generate dataset     
        if dataset=="toy":
                X, y, E = load_toy_dataset()
                C = 1e2
                r = 3
                h = 3
        elif dataset=="school":
                X, y = load_school_dataset()
                C = 1e1
                r = 6
                h = 3
        elif dataset=="sarcos":
                X, y = load_sarcos_dataset()
                C = 1e4
                r = 6
                h = 3
        else:
                print("Unkown dataset.")
                sys.exit()

        plot_results(X,y,n_splits=splits, name=dataset, C=C, r=r, h=h)
