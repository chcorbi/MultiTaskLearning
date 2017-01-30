#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 19:46:23 2017

@author: corbi
"""

import scipy
import numpy as np
from time import time
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


class ClusteredLinearRegression(BaseEstimator):
    def __init__(self, r, m, epsilon_m, epsilon_w, epsilon_b, E=None, mu=1e0,
                 maxiter=5000, step=1e-8, eps=5e1, verbose=True):
        """init the params
        """
        self.r = r
        self.epsilon_m = epsilon_m
        self.epsilon_w = epsilon_w
        self.epsilon_b = epsilon_b
        self.mu = mu
        self.W = None
        self.E = None
        self.true_inv_s_star = None
        if E is not None:
            self.E = E
            M = E.dot(np.linalg.inv(E.T.dot(E))).dot(E.T)
            I = np.eye(m)
            U = np.ones((m, m))
            self.true_inv_s_star = epsilon_b*(M-U) + epsilon_w*(I-M)
        self.maxiter = maxiter
        self.step = step
        self.eps = eps
        self.verbose = verbose
        self.insp_grad = []

    def get_W(self):
        """return W.
        """
        return self.W

    def get_insp_grad(self):
        """return lngrad
        """
        return self.insp_grad

    def _grad(self, W, X, Y):
        """compute the gradient of the obj function
        """
        m = len(np.unique(X[:, -1]))
        d = X.shape[1] - 1
        I = np.eye(m)
        U = np.ones((m, m))
        Pi = I - U
        alpha = 1. / self.epsilon_w
        beta = 1. / self.epsilon_b
        gamma = (m-self.r+1)*alpha + (self.r-1)*beta
        if self.true_inv_s_star is None:
            lbda_star = np.diag(_get_lambda_star(W, alpha, beta, gamma)) #XXX bug
            lbda_star = scipy.linalg.block_diag(lbda_star, np.zeros((d-m, d-m)))
            V = np.linalg.eig(W.dot(Pi).dot(W.T))[1]
            s_star = V.dot(lbda_star).dot(np.linalg.pinv(V)).real
            inv_s_star = np.linalg.pinv(s_star)
            pen_grad = inv_s_star.dot(W.dot(Pi).dot(Pi.T))
            pen_grad += inv_s_star.T.dot(W.dot(Pi).dot(Pi.T))
        else:
            inv_s_star = self.true_inv_s_star
            pen_grad = 2 * W.dot(inv_s_star)
        struct_grad = 2 * self.epsilon_m * W.dot(U)
        loss_grad = self._loss_grad(X, Y, W)
        return loss_grad + struct_grad + self.mu * pen_grad

    def _loss_grad(self, X, Y, W):
        """
        """
        m = len(np.unique(X[:, -1]))
        d = X.shape[1] - 1
        # init loop
        t = np.unique(X[:, -1])[0].astype(int)-1
        X_t = X[X[:, -1]==t][:, :-1]
        pred = X_t.dot(W[:, t])[:, None]
        Y_t = Y[X[:, -1]==t][:, :-1]
        e = pred - Y_t
        grad = X_t.T.dot(e)
        # loop
        for t in np.unique(X[:, -1])[1:].astype(int)-1:
            X_t = X[X[:, -1]==t][:, :-1]
            pred = X_t.dot(W[:, t])[:, None]
            Y_t = Y[X[:, -1]==t][:, :-1]
            e = pred - Y_t
            grad = np.c_[grad, X_t.T.dot(e)]
        return 2 * grad

    def fit(self, X, Y):
        """run the double optimisation (based on FISTA), ie fit the model
        """
        m = len(np.unique(X[:, -1]))
        d = X.shape[1] - 1
        W = np.random.sample((d, m))
        W_old = np.random.sample((d, m))
        Z = W
        old_grad = np.zeros_like(W)
        t = 1
        t_old = 0
        for idx in range(self.maxiter):
            t = 0.5 * (1 + np.sqrt(1+4*t**2))
            grad = self._grad(W, X, Y)
            W = Z - self.step * grad
            Z = W + (t_old - 1) / t *(W - W_old)
            W_old = W

            norm_grad = np.linalg.norm(grad)
            self.insp_grad.append(norm_grad)
            if self.verbose and ((idx%100)==0):
                print("iter: %d |df|=%f" % (idx,  norm_grad))
            if np.linalg.norm(grad - old_grad) < self.eps:
                print("iter: %d |df|=%f" % (idx,  norm_grad))
                break
            old_grad = grad
        self.W = W

    def predict(self, X):
        """return the prediction for the given X
        """
        n = X.shape[0]
        pred = np.empty((n, 2))
        pred[:, 1] = X[:, -1]
        for t in np.unique(X[:, -1]).astype(int)-1:
            X_t = X[X[:, -1]==t][:, :-1]
            pred[pred[:, 1]==t, 0] = X_t.dot(self.W[:, t])
        return pred


    def score(self, X, y):
        y_pred = self.predict(X)
        return 1.- np.sqrt(mean_squared_error(y[:,0], y_pred[:,0]))/(np.max(y[:,0])-np.min(y[:,0]))


def _get_lambda_star(W, alpha, beta, gamma): #XXX bug
    """return the optimal lambda to compute sigma_c_star
    """
    # code directly taken from Laurent Jacob's demo:
    # see: https://lbbe.univ-lyon1.fr/-Jacob-Laurent-.html?lang=fr
    _, s, _ = np.linalg.svd(W)
    m = len(s)
    s2 = s**2
    s2beta2 = s2 / beta**2
    s2alpha2 = s2 / alpha**2
    palpha = -1
    pbeta = 0
    chidx = pbeta
    chval = 2
    partition = np.ones(m)
    b = s2beta2[0]
    nustar = 0
    while not ((pbeta+1 < len(s2beta2)) or (palpha+1 < len(s2alpha2))):
        # update a, b
        a = b
        partition[chidx] = chval
        if ((pbeta < len(s2beta2)) and \
            (palpha < len(s2alpha2) ) and \
            (s2beta2[pbeta+1] > s2alpha2[palpha+1])) or \
            (pbeta >= len(s2beta2) and \
            (palpha < len(s2alpha2))):
            palpha = palpha + 1
            chidx = palpha
            chval = 3
            b = s2alpha2[palpha]
        else:
            pbeta = pbeta + 1
            chidx = pbeta
            chval = 2
            b = s2beta2[pbeta]

        # compute nustar
        n_p = (partition == 1).sum()
        ssi = s[partition == 2].sum()
        n_m = (partition == 3).sum()
        snsden = gamma - alpha*n_m - beta*n_p

        # breaking conditions
        if not ssi:
            if snsden <= 0:
                continue
            else:
                nustar = a
                break
        if not snsden:
            continue
        sqrtnustar = ssi / snsden
        if sqrtnustar < 0:
            continue
        nustar = sqrtnustar**2
        if nustar < b:
            if nustar <= a:
                    nustar = a
            break

    # compute lbda
    lbda = np.zeros(m)
    lbda[partition == 1] = beta
    lbda[partition == 2] = s[partition == 2] / np.sqrt(nustar)
    lbda[partition == 3] = alpha
    return lbda


if __name__ == '__main__':
    """run a simple regression example with a toy dataset
    """
    print ("running Multi-tasks learning on toy dataset...")
    n, d, m, r = 2000, 30, 4, 2
    X, Y, E = toy_dataset(n, d, m, r)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    epsilon = 0.5
    epsilon_m = 0.2*epsilon
    epsilon_b = 3.5*epsilon
    epsilon_w = 4.5*epsilon

    # linReg with estimated clusters
    t0 = time()
    reg1 = ClusteredLinearRegression(r, epsilon_m, epsilon_w, epsilon_b, step=1e-7, mu=2.5)
    reg1.fit(X_train, Y_train)
    pred1 = reg1.predict(X_test)
    error1 = mean_squared_error(Y_test, pred1)
    print ("linReg with estimated clusters: mse = %f, run in %fs" % (error1, time() - t0))

    # linReg with given clusters
    t0 = time()
    reg2 = ClusteredLinearRegression(r, epsilon_m, epsilon_w, epsilon_b, E, mu=2.5)
    reg2.fit(X_train, Y_train)
    pred2 = reg2.predict(X_test)
    error2 = mean_squared_error(Y_test, pred2)
    print ("linReg with true given clusters: mse = %f, run in %fs" % (error2, time() - t0))

    # linReg with no clusters
    t0 = time()
    reg3 = ClusteredLinearRegression(r, epsilon_m, epsilon_w, epsilon_b, E, mu=0)
    reg3.fit(X_train, Y_train)
    pred3 = reg3.predict(X_test)
    error3 = mean_squared_error(Y_test, pred3)
    print ("linReg with no clusters: mse = %f, run in %fs" % (error3, time() - t0))

    # Scikit-Learn linReg ref
    t0 = time()
    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    pred = reg.predict(X_test)
    error = mean_squared_error(Y_test, pred)
    print ("Scikit-Learn linReg ref: mse = %f, run in %fs" % (error, time() - t0))
