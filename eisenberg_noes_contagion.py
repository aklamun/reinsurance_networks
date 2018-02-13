# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:17:06 2018

Implements Eisenberg-Noe contagion clearing
Refer to their paper for notation

@author: aak228
"""

import numpy as np
from scipy import optimize

#inputs: e cash flow vector, L nominal liabilities matrix

def calc_p_bar(L):
    #sum rows of L
    p_bar = np.sum(L,axis=1)
    return p_bar

def calc_Pi(L):
    p_bar = calc_p_bar(L)
    Pi = np.zeros((len(p_bar),len(p_bar)))
    for i in range(len(p_bar)):
        if p_bar[i] != 0:
            Pi[i,:] = L[i,:]/p_bar[i]
    return Pi

def Phi(p, Pi, p_bar, e):
    return np.minimum( np.dot(np.transpose(Pi),p) + e, p_bar )

def next_default_ind(p, Pi, p_bar, e):
    pp = Phi(p, Pi, p_bar, e)
    D = np.zeros(len(p_bar))
    for i in range(len(p_bar)):
        if p_bar[i] > pp[i]:
            D[i] = 1
    return D

def FF(p, p_0,Pi,Lambda,e,p_bar,I):
    return np.dot(Lambda,  np.dot(Pi, (np.dot(Lambda,p) + np.dot(I-Lambda,p_bar)) )  + e  ) + np.dot(I-Lambda,p_bar)

def next_p(p_0, D_0, Pi, e, p_bar, I):
    Lambda = np.diag(D_0)
    
    #p_1 = f(p_0) solve fixed point
    p_1 = optimize.fixed_point(FF, p_0, args=(p_0,Pi,Lambda,e,p_bar,I))
    D_1 = next_default_ind(p_1, Pi, p_bar, e)
    return p_1, D_1

def clearing_p(L,e):
    #This implements the iterative solver
    p_bar = calc_p_bar(L)
    Pi = calc_Pi(L)
    p_0 = p_bar[:]
    D_0 = next_default_ind(p_0, Pi, p_bar, e)
    
    I = np.eye(len(p_bar))
    p_1, D_1 = next_p(p_0, D_0, Pi, e, p_bar, I)
    
    while list(D_1) != list(D_0):
        D_0 = D_1[:]
        p_0 = p_1[:]
        p_1, D_1 = next_p(p_0, D_0, Pi, e, p_bar, I)
    
    return p_1, D_1


'''=========================================================================
LP formulation

minimize:     -1^T * x

subject to:   (I-Pi^T)p <= e
encoded as     A_ub * x <= b_ub

'''


def clearing_p_LP(L,e):
    #This implements a LP solver
    p_bar = calc_p_bar(L)
    I = np.eye(len(p_bar))
    Pi = calc_Pi(L)
    obj_coeff = -np.ones(len(e))
    A = (I-np.transpose(Pi))
    bnds = [(0,pb) for pb in p_bar]

    res = optimize.linprog(obj_coeff, A_ub=A, b_ub=e, bounds=bnds)
    p_lp = res.x
    D_lp = next_default_ind(p_lp, Pi, p_bar, e)
    return p_lp, D_lp


#Run an example
L = np.array([[0,1],[2,0]])
e = np.array([1,0])
p_1, D_1 = clearing_p(L,e)
p_lp, D_lp = clearing_p_LP(L,e)


