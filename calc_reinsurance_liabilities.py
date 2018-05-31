# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:04:54 2018

@author: aak228
"""

import networkx as nx
import numpy as np
from collections import deque
from scipy import sparse, optimize
import sys
import warnings
warnings.filterwarnings("error")

###############################################################################
# Define data structures

class FinancialSystem(object):
    ''' Attributes:
    n = number of firms
    m = number of contracts
    G = nxn nx.DiGraph directed from reinsurer to reinsured firm with data on Gamma, D, and C
    Gamma = nxn sparse graph of coinsurance rates on contracts, directed from reinsurer to reinsured firm
    D = nxn sparse graph of deductibles on contracts, directed from reinsurer to reinsured firm
    C = nxn sparse graph of payout caps (i.e., limits) on contracts, directed from reinsurer to reinsured firm
    sh = nx1 vector of shocks to firms
    Gprems = graph on premiums (edges are reinsured -> reinsurer)
    prlb = lower bound on premiums ceded/premiums received for calculating primary premiums
    prub = upper bound ...
    rrlb = lower bound on premiums ceded/premiums recieved for retro reinsurance
    rrub = upper bound ...
    state2cos = dictionary mapping states to companies
    prems_prim = dictionary of primary premiums; if none provided, will calculate randomly
    prems_f_reins = dictionary of reinsurance premiums received from foreign companies; if none provided, will calculate randomly
    capital = dictionary of capital each node can pay out; if none provided, will calculate randomly
    lrlb = leverage ratio lower bound for determining random capital
    lrub = leverage ratio upper bound ...
    '''
    
    def __init__(self, Gprems, G, state2cos, prlb=0.05, prub=0.5, rrlb=0.1, rrub=0.3, lrlb=0.7, lrub=2.0, prems_prim=None, prems_f_reins=None, capital=None):
        self.Gprems = Gprems
        self.prlb = prlb
        self.prub = prub
        self.rrlb = rrlb
        self.rrub = rrub
        self.lrlb = lrlb
        self.lrub = lrub
        self.prems_prim = {}
        self.prems_f_reins = {}
        self.capital = {}
        self.G = G
        self.n = len(G.nodes())
        self.m = len(G.edges())
        self.Gamma = sparse.lil_matrix(nx.adjacency_matrix(G,weight='gamma'))
        self.D = sparse.lil_matrix(nx.adjacency_matrix(G,weight='ded'))
        self.C = sparse.lil_matrix(nx.adjacency_matrix(G,weight='cap'))
        self.sh = np.zeros(self.n)
        self.GT = nx.reverse(G,copy=True) #transpose graph
        self.node2ind = node2ind_dict(G)
        self.state2cos = state2cos
        
        
        #line graph-related variables
        self.H = nx.line_graph(G) #line graph
        self.node2ind_H = node2ind_dict(self.H)
        self.X = nx.adjacency_matrix(self.H)
        self.d = np.array([self.D[self.node2ind[elem[0]],self.node2ind[elem[1]]] for elem in self.H.nodes()])
        self.c = np.array([self.C[self.node2ind[elem[0]],self.node2ind[elem[1]]] for elem in self.H.nodes()])
        self.gamma = sparse.diags([self.Gamma[self.node2ind[elem[0]],self.node2ind[elem[1]]] for elem in self.H.nodes()])
        self.s = np.zeros(self.m)
        assert np.min(self.d) >= 0 #deductibles must non-negative
        assert np.min(self.c) > 0 #caps must be positive
        
        if prems_prim == None:
            self.set_rand_primary_prems(prlb, prub)
        else:
            self.prems_prim = prems_prim
        
        if prems_f_reins == None:
            self.set_rand_foreign_reins_prems(rrlb, rrub)
        else:
            self.prems_f_reins = prems_f_reins
        
        if capital == None:
            self.set_rand_node_capital(lrlb, lrub)
        else:
            self.capital = capital
    
    def rand_primary_prems(self, prlb, prub):
        '''output random primary insurance premiums for each firm consistent with data
        G is graph on reinsurance premiums ceded
        ub = upper bound on premiums ceded / premiums received, lb = lower bound'''
        assert 0 <= prlb and prlb <= prub
        prems_reins = dict(self.Gprems.in_degree(weight='weight')) #reinsurance premiums received
        prems_ceded = dict(self.Gprems.out_degree(weight='weight')) #reinsurance premiums ceded
        prems_prim = {} #primary insurance premiums received
        for node in self.Gprems.nodes():
            #if prems_reins[node] == 0 or prems_ceded[node]/prems_reins[node] > self.prub:
            if prems_reins[node] == 0: #reinsurance companies don't have primary lines
                ratio = np.random.uniform(prlb, prub)
                prems_prim[node] = prems_ceded[node]/ratio - prems_reins[node]
                assert prems_prim[node] >= 0
            else:
                prems_prim[node] = 0
        return prems_prim
    
    def rand_foreign_reins_prems(self, rrlb, rrub):
        '''output random foreign reinsurance premiums for each firm consistent with data
        G is graph on reinsurance premiums ceded
        ub = upper bound on premiums ceded / premiums received, lb = lower bound'''
        assert 0 <= rrlb and rrlb <= rrub
        prems_reins = dict(self.Gprems.in_degree(weight='weight')) #reinsurance premiums received
        prems_ceded = dict(self.Gprems.out_degree(weight='weight')) #reinsurance premiums ceded
        prems_f_reins = {} #foreign reinsurance premiums received
        for node in self.Gprems.nodes():
            if prems_reins[node] != 0 and prems_ceded[node]/prems_reins[node] > self.rrub:
                ratio = np.random.uniform(rrlb, rrub)
                prems_f_reins[node] = prems_ceded[node]/ratio - prems_reins[node]
                assert prems_f_reins[node] >= 0
            else:
                prems_f_reins[node] = 0
        return prems_f_reins
    
    def rand_node_capital(self, lrlb, lrub):
        '''calculates random node capital available for payout based on random leverage ratios = equity/net premiums
        lrlb = lower bound on leverage ratio, lrub = upper bound'''
        assert 0 <= lrlb and lrlb <= lrub
        prems_reins = self.Gprems.in_degree(weight='weight')
        prems_ceded = self.Gprems.out_degree(weight='weight')
        
        capital = {}
        for node in self.Gprems.nodes():
            rlr = np.random.uniform(lrlb,lrub) #random leverage ratio
            prems_net = prems_reins[node] + self.prems_f_reins[node] + self.prems_prim[node] - prems_ceded[node]
            capital[node] = rlr*prems_net
            assert capital[node] >= 0
        return capital
    
    def set_rand_primary_prems(self, prlb=None, prub=None):
        if prlb != None:
            self.prlb = prlb
        if prub != None:
            self.prub = prub
        self.prems_prim = self.rand_primary_prems(self.prlb, self.prub)
        return
    
    def set_rand_foreign_reins_prems(self, rrlb=None, rrub=None):
        if rrlb != None:
            self.rrlb = rrlb
        if rrub != None:
            self.rrub = rrub
        self.prems_f_reins = self.rand_foreign_reins_prems(self.rrlb, self.rrub)
        return
    
    def set_rand_node_capital(self, lrlb=None, lrub=None):
        if lrlb != None:
            self.lrlb = lrlb
        if lrub != None:
            self.lrub = lrub
        self.capital = self.rand_node_capital(self.lrlb, self.lrub)
        return
    
    def set_shock(self, sh):
        self.sh = sh
        self.s = np.array([sh[self.node2ind[elem[1]]] for elem in self.H.nodes()])
        return
        

###############################################################################
# Dense matrix solvers

def solve_no_caps(G,D,sh):
    '''Returns liabilities given shock sh in a system with deductibles D but no caps
    to guarantee convergence:
        (I-gamma*X) must be invertible, where X is line graph of G, gamma is diagonal matrix of reinsurance rates per edge'''
    node2ind = node2ind_dict(G)
    H = nx.line_graph(G)
    X = nx.adjacency_matrix(H).toarray()
    
    #set up d, gamma, s indexed appropriately for H
    d = np.array([D[node2ind[elem[0]],node2ind[elem[1]]] for elem in H.nodes()])
    assert np.min(d) >= 0 #deductibles must non-negative
    Gamma = nx.adjacency_matrix(G).toarray()
    gamma = np.diag([Gamma[node2ind[elem[0]],node2ind[elem[1]]] for elem in H.nodes()])
    s = np.array([sh[node2ind[elem[1]]] for elem in H.nodes()])
    
    #iterative solver
    m = len(H.nodes())
    check = s-d
    b_0 = [1 if elem >=0 else 0 for elem in check]
    finish = False
    while finish == False:
        ell = np.linalg.solve((np.eye(m)-np.dot(gamma,np.dot(np.diag(b_0),X))),
                                np.dot(gamma, np.dot(np.diag(b_0), s-d)))
        check = s + np.dot(X,ell) - d
        b_1 = [1 if elem >=0 else 0 for elem in check]
        if b_0 == b_1:
            finish = True
        else:
            b_0 = b_1[:]
    
    #translate ell to L matrix on nodes:
    n = len(G.nodes())
    node2ind_H = node2ind_dict(H)
    L = np.zeros((n,n))
    for elem in H.nodes():
        L[elem[0],elem[1]] = ell[node2ind_H[elem]]
    return L

def solve_caps(G,D,C,sh):
    '''Returns liabilities given shock sh in a system with deductibles D and caps C
    to guarantee convergence:
        (I-gamma*X) must be invertible, where X is line graph of G, gamma is diagonal matrix of reinsurance rates per edge'''
    node2ind = node2ind_dict(G)
    H = nx.line_graph(G)
    X = nx.adjacency_matrix(H).toarray()
    
    #set up d, c, gamma, s indexed appropriately for H
    d = np.array([D[node2ind[elem[0]],node2ind[elem[1]]] for elem in H.nodes()])
    c = np.array([C[node2ind[elem[0]],node2ind[elem[1]]] for elem in H.nodes()])
    assert np.min(d) >= 0 #deductibles must non-negative
    assert np.min(c) > 0 #caps must be positive
    Gamma = nx.adjacency_matrix(G).toarray()
    gamma = np.diag([Gamma[node2ind[elem[0]],node2ind[elem[1]]] for elem in H.nodes()])
    s = np.array([sh[node2ind[elem[1]]] for elem in H.nodes()])
    
    #iterative solver
    m = len(H.nodes())
    check = s-d
    B_0 = [1 if elem >=0 else 0 for elem in check]
    check = np.dot(gamma,check) - c
    C_0 = [1 if elem >= 0 else 0 for elem in check]
    finish = False
    while finish == False:
        ellbar = [c[i] if C_0[i] == 1 else 0 for i in range(m)]
        Psi, psi = map_nnz_dim(np.ones(m) - np.array(C_0))
        Psi = Psi.toarray()
        tm = len(psi.keys())
        tgamma = np.dot(Psi, np.dot(gamma, np.transpose(Psi)))
        tB = np.dot(Psi, np.dot(np.diag(B_0), np.transpose(Psi)))
        tX = np.dot(Psi, np.dot(X, np.transpose(Psi)))
        tv = np.dot(Psi, s + np.dot(X, ellbar) - d)
        
        tell = np.linalg.solve((np.eye(tm)-np.dot(tgamma,np.dot(tB,tX))),
                                np.dot(tgamma, np.dot(tB, tv)))
        check = np.dot(tX,tell) + tv
        B_1 = [1 if B_0[i] == 1 else 1 if check[psi[i]]>= 0 else 0 for i in range(m)]
        check = np.dot(tgamma,check) - np.dot(Psi, c)
        C_1 = [1 if C_0[i] == 1 else 1 if check[psi[i]]>= 0 else 0 for i in range(m)]
        if B_1 == B_0 and C_1 == C_0:
            ell = np.dot(np.transpose(Psi),tell) + ellbar
            finish = True
        else:
            B_0 = B_1[:]
            C_0 = C_1[:]
    
    #translate ell to L matrix on nodes:
    n = len(G.nodes())
    node2ind_H = node2ind_dict(H)
    L = np.zeros((n,n))
    for elem in H.nodes():
        L[elem[0],elem[1]] = ell[node2ind_H[elem]]
    return L

###############################################################################
# Sparse matrix solvers

def solve_no_caps_sp(fs, illcond=2e3):
    '''Returns liabilities given shock sh in a system with deductibles D but no caps, uses sparse matrix methods
    illcond is condition number threshold for linear solver
    to guarantee convergence:
    (I-gamma*X) must be invertible, where X is line graph of G, gamma is diagonal matrix of reinsurance rates per edge'''
    #iterative solver
    check = fs.s-fs.d
    b_0 = [1 if elem >=0 else 0 for elem in check]
    finish = False
    while finish == False:
        B = sparse.diags(b_0)
        
        #check condition number
        M = sparse.eye(fs.m) - fs.gamma*B*fs.X
        cond_num = sp_cond_num(M)
        if cond_num > illcond: #1/sys.float_info.epsilon:
            raise Exception('Ill-conditioned matrix')
        
        ell = sparse.linalg.spsolve(M, fs.gamma*B*(fs.s-fs.d))
        check = fs.s + fs.X*ell - fs.d
        b_1 = [1 if elem >=0 else 0 for elem in check]
        if b_0 == b_1:
            finish = True
        else:
            b_0 = b_1[:]
    
    #translate ell to L matrix on nodes:
    return ell2L(ell,fs), b_1

def solve_caps_sp(fs, illcond=2e3):
    '''Returns liabilities given shock sh in a system with deductibles D and caps C; illcond is condition number threshold for linear solver
    to guarantee convergence:
    (I-gamma*X) must be invertible, where X is line graph of G, gamma is diagonal matrix of reinsurance rates per edge'''
    #iterative solver
    check = fs.s-fs.d
    B_0 = [1 if elem >=0 else 0 for elem in check]
    check = fs.gamma*check - fs.c
    C_0 = [1 if elem >= 0 else 0 for elem in check]
    finish = False
    while finish == False:
        ell, B_1, C_1 = iterate_capped_lsystem(fs, B_0, C_0, illcond=illcond)
        if B_1 == B_0 and C_1 == C_0:
            finish = True
        else:
            B_0 = B_1[:]
            C_0 = C_1[:]
    #translate ell to L matrix on nodes:
    return ell2L(ell,fs), B_1, C_1

def iterate_capped_lsystem(fs, B_0, C_0, last_ell=None, illcond=2e3):
    '''performs one lsystem update with caps; illcond is condition number threshold for linear solver'''
    ellbar = [fs.c[i] if C_0[i] == 1 else 0 for i in range(fs.m)]
    Psi, psi = map_nnz_dim(np.ones(fs.m) - np.array(C_0))
    Psi_T = sparse.csr_matrix(Psi.T)
    Psi = sparse.csr_matrix(Psi)
    tm = len(psi.keys())
    tgamma = Psi*fs.gamma*Psi_T
    tB = Psi*sparse.diags(B_0)*Psi_T
    tX = Psi*fs.X*Psi_T
    tv = Psi*(fs.s + fs.X*ellbar - fs.d)
    
    #check condition number
    M = sparse.eye(tm)-tgamma*tB*tX
    cond_num = sp_cond_num(M)
    print('cond_num: {}'.format(cond_num))
    if cond_num > illcond: #1/sys.float_info.epsilon:
        raise Exception('Ill-conditioned matrix: {}>{}'.format(cond_num,illcond))
    
    tell = sparse.linalg.spsolve(M, tgamma*tB*tv)

    check = tX*tell + tv
    B_1 = [1 if B_0[i] == 1 else 1 if check[psi[i]]>= 0 else 0 for i in range(fs.m)]
    check = tgamma*check - Psi*fs.c
    C_1 = [1 if C_0[i] == 1 else 1 if check[psi[i]]>= 0 else 0 for i in range(fs.m)]
    if C_1 == C_0: #ell is overestimate of some values if C_1 != C[0]
        ell = Psi_T*tell + ellbar
    else:
        ell = last_ell
    return ell, B_1, C_1

def solve_caps_combo(fs, illcond=2e3):
    '''Solves for liabilities using mix of lsystem update when applicable, otherwise push updates
    illcond is condition number threshold for linear solver
    guaranteed to converge (in possibly exponential time) given finite caps, may not otherwise'''
    L_1 = sparse.dok_matrix((fs.n,fs.n))
    last_ell = np.zeros(fs.m)
    check = fs.s-fs.d
    B_0 = [1 if elem >=0 else 0 for elem in check]
    check = fs.gamma*check - fs.c
    C_0 = [1 if elem >= 0 else 0 for elem in check]
    finish = False
    inv_error = False
    inv_error_last = False
    while finish == False:
        if inv_error == False:
            try:
                ell, B_1, C_1 = iterate_capped_lsystem(fs, B_0, C_0, last_ell, illcond=illcond)
                last_ell = ell[:]
                print('ell={}'.format(np.sum(ell)))
                inv_error_last = False
                if B_1 == B_0 and C_1 == C_0:
                    finish = True
                    return ell2L(ell,fs), B_1, C_1
                else:
                    B_0 = B_1[:]
                    C_0 = C_1[:]
            except:
                inv_error = True
        else:
            if inv_error_last == False:
                L_0 = ell2L(last_ell,fs)
                inv_error_last = True
            else:
                L_0 = L_1[:]
            L_1, B_1, C_1 = push(fs,L_0)
            last_ell = L2ell(L_1,fs)
            print('L_1={}'.format(np.sum(L_1)))
            if B_1 != B_0 or C_1 != C_0:
                inv_error = False
                B_0 = B_1[:]
                C_0 = C_1[:]
            elif (L_0 - L_1).nnz == 0: #might want to replace with a convergence threshold
                finish = True
                return L_1, B_1, C_1
    
        
###############################################################################
#solve linear systems via fixed point iterations

def solve_caps_sp_fp(fs):
    '''Returns liabilities given shock sh in a system with deductibles D and caps C; illcond is condition number threshold for linear solver
    to guarantee convergence:
    (I-gamma*X) must be invertible, where X is line graph of G, gamma is diagonal matrix of reinsurance rates per edge'''
    #iterative solver
    ell = np.zeros(fs.m)
    check = fs.s-fs.d
    B_0 = [1 if elem >=0 else 0 for elem in check]
    check = fs.gamma*check - fs.c
    C_0 = [1 if elem >= 0 else 0 for elem in check]
    finish = False
    while finish == False:
        print(np.sum(ell))
        ell, B_1, C_1 = iterate_capped_fp(fs, B_0, C_0, ell)
        if B_1 == B_0 and C_1 == C_0:
            finish = True
        else:
            B_0 = B_1[:]
            C_0 = C_1[:]
    #translate ell to L matrix on nodes:
    return ell2L(ell,fs), B_1, C_1

def iterate_capped_fp(fs, B_0, C_0, last_ell):
    '''performs one fixed point update with caps'''
    ellbar = [fs.c[i] if C_0[i] == 1 else 0 for i in range(fs.m)]
    Psi, psi = map_nnz_dim(np.ones(fs.m) - np.array(C_0))
    Psi_T = sparse.csr_matrix(Psi.T)
    Psi = sparse.csr_matrix(Psi)
    tgamma = Psi*fs.gamma*Psi_T
    tB = Psi*sparse.diags(B_0)*Psi_T
    tX = Psi*fs.X*Psi_T
    tv = Psi*(fs.s + fs.X*ellbar - fs.d)
    last_tell = Psi*last_ell
    
    def fun(tell, tB,tX,tgamma,tv):
        return tgamma*tB*(tX*tell + tv)
    
    tell = optimize.fixed_point(fun, last_tell, args=(tB,tX,tgamma,tv), maxiter=2000)

    check = tX*tell + tv
    B_1 = [1 if B_0[i] == 1 else 1 if check[psi[i]]>= 0 else 0 for i in range(fs.m)]
    check = tgamma*check - Psi*fs.c
    C_1 = [1 if C_0[i] == 1 else 1 if check[psi[i]]>= 0 else 0 for i in range(fs.m)]
    '''
    if C_1 == C_0: #ell is overestimate of some values if C_1 != C[0]
        ell = Psi_T*tell + ellbar
    else:
        ell = last_ell
    '''
    ell = Psi_T*tell + ellbar
    return ell, B_1, C_1



###############################################################################
# Solve backward iterations

def solve_caps_back(fs, illcond=2e3):
    '''Returns liabilities given shock sh in a system with deductibles D and caps C; illcond is condition number threshold for linear solver
    to guarantee convergence:
    (I-gamma*X) must be invertible, where X is line graph of G, gamma is diagonal matrix of reinsurance rates per edge'''
    #iterative solver
    check = fs.X*fs.c + fs.s - fs.d
    B_0 = [1 if elem >=0 else 0 for elem in check]
    check = fs.gamma*check - fs.c
    C_0 = [1 if elem >= 0 else 0 for elem in check]
    finish = False
    while finish == False:
        ell, B_1, C_1 = iterate_capped_lsystem_back(fs, B_0, C_0, illcond=illcond)
        if B_1 == B_0 and C_1 == C_0:
            finish = True
        else:
            B_0 = B_1[:]
            C_0 = C_1[:]
    #translate ell to L matrix on nodes:
    return ell2L(ell,fs), B_1, C_1

def iterate_capped_lsystem_back(fs, B_0, C_0, illcond=2e3):
    '''performs one lsystem update with caps; illcond is condition number threshold for linear solver'''
    ellbar = [fs.c[i] if C_0[i] == 1 else 0 for i in range(fs.m)]
    Psi, psi = map_nnz_dim(np.ones(fs.m) - np.array(C_0))
    Psi_T = sparse.csr_matrix(Psi.T)
    Psi = sparse.csr_matrix(Psi)
    tm = len(psi.keys())
    tgamma = Psi*fs.gamma*Psi_T
    tB = Psi*sparse.diags(B_0)*Psi_T
    tX = Psi*fs.X*Psi_T
    tv = Psi*(fs.s + fs.X*ellbar - fs.d)
    
    #check condition number
    M = sparse.eye(tm)-tgamma*tB*tX
    cond_num = sp_cond_num(M)
    print('cond_num: {}'.format(cond_num))
    if cond_num > illcond: #1/sys.float_info.epsilon:
        print('cond_num too high!')
        raise Exception('Ill-conditioned matrix: {}>{}'.format(cond_num,illcond))
    
    tell = sparse.linalg.spsolve(M, tgamma*tB*tv)
    
    ell = Psi_T*tell + ellbar
    check = fs.X*ell + fs.s - fs.d
    B_1 = [1 if elem >=0 else 0 for elem in check]
    check = fs.gamma*check - fs.c
    C_1 = [1 if elem >= 0 else 0 for elem in check]

    return ell, B_1, C_1

def solve_caps_combo_back(fs, illcond=2e3):
    '''Solves for liabilities using mix of lsystem update when applicable, otherwise push updates
    illcond is condition number threshold for linear solver
    guaranteed to converge (in possibly exponential time) given finite caps, may not otherwise'''
    ell_0 = fs.c
    check = fs.X*fs.c + fs.s - fs.d
    B_0 = [1 if elem >=0 else 0 for elem in check]
    check = fs.gamma*check - fs.c
    C_0 = [1 if elem >= 0 else 0 for elem in check]
    finish = False
    inv_error = False
    while finish == False:
        if inv_error == False:
            try:
                ell_1, B_1, C_1 = iterate_capped_lsystem_back(fs, B_0, C_0, illcond=illcond)
                ell_0 = ell_1[:]
                print('ell_1={}'.format(np.sum(ell_1)))
                if B_1 == B_0 and C_1 == C_0:
                    finish = True
                else:
                    B_0 = B_1[:]
                    C_0 = C_1[:]
            except:
                inv_error = True
        else:
            ell_1, B_1, C_1 = push_mat(fs,ell_0)
            if (ell_0 == ell_1).all():
                finish = True
            elif B_1 != B_0 or C_1 != C_0:
                inv_error = False
                B_0 = B_1[:]
                C_0 = C_1[:]
            ell_0 = ell_1[:]
            print('ell_1={}'.format(np.sum(ell_1)))
    
    return ell2L(ell_1,fs), B_1, C_1





###############################################################################
# Step-push solver = fixed point iteration

def push_mat(fs,ell_0):
    check = fs.X*ell_0 + fs.s - fs.d
    B_1 = [1 if elem >=0 else 0 for elem in check]
    ell = fs.gamma*check
    ell = np.array([ell[i] if B_1[i]==1 else 0 for i in range(fs.m)])
    check = ell - fs.c
    C_1 = [1 if elem >= 0 else 0 for elem in check]
    ell = np.minimum(ell, fs.c)
    print(np.sum(ell))
    return ell, B_1, C_1

def solve_push_mat(fs, typ):
    '''Performs a fixed point iteration from 0 if typ='forward', from cap values if typ='back'
    guaranteed to converge (in possibly arbitrary time) given finite caps, may not otherwise'''
    assert typ in ['forward','back']
    if typ=='forward':
        ell_0 = np.zeros(fs.m)
    elif typ=='back':
        ell_0 = fs.c
    ell_1, B_1, C_1 = push_mat(fs,ell_0)
    while (ell_0 != ell_1).any():
        ell_0 = ell_1[:]
        ell_1, B_1, C_1 = push_mat(fs,ell_0)
    return ell2L(ell_1, fs), B_1, C_1
    

def solve_push(fs):
    '''Solves for liablities using push updates (not solving linear systems), equivalent to fixed point iteration
    guaranteed to converge (in possibly arbitrary time) given finite caps, may not otherwise'''
    L_0 = sparse.dok_matrix((fs.n,fs.n))
    L_1, B_1, C_1 = push(fs,L_0)
    while (L_0 - L_1).nnz > 0: #might want to replace with a convergence threshold
        L_0 = L_1[:]
        L_1, B_1, C_1 = push(fs,L_0)
    return L_1, B_1, C_1

def push(fs,L):
    '''performs one push update step'''
    L_1 = L[:]
    L1 = L.tocsr()*np.ones(fs.n) + fs.sh #calculate aggregate liabilities per node
    B_1 = [0 for i in range(fs.m)] #record if edge deductibles met
    C_1 = [0 for i in range(fs.m)] #record if edge caps met
    #candidate edges for push update are edges with head node nonzero in L1+sh
    for node1 in [elem for elem in fs.G.nodes() if L1[fs.node2ind[elem]]>0]:
        i = fs.node2ind[node1]
        for node2 in list(fs.GT[node1]): #neighbors in transpose graph
            j = fs.node2ind[node2]
            if L1[i] >= fs.D[j,i]:
                B_1[fs.node2ind_H[(node2,node1)]] = 1
                if fs.Gamma[j,i]*(L1[i]-fs.D[j,i]) >= fs.C[j,i]:
                    C_1[fs.node2ind_H[(node2,node1)]] = 1
                    L_1[j,i] = fs.C[j,i]
                else:
                    L_1[j,i] = fs.Gamma[j,i]*(L1[i]-fs.D[j,i])
    return L_1, B_1, C_1

###############################################################################
# Referenced functions

def node2ind_dict(G):
    '''returns dictionary mapping node to index number in the graph'''
    nodes = list(G.nodes())
    node2ind = {}
    for i in range(len(nodes)):
        node2ind[nodes[i]] = i
    return node2ind

def ind2node_dict(G):
    nodes = list(G.nodes())
    ind2node = {}
    for i in range(len(nodes)):
        ind2node[i] = nodes[i]
    return ind2node

def multi2digraph(G):
    '''converts multidigraph to digraph summing parallel edges'''
    ind2node = ind2node_dict(G)
    A = nx.adjacency_matrix(G) #treats parallel edges as sum
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(True))
    arrs = sparse.find(A)
    for ind in range(A.count_nonzero()):
        i = arrs[0][ind]
        j = arrs[1][ind]
        v = arrs[2][ind]
        H.add_edge(ind2node[i],ind2node[j], weight = v)
    return H

def cycle_edges(G,Cap,cycle,node2ind):
    '''given cycle from simple_cycles generator, returns edges in cycle
    the third element in an edge tuple represents the edge weight'''
    A = nx.adjacency_matrix(G)
    start = node2ind[cycle[-1]]
    edges_g = []
    edges_c = []
    for elem in cycle:
        end = node2ind[elem]
        edges_g.append((start, end, A[start,end]))
        edges_c.append((start, end, Cap[start,end]))
        start = end
    return edges_g, edges_c

def identify_cycle_problem(G,Cap,cycle,node2ind):
    '''checks if cycle has 100% reinsurance rate all around with no caps on reinsurance payout'''
    edges_g, edges_c = cycle_edges(G,Cap,cycle,node2ind)
    gs = [edges_g[i][2] for i in range(len(edges_g))]
    cs = [edges_c[i][2] for i in range(len(edges_c))]
    cycle_problem = 1
    for i in range(len(gs)):
        if gs[i] < 1 or np.isfinite(cs[i]):
            cycle_problem = 0
    if cycle_problem == 1:
        return True
    else:
        return False

def map_nnz_dim(v):
    '''returns mapping matrix to the nonzero dimensions of input vector v
    maintains order of coordinates in v, transpose is the reverse mapping'''
    nnz_inds = deque(np.nonzero(v)[0])
    n = len(v)
    m = len(nnz_inds)
    M = sparse.dok_matrix((m,n))
    nnz_ind_map = {}
    for i in range(m):
        j = nnz_inds.popleft()
        M[i,j] = 1
        nnz_ind_map[j] = i
    return M, nnz_ind_map

def ell2L(ell,fs):
    '''translate ell vector on edges to L matrix on nodes'''
    L = sparse.dok_matrix((fs.n,fs.n))
    for elem in fs.H.nodes():
        L[fs.node2ind[elem[0]],fs.node2ind[elem[1]]] = ell[fs.node2ind_H[elem]]
    return L

def L2ell(L,fs):
    ell = np.zeros(fs.m)
    node2ind = node2ind_dict(fs.G)
    node2ind_H = node2ind_dict(fs.H)
    for edge in fs.H.nodes():
        ell[node2ind_H[edge]] = L[node2ind[edge[0]],node2ind[edge[1]]]
    return ell

def sp_cond_num(M):
    '''Estimate the condition number of a sparse matrix M; note: M^* M may not be sparse, but tends to be in most use cases
    If M^* M not sparse, then tradeoff between using eigsh(M^* M) and eigs of M, M^* (assuming M normal)'''
    A = M.conj().transpose()*M
    eig_large = sparse.linalg.eigsh(A, k=1, which='LM', tol=1e-3, return_eigenvectors=False)
    eig_small = sparse.linalg.eigsh(A, k=1, sigma=0, which='LM', tol=1e-3, return_eigenvectors=False)
    if np.abs(eig_small[0]) > 0:
        cond_num = np.sqrt(np.abs(eig_large[0]))/np.sqrt(np.abs(eig_small[0]))
    else:
        cond_num = np.inf
    return cond_num


