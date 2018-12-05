# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:59:38 2018

@author: aak228
"""

import networkx as nx
import numpy as np
from scipy import sparse, optimize
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from calc_reinsurance_liabilities import *
import eisenberg_noes_contagion_sparse as eis_noe


#set directory as appropriate
os.chdir('...')


###############################################################################
#functions related to shock calculation

def get_co_state_data(G, year):
    '''get data on domicile state for companies in G'''
    if int(year) not in [2012, 2013, 2014, 2015, 2016]:
        raise Exception("No data for year {}".format(year))
        
    #Get company data
    df_co = pd.read_excel('reinsurance_graph/P{}000.xlsx'.format(year),sheetname='P{}000'.format(year),header=None)
    if year in [2012, 2013, 2014]:
        df_co.columns=['COCODE','SHORT_COMPANY_NAME','FULL_COMPANY_NAME','SURVIVING_COCODE','BUSINESS_TYPE','BUSINESS_TYPE_DESC','BUSINESS_SUB_TYPE','BUSINESS_SUB_TYPE_DESC','FILING_TYPE','FILING_TYPE_DESC','COMPANY_TYPE','COMPANY_TYPE_DESC','COMPANY_SUB_TYPE','COMPANY_SUB_TYPE_DESC','FEIN','STATE_DOMICILE','COMM_BUS_DATE','GROUP_CODE','GROUP_NAME','GROUP_CODE_PRIOR_PERIOD','GROUP_NAME_PRIOR_PERIOD','COMPANY_STATUS','COMPANY_STATUS_DESC','COUNTRY_NAME']
    elif year in [2015, 2016]:
        df_co.columns=['COCODE','SHORT_COMPANY_NAME','FULL_COMPANY_NAME','SURVIVING_COCODE','BUSINESS_TYPE','BUSINESS_TYPE_DESC','BUSINESS_SUB_TYPE','BUSINESS_SUB_TYPE_DESC','FILING_TYPE','FILING_TYPE_DESC','COMPANY_TYPE','COMPANY_TYPE_DESC','COMPANY_SUB_TYPE','COMPANY_SUB_TYPE_DESC','FEIN','STATE_DOMICILE','COMM_BUS_DATE','GROUP_CODE','GROUP_NAME','GROUP_CODE_PRIOR_PERIOD','GROUP_NAME_PRIOR_PERIOD','COMPANY_STATUS','COMPANY_STATUS_DESC','COUNTRY_NAME','DOMESTIC_CRIN','CERTIFIED_REINSURER']
    df_co.index=df_co['COCODE']
    
    nodes = [int(node) for node in list(G.nodes())]
    df_co = df_co.loc[df_co['COCODE'].isin(nodes)]
    return df_co[['COCODE','SHORT_COMPANY_NAME','STATE_DOMICILE']]

def state2cos_dict(df_co):
    '''returns dictionary mapping states to company lists'''
    states = list(set(df_co['STATE_DOMICILE']))
    states.remove(' ')
    state2cos = {}
    for st in states:
        cos = df_co.loc[df_co['STATE_DOMICILE'] == st]
        cos = list(cos['COCODE'])
        cos = [str(co) for co in cos]
        state2cos[st] = cos
    return state2cos

def rand_agg_shock(typ=None):
    '''calculate random aggregate shock; calibrated using data on tail risk and AAL
    typ in ['avg', 100, 250] or None to specify avg shock, 1-in-100, or 1-in-250'''
    x = np.random.uniform(0,1)
    if x <= 0.004 or typ == 250:
        sag = 290.6
    elif x <= 0.014 or typ == 100:
        sag = 215.2
    elif x <= 0.507 or typ == 'avg':
        sag = 91.3712
    else:
        sag = 0
    return sag

def distribute_to_states(states, s):
    '''distribute ag uniformly randomly to states'''
    r = [0.] + list(np.sort( np.random.uniform(0,1,size=(len(states)-1)) )) + [1.]
    r = np.array(r)
    r = r[1:] - r[0:len(r)-1]
    sh2sts = {}
    for i in range(len(states)):
        sh2sts[states[i]] = r[i]*s
    return sh2sts

def distribute_to_nodes(G, sh2sts, state2cos, prems_prim):
    '''distribute state shocks to nodes proportional to primary insurance premiums'''
    nodes = list(G.nodes())
    sh = np.zeros(len(nodes))
    for st in sh2sts.keys():
        shock = sh2sts[st]
        cos = state2cos[st]
        tot_prim_st = sum([prems_prim[co] for co in cos])
        for co in cos:
            i = nodes.index(co)
            sh[i] = shock*prems_prim[co]/tot_prim_st
    return sh

#def random_shock(fs, p_l=0.2, typ=None):
    '''Calculate random shock to system, p_l is probability tail event affects a high risk state
    typ in ['avg', 100, 250] or None to specify avg shock, 1-in-100, or 1-in-250'''
    '''high_risk_states = ['TX', 'LA', 'PR', 'GA', 'NC', 'CA', 'VA', 'NY', 'VI', 'SC', 'MD', 'NJ', 'OR', 'WA', 'AL', 'FL']
    
    #eligible states for small shock (check that primary premiums > 0)
    nz_states = []
    for st in fs.state2cos.keys():
        cos = fs.state2cos[st]
        tot_prim_st = sum([fs.prems_prim[co] for co in cos])
        if tot_prim_st > 0:
            nz_states.append(st)
    
    sag = rand_agg_shock(typ=typ)
    if sag > 200: #if a tail event happens
        finish = False
        while finish == False:
            r = np.random.binomial(1, p_l, size=len(high_risk_states)) #flip coin for affected states
            if sum(r) > 0: #some states must be affected
                finish = True
        affected_states = [high_risk_states[i] for i in range(len(r)) if r[i]>0]
        sh2sts = distribute_to_states(affected_states, sag)
        sh = distribute_to_nodes(fs.Gprems, sh2sts, fs.state2cos, fs.prems_prim)
    elif sag > 0:
        sh2sts = distribute_to_states(nz_states, sag)
        sh = distribute_to_nodes(fs.Gprems, sh2sts, fs.state2cos, fs.prems_prim)
    else:
        sh = np.zeros(fs.n)
    return sh*1e6 #convert to billions, in units of thousands'''

def random_shock(fs, typ=None):
    '''Calculate random shock to system, typ in ['avg', 100, 250] or None to specify avg shock, 1-in-100, or 1-in-250
    uniformly distributes aggregate shock across primary insurers'''
    sag = rand_agg_shock(typ=typ)
    rn = np.random.uniform(0,1,fs.n)
    prems_prim = np.array([fs.prems_prim[node] for node in fs.Gprems.nodes()])
    sh = np.multiply(rn, prems_prim)
    sh = sh/np.sum(sh)
    return sh*sag*1e6 #convert to billions, in units of thousands


###############################################################################
#Auxiliary functions for creating FinancialSystem objects

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
        if i != j: #don't include self-contracts; there are some of these in the data--is this data error?
            H.add_edge(ind2node[i],ind2node[j], weight = v)
    return H

def solve_knapsack_DP(items, v, w, W):
    '''Solves 0-1 knapsack problem using dynamic programming, returns the maximum objective value and the corresponding set of items
    items is list of items, v and w are lists of corresponding values and weights, W is weight capacity; v,w, and W should have int entries
    note: this can be quite slow if n,W are large'''
    n = len(items)
    assert isinstance(W, int)
    for i in range(n):
        assert isinstance(v[i], int)
        assert isinstance(w[i], int)
    
    m = np.zeros((n+1, W+1),dtype=int)
    ml = {}
    for j in range(W+1):
        m[0,j] = 0
        ml[(0,j)] = []
    
    for i in range(1,n+1):
        for j in range(W+1):
            if w[i-1] > j:
                m[i,j] = m[i-1, j]
                ml[(i,j)] = ml[(i-1,j)]
            else:
                if m[i-1,j] > m[i-1,j-w[i-1]] + v[i-1]:
                    m[i,j] = m[i-1,j]
                    ml[(i,j)] = ml[(i-1,j)]
                else:
                    m[i,j] = m[i-1,j-w[i-1]] + v[i-1]
                    ml[(i,j)] = ml[(i-1,j-w[i-1])] + [i-1]
    
    return m[n,W], [items[i] for i in ml[(n,W)]]

def solve_knapsack_approx(items, v, w, W, eps):
    '''approximate knapsack 0-1 solution by rescaling weights, eps in (0,1)
    In general, this is not FPTAS (there is another approx that achieves that), but it is if w=v'''
    n = len(items)
    K = eps*W/n
    wp = [int(np.ceil(w[i]/K)) for i in range(n)]
    Wp = int(np.floor(W/K))
    return solve_knapsack_DP(items,v,wp,Wp)

def knapsack(edges, W):
    '''sets up and solves knapsack problem to sort edges into 2 partitions, one with weight close to W
    edge weights and W should be ints or they will be converted, uses approximate solver if n,W large'''
    n = len(edges)
    items = range(n)
    W = int(W)
    v = [int(edge[2]['weight']) for edge in edges]
    w = v[:] #set v=w in traditional 0-1 knapsack problem
    
    if n*W < 100000:
        max_v, max_it = solve_knapsack_DP(items,v,w,W)
    else:
        max_v, max_it = solve_knapsack_approx(items,v,w,W,eps=0.3)
    
    sum2 = float(max_v)
    sum1 = sum(v) - sum2
    edges2 = [edges[i] for i in max_it]
    edges1 = [edge for edge in edges if edge not in edges2]
    return edges1, edges2, sum1, sum2



###############################################################################
#Functions for creating FinancialSystem objects

def random_system_single_XLprop(Gprems, glb, gub, pcratio=0.1, dcratio=4., cdratio=5.):
    '''generate random financial system that conforms to given year's data with single layer proportional XL
    glb = lower bound on gamma, gub = upper bound on gamma
    output will have single layer of XL with proportional coinsurance
    pcratio = premiums paid / coverage limit, dcratio = coverage limit / deductible, cdratio = coverage / deductible
    note coverage = coverage limit + deductible'''
    node2ind = node2ind_dict(Gprems)
    #note: the premiums ceded graph (G above) has edges reinsured->reinsurer; we will want the transpose eventually
    prems_paid = Gprems.out_degree(weight='weight')
    #generate random number for each node representing total reinsurance rate (Gamma^T*1) except if node has no reins contracts
    GammaT1 = np.random.uniform(glb,gub,len(Gprems.nodes()))
    
    GG = nx.DiGraph()
    GG.add_nodes_from(Gprems.nodes(data=True))
    #d_dict = {}
    for edge in Gprems.edges(data=True):
        node1, node2, prem = [edge[0], edge[1], edge[2]['weight']]
        gam = prem/prems_paid[node1]*GammaT1[node2ind[node1]]
        cov = prems_paid[node1]*GammaT1[node2ind[node1]]/pcratio
        d = cov/dcratio
        #d_dict[node1] = d
        c = cdratio*cov/dcratio
        GG.add_edge(node2, node1, gamma=gam, ded=d, cap=(c-d)*gam)
    return GG

def random_system_xl(Gprems, pcratio=0.1, dcratio=4., cdratio=5., tlp=0.2):
    '''generate a financial system that conforms to given year's data with 2 layer towers of XL
    pcratio = premiums paid / coverage limit, dcratio = coverage limit / deductible, cdratio = coverage / deductible
    note coverage = coverage limit + deductible
    tlp = top layer premiums / total tower premiums'''
    #note: the premiums ceded graph (G above) has edges reinsured->reinsurer; we will want the transpose eventually
    prems_paid = Gprems.out_degree(weight='weight')
    GG = nx.DiGraph()
    GG.add_nodes_from(Gprems.nodes(data=True))
    
    #handle reinsurance node-by-node (all contracts reinsuring given node)
    for node in Gprems.nodes():
        cov = prems_paid[node]/pcratio #coverage limit
        d = cov/dcratio #deductible
        c = cdratio*cov/dcratio #total coverage cap
        edges = list(Gprems.out_edges(node,data=True))
        edges1, edges2, sum1, sum2 = knapsack(edges, tlp*prems_paid[node]) #sort into layers
        d2 = d + cov/2. #layer 2 deductible = layer 1 cap
        if len(edges2) > 0: #check if edges can constitute 2 layers, otherwise use 1
            for edge in edges1:
                #d2 is cap of this layer => contract payout caps=limits=gam*(d2-d)
                node1, node2, prem = [edge[0], edge[1], edge[2]['weight']]
                gam = prem/sum1
                GG.add_edge(node2, node1, gamma=gam, ded=d, cap=(d2-d)*gam)
            for edge in edges2:
                #d2 is cap of this layer => contract payout caps=limits=gam*(c-d2)
                node1, node2, prem = [edge[0], edge[1], edge[2]['weight']]
                gam = prem/sum2
                GG.add_edge(node2, node1, gamma=gam, ded=d2, cap=(c-d2)*gam)
        else:
            for edge in edges1:
                node1, node2, prem = [edge[0], edge[1], edge[2]['weight']]
                gam = prem/sum1
                GG.add_edge(node2, node1, gamma=prem/sum1, ded=d, cap=(c-d)*gam)
    return GG

def comparable_prop_system(Gprems, prems_prim, prems_f_reins):
    '''Calculates a proportional reinsurance system given premiums'''
    prems_reins = Gprems.in_degree(weight='weight')
    GG = nx.DiGraph()
    GG.add_nodes_from(Gprems.nodes(data=True))
    for edge in Gprems.edges(data=True):
        node1, node2, prem = [edge[0], edge[1], edge[2]['weight']]
        gam = prem/(prems_prim[node1] + prems_f_reins[node1] + prems_reins[node1])
        GG.add_edge(node2, node1, gamma=gam, ded=0, cap=np.inf) #zero deductibles, inf caps
    return GG

def random_system_prop(Gprems, glb=0.05, gub=0.5):
    '''generate random financial system that conforms to given year's data with proportional contracts
    glb = lower bound on gamma, gub = upper bound on gamma'''
    node2ind = node2ind_dict(G)
    #note: the premiums ceded graph (G above) has edges reinsured->reinsurer; we will want the transpose eventually
    prems_paid = Gprems.out_degree(weight='weight')
    #generate random number for each node representing total reinsurance rate (Gamma^T*1) except if node has no reins contracts
    GammaT1 = np.random.uniform(glb,gub,len(Gprems.nodes()))
    
    GG = nx.DiGraph()
    GG.add_nodes_from(Gprems.nodes(data=True))
    #d_dict = {}
    for edge in Gprems.edges(data=True):
        node1, node2, prem = [edge[0], edge[1], edge[2]['weight']]
        gam = prem/prems_paid[node1]*GammaT1[node2ind[node1]]
        GG.add_edge(node2, node1, gamma=gam, ded=0, cap=np.inf) #zero deductibles, inf caps
    return GG

def set_comparable_systems(year):
    '''create XL and proportional models given data with all random numbers constant'''
    Gprems = multi2digraph(nx.read_gexf('reinsurance_graph/reinsurance_graph{}.gexf'.format(year)))
    co_state_data = get_co_state_data(Gprems,year)
    state2cos = state2cos_dict(co_state_data)
    G1 = random_system_xl(Gprems)
    fs_xl = FinancialSystem(Gprems, G1, state2cos)
    
    G2 = comparable_prop_system(Gprems, fs_xl.prems_prim, fs_xl.prems_f_reins)
    fs_prop = FinancialSystem(Gprems, G2, state2cos, prems_prim=fs_xl.prems_prim, prems_f_reins=fs_xl.prems_f_reins, capital=fs_xl.capital)
    return fs_xl, fs_prop



###############################################################################
#Automate simulations

def simulate_2(fs_xl, fs_prop, num=10):
    '''Simulate multiple shocks for XL and proportional models'''
    capital_vec = [fs_xl.capital[node] for node in fs_xl.G.nodes()]
    data250_xl = {}
    data250_prop = {}
    shocks_250 = {}
    shocks_100 = {}
    for i in range(num):
        sh = random_shock(fs_xl, typ=250)
        fs_xl.set_shock(sh)
        fs_prop.set_shock(sh)
        L_1, B_1, C_1 = solve_push(fs_xl)
        L_2, B_2, C_2 = solve_push(fs_prop)
        p_1, D_1 = eis_noe.clearing_p(L_1, capital_vec)
        p_2, D_2 = eis_noe.clearing_p(L_2, capital_vec)
        data250_xl[i] = [L_1, p_1, D_1]
        data250_prop[i] = [L_2, p_2, D_2]
        shocks_250[i] = sh
    
    data100_xl = {}
    data100_prop = {}
    for i in range(num):
        sh = random_shock(fs_xl, typ=100)
        fs_xl.set_shock(sh)
        fs_prop.set_shock(sh)
        L_1, B_1, C_1 = solve_push(fs_xl)
        L_2, B_2, C_2 = solve_push(fs_prop)
        p_1, D_1 = eis_noe.clearing_p(L_1, capital_vec)
        p_2, D_2 = eis_noe.clearing_p(L_2, capital_vec)
        data100_xl[i] = [L_1, p_1, D_1]
        data100_prop[i] = [L_2, p_2, D_2]
        shocks_100[i] = sh
    return data250_xl, data250_prop, data100_xl, data100_prop, shocks_250, shocks_100



###############################################################################
'''Comparing XL and proportional models: Simulation metrics'''

def tot_coverage(fs, node):
    '''Calculate total potential reinsurance coverage a node may need to payout'''
    cov = 0
    for edge in fs.G.in_edges(node, data=True):
        gam, d, c = edge[2]['gamma'], edge[2]['ded'], edge[2]['cap']
        cov += c #c is the max possible/limit payout of this contract
    return cov

def loss_to_equity(p, e):
    '''calculate loss/equity for clearing payment p and capital e'''
    for i in e:
        assert i>0
    return np.divide(p,e)

def liability_to_equity(L, e, sh):
    '''calculate net liability/equity for liability matrix L and capital e'''
    for i in e:
        assert i>0
    net_L = L.tocsr()*np.ones(len(e)) + sh - L.transpose().tocsr()*np.ones(len(e))
    return np.divide(net_L,e)

def equity_change(L, e, sh, p):
    '''calculate end_equity/start_equity given liabilities L, capital e, shock sh, and clearing payments p'''
    for i in e:
        assert i>0
    L1 = L.tocsr()*np.ones(len(e))
    alpha = np.array([p[i]/L1[i] if L1[i]>0 else 0 for i in range(len(p))])
    end_equity = e - p + L.transpose().tocsr()*alpha - sh
    return np.divide(end_equity, e)

def Freedman_Diaconis_h(data, n=None):
    '''rule for histogram bin width'''
    if n == None:
        n = len(data)
    #IQR = np.percentile(data, 75) - np.percentile(data,25)
    IQR = np.percentile(data, 95) - np.percentile(data,5)
    return 2.*IQR/n**(1/3.)

def make_histograms_2(e, shocks_250, shocks_100, data250_xl, data250_prop, data100_xl, data100_prop, meas='equity_change'):
    assert meas in ['liability_to_equity','equity_change']
    xl_250 = []
    xl_100 = []
    prop_250 = []
    prop_100 = []
    for i in range(len(data250_xl.keys())):
        if meas == 'equity_change':
            sh = shocks_250[i]
            L,p,D = data250_xl[i]
            xl_250 += list(equity_change(L,e,sh,p))
            L,p,D = data250_prop[i]
            prop_250 += list(equity_change(L,e,sh,p))
            
            sh = shocks_100[i]
            L,p,D = data100_xl[i]
            xl_100 += list(equity_change(L,e,sh,p))
            L,p,D = data100_prop[i]
            prop_100 += list(equity_change(L,e,sh,p))
        elif meas == 'liability_to_equity':
            sh = shocks_250[i]
            L,p,D = data250_xl[i]
            xl_250 += list(liability_to_equity(L, e, sh))
            L,p,D = data250_prop[i]
            prop_250 += list(liability_to_equity(L, e, sh))
            
            sh = shocks_100[i]
            L,p,D = data100_xl[i]
            xl_100 += list(liability_to_equity(L, e, sh))
            L,p,D = data100_prop[i]
            prop_100 += list(liability_to_equity(L, e, sh))
    
    mn = min(min(xl_250), min(xl_100), min(prop_250), min(prop_100))
    mx = max(max(xl_250), max(xl_100), max(prop_250), max(prop_100))
    h = Freedman_Diaconis_h(xl_250+xl_100+prop_250+prop_100, n=2*len(xl_250))
    num_bins = int(np.ceil((mx-mn)/h))
    wts = [0.4 for i in range(len(xl_250))] + [0.6 for i in range(len(xl_250))]
    hist_xl, bin_edges_xl = np.histogram(xl_250+xl_100, bins=num_bins, range=(mn,mx), weights=wts, density=False)
    hist_prop, bin_edges_prop = np.histogram(prop_250+prop_100, bins=num_bins, range=(mn,mx), weights=wts, density=False)
    
    plt.hist(xl_250+xl_100, bins=num_bins, range=(mn,mx), weights=wts, density=False, log=True)
    plt.xlim(min(bin_edges_xl), max(bin_edges_xl))
    plt.ylabel('Count (logscale)', fontsize=14)
    plt.title('Histogram Firm Returns: XL Model', fontsize=14)
    plt.xlabel('Firm equity return under XL model', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig('figures/hist_firm_equity_xl.eps')
    plt.show()
    plt.close()
    
    plt.hist(prop_250+prop_100, bins=num_bins, range=(mn,mx), weights=wts, density=False, log=True)
    plt.xlim(min(bin_edges_prop), max(bin_edges_prop))
    plt.ylabel('Count (logscale)', fontsize=14)
    plt.title('Histogram Firm Returns: Propotional Model', fontsize=14)
    plt.xlabel('Firm equity return under proportional model', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig('figures/hist_firm_equity_prop.eps')
    plt.show()
    return hist_xl, bin_edges_xl, hist_prop, bin_edges_prop, xl_250, xl_100, prop_250, prop_100

def compare_firm_equity_changes_histogram(xl_250, xl_100, prop_250, prop_100):
    '''compare firm-level difference in equity change between xl and prop models'''
    data_250 = list(np.array(prop_250) - np.array(xl_250))
    data_100 = list(np.array(prop_100) - np.array(xl_100))
    wts = [0.4 for i in range(len(xl_250))] + [0.6 for i in range(len(xl_250))]
    
    mn = min(min(data_250),min(data_100))
    mx = max(max(data_250),max(data_100))
    h = Freedman_Diaconis_h(data_250 + data_100, n=2*len(data_100))
    num_bins = int(np.ceil((mx-mn)/h))
    hist, bin_edges = np.histogram(data_250 + data_100, bins=num_bins, range=(mn,mx), weights=wts, density=False)
    plt.hist(data_250 + data_100, bins=num_bins, range=(mn,mx), weights=wts, density=False, log=True)
    #plt.bar(bin_edges[:-1], hist, width = bin_edges[1]-bin_edges[0])
    plt.xlim(mn,mx)
    plt.title('Histogram Difference in Firm Returns', fontsize=14)
    #plt.ylim(0, max(hist)*1.1)
    plt.ylabel('Count (logscale)', fontsize=14)
    plt.xlabel('Return Propotional model - Return XL model', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig('figures/hist_firm_equity_diff.eps')
    plt.show()
    
    #calculate probability wt on positive side of histogram:
    pos_wt_250 = np.sum([1 for elem in data_250 if elem>=0])/len(data_250)
    pos_wt_100 = np.sum([1 for elem in data_100 if elem>=0])/len(data_100)
    pos_wt = 0.6*pos_wt_100 + 0.4*pos_wt_250
    return hist, bin_edges, pos_wt

def compare_firm_equity_changes(xl_250, xl_100, prop_250, prop_100):
    '''plot equity change under xl vs. equity change under prop'''
    xl_data = xl_250 + xl_100
    prop_data = prop_250 + prop_100
    mn = min(min(xl_data),min(prop_data))
    mx = max(max(xl_data),max(prop_data))

    plt.plot([mn,mx],[mn,mx], 'k-', alpha=0.75, zorder=1)
    #d250 = plt.scatter(xl_250, prop_250, s=1, alpha=0.2)
    #d100 = plt.scatter(xl_100, prop_100, s=5, alpha=0.2)
    #plt.legend((d250, d100), ('250 yr shock','100 yr shock'))
    plt.scatter(xl_data, prop_data, alpha=0.2)
    plt.title('Firm Returns across all Simulations', fontsize=14)
    plt.xlim(mn, mx)
    plt.xlabel('XL model', fontsize=14)
    plt.ylim(mn, mx)
    plt.ylabel('Proportional model',fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig('figures/compare_firm_equity.eps')
    plt.show()
    plt.close()
    
    #histogram
    #h = Freedman_Diaconis_h(xl_250+xl_100+prop_250+prop_100, n=2*len(xl_250))
    #num_bins = int(np.ceil((mx-mn)/h))
    wts = [0.4 for i in range(len(xl_250))] + [0.6 for i in range(len(xl_250))]
    plt.plot([mn,mx],[mn,mx], 'k-', alpha=0.75, zorder=1)
    plt.hist2d(xl_data, prop_data, bins=60, weights=wts, norm=LogNorm())
    plt.title('2D-Histogram Firm Returns: XL vs. Prop.', fontsize=14)
    plt.xlim(mn, mx)
    plt.xlabel('XL model', fontsize=14)
    plt.ylim(mn, mx)
    plt.ylabel('Proportional model', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig('figures/histogram2d_compare_firm_equity.pdf')
    plt.show()
    return

def compare_agg_defaults(shocks_250, shocks_100, data250_xl, data250_prop, data100_xl, data100_prop):
    '''plot number of defaults under xl vs. number defaults under prop for each trial'''
    xl_250_defaults = []
    prop_250_defaults = []
    xl_100_defaults = []
    prop_100_defaults = []
    for i in range(len(data250_xl.keys())):
        sh = shocks_250[i]
        L,p,D = data250_xl[i]
        xl_250_defaults.append(np.sum(D))
        L,p,D = data250_prop[i]
        prop_250_defaults.append(np.sum(D))
        
        sh = shocks_100[i]
        L,p,D = data100_xl[i]
        xl_100_defaults.append(np.sum(D))
        L,p,D = data100_prop[i]
        prop_100_defaults.append(np.sum(D))
    
    xl_data = xl_250_defaults + xl_100_defaults
    prop_data = prop_250_defaults + prop_100_defaults
    mn = min(min(xl_data),min(prop_data))
    mx = max(max(xl_data),max(prop_data))
    
    plt.plot([mn,mx],[mn,mx], 'k-', alpha=0.75, zorder=1)
    #plt.scatter(xl_data, prop_data)
    d250 = plt.scatter(xl_250_defaults, prop_250_defaults, alpha=0.75)
    d100 = plt.scatter(xl_100_defaults, prop_100_defaults, alpha=0.75)
    plt.title('# Defaults per Shock', fontsize=14)
    plt.legend((d250, d100), ('250 yr shock','100 yr shock'), fontsize=14)
    plt.xlim(mn, mx)
    plt.xlabel('XL model', fontsize=14)
    plt.ylim(mn, mx)
    plt.ylabel('Proportional model', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig('figures/compare_agg_defaults.eps')
    plt.show()
    return

def agg_uncovered_claims(L, e, sh, p):
    d1 = np.multiply(equity_change(L, e, sh, p), e)
    d2 = [elem for elem in d1 if elem<0]
    return np.abs(np.sum(d2))

def compare_agg_uncovered_claims(e, shocks_250, shocks_100, data250_xl, data250_prop, data100_xl, data100_prop):
    '''plot aggregate net uncovered primary claims under xl model vs. proportional model'''
    xl_250_uncovered = []
    prop_250_uncovered = []
    xl_100_uncovered = []
    prop_100_uncovered = []
    for i in range(len(data250_xl.keys())):
        sh = shocks_250[i]
        L,p,D = data250_xl[i]
        xl_250_uncovered.append(agg_uncovered_claims(L, e, sh, p))
        L,p,D = data250_prop[i]
        prop_250_uncovered.append(agg_uncovered_claims(L, e, sh, p))
        
        sh = shocks_100[i]
        L,p,D = data100_xl[i]
        xl_100_uncovered.append(agg_uncovered_claims(L, e, sh, p))
        L,p,D = data100_prop[i]
        prop_100_uncovered.append(agg_uncovered_claims(L, e, sh, p))
    
    xl_data = xl_250_uncovered + xl_100_uncovered
    prop_data = prop_250_uncovered + prop_100_uncovered
    mn = min(min(xl_data),min(prop_data))
    mx = max(max(xl_data),max(prop_data))
    
    plt.plot([mn,mx],[mn,mx], 'k-', alpha=0.75, zorder=1)
    d250 = plt.scatter(xl_250_uncovered, prop_250_uncovered, alpha=0.75)
    d100 = plt.scatter(xl_100_uncovered, prop_100_uncovered, alpha=0.75)
    plt.title('Uncovered Primary Claims per Shock', fontsize=14)
    plt.legend((d250, d100), ('250 yr shock','100 yr shock'), fontsize=14)
    plt.xlim(mn, mx)
    plt.xlabel('XL model', fontsize=14)
    plt.ylim(mn, mx)
    plt.ylabel('Proportional model', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig('figures/compare_agg_uncoveredl.eps')
    plt.show()
    return


