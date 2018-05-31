# -*- coding: utf-8 -*-
"""
Created on Sun May 20 14:32:45 2018

@author: aak228
"""

import networkx as nx
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import sparse
from simulate_contagion import *

###############################################################################
'''Perturb XL model'''

def calc_layer_structure(Gprems, tlp=0.2):
    '''solve knapsack for each node to sort into 2-layer structure,  tlp = top layer premiums / total tower premiums'''
    prems_paid = Gprems.out_degree(weight='weight')
    G = nx.DiGraph()
    G.add_nodes_from(Gprems.nodes(data=True))
    node_layers = {}
    for node in Gprems.nodes():
        edges = list(Gprems.out_edges(node,data=True))
        edges1, edges2, sum1, sum2 = knapsack(edges, tlp*prems_paid[node]) #sort into layers
        node_layers[node] = [edges1, edges2]
    return node_layers

def perturb_Gprems(Gprems, err_pct=0.05):
    '''perturb graph weights with err_pct=% uncertainty uniformly distributed'''
    Gprems2 = nx.DiGraph()
    Gprems2.add_nodes_from(Gprems.nodes(True))
    for edge in Gprems.edges(data=True):
        rn = np.random.uniform(1-err_pct, 1+err_pct)
        wt = edge[2]['weight']*rn
        Gprems2.add_edge(edge[0],edge[1], weight=wt)
    return Gprems2

def perturb_G_xl(Gprems1, node_layers, err_pct=0.05, pcratio=0.1, dcratio=4., cdratio=5., tlp=0.2):
    '''generate perturbed XL financial system given Gprems, prescribed node_layers, and coverage rules of thumb
    err_pct = percentage error applied to network parameters, uniformly distributed centered at the input value
    pcratio = premiums paid / coverage limit, dcratio = coverage limit / deductible, cdratio = coverage / deductible
    note coverage = coverage limit + deductible
    tlp = top layer premiums / total tower premiums'''
    #note: the premiums ceded graph (G above) has edges reinsured->reinsurer; we will want the transpose eventually
    Gprems = perturb_Gprems(Gprems1, err_pct)
    prems_paid = Gprems.out_degree(weight='weight')
    GG = nx.DiGraph()
    GG.add_nodes_from(Gprems.nodes(data=True))
    
    #handle reinsurance node-by-node (all contracts reinsuring given node)
    for node in Gprems.nodes():
        rn = np.random.uniform(1-err_pct, 1+err_pct,2)
        cov = prems_paid[node]/pcratio
        d = cov/dcratio*rn[0] #deductible
        c = cdratio*cov/dcratio*rn[1] #total coverage cap
        edges1, edges2 = node_layers[node] #use pre-sorted layers
        sum1 = np.sum([elem[2]['weight'] for elem in edges1])
        sum2 = np.sum([elem[2]['weight'] for elem in edges2])
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
    return GG, Gprems

def perturb_outside_prems(prems_prim, prems_f_reins, err_pct=0.05):
    '''perturb prems_prim and prems_f_reins by err_pct uniform distribution'''
    prems_prim2 = {}
    prems_f_reins2 = {}
    for node in prems_prim.keys():
        rn = np.random.uniform(1-err_pct, 1+err_pct)
        prems_prim2[node] = prems_prim[node]*rn
    for node in prems_f_reins.keys():
        rn = np.random.uniform(1-err_pct, 1+err_pct)
        prems_f_reins2[node] = prems_f_reins[node]*rn
    return prems_prim2, prems_f_reins2

def perturb_capital(capital, err_pct=0.05):
    '''perturb capital by err_pct uniform distribution'''
    capital2 = {}
    for node in capital.keys():
        rn = np.random.uniform(1-err_pct, 1+err_pct)
        capital2[node] = capital[node]*rn
    return capital

###############################################################################
'''Automate perturbation testing'''

def perturb_simulations(fs_xl, node_layers, err_pct=0.05, num_trials=1):
    results = []
    #simulate base case
    L_1, B_1, C_1 = solve_push(fs_xl)
    capital_vec = [fs_xl.capital[node] for node in fs_xl.G.nodes()]
    p_1, D_1 = eis_noe.clearing_p(L_1, capital_vec)
    results.append([L_1, p_1, D_1, capital_vec])
    
    #for loop: perturb system and simulate new case
    for i in range(num_trials):
        G, Gprems = perturb_G_xl(fs_xl.Gprems, node_layers, err_pct=err_pct)
        prems_prim, prems_f_reins = perturb_outside_prems(fs_xl.prems_prim, fs_xl.prems_f_reins, err_pct=err_pct)
        capital = perturb_capital(fs_xl.capital, err_pct=err_pct)
        fs = FinancialSystem(Gprems, G, fs_xl.state2cos, prems_prim=prems_prim, prems_f_reins=prems_f_reins, capital=capital)
        fs.set_shock(fs_xl.sh)
        L_1, B_1, C_1 = solve_push(fs)
        capital_vec = [fs.capital[node] for node in fs.G.nodes()]
        p_1, D_1 = eis_noe.clearing_p(L_1, capital_vec)
        results.append([L_1, p_1, D_1, capital_vec])
    return results

###############################################################################
'''Compare perturbation results'''

def compare_node_equities(results, sh, pct_err):
    diffs = []
    max_diffs = []
    d_diffs = []
    end_equities = []
    #base case
    [L,p,d,e] = results[0]
    e_chng = equity_change(L, e, sh, p)
    base_end_e = np.multiply(e_chng,e)
    d1 = d[:]
    
    for i in range(1,len(results)):
        [L,p,d,e] = results[i]
        e_chng1 = equity_change(L, e, sh, p)
        end_equities.append( np.multiply(e_chng1,e) )
        diff = e_chng - e_chng1
        diffs.append(diff)
        max_diffs.append(np.max(np.abs(diff)))
        d_diff = np.sum(np.abs(np.array(d) - np.array(d1)))
        d_diffs.append(d_diff)
    
    node_diffs_mat = np.transpose(np.array(diffs))
    node_max_diffs = [np.max(node_diffs_mat[i,:]) for i in range(len(e)) if np.max(node_diffs_mat[i,:]) > 0]
    #plt.hist(np.array(diffs).flatten(), bins=70)
    plt.hist(node_max_diffs, bins=70, density=False, log=True)
    plt.ylabel('Count (logscale)',fontsize=14)
    plt.xlabel('|Difference firm equity return| > 0', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title('Histogram: {}% Perturbation Effects'.format(pct_err), fontsize=14)
    #Note: this is after removing elements that have not changed
    plt.xlim(np.min(node_max_diffs),np.max(node_max_diffs))
    plt.tight_layout()
    plt.savefig('figures/compare_equity_perturb_{}.eps'.format(pct_err))
    plt.show()
    plt.close()
    
    perturbed_end_e = []
    extend_base_end_e = []
    for i in range(len(results)-1):
        perturbed_end_e += list(end_equities[i])
        extend_base_end_e += list(base_end_e)
    mx_e_chng = np.max(np.abs(np.array(perturbed_end_e) - np.array(extend_base_end_e)))
    plt.hist2d(perturbed_end_e, extend_base_end_e, bins=40, norm=LogNorm())
    plt.show()
    
    return max_diffs, d_diffs, mx_e_chng



