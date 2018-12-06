# -*- coding: utf-8 -*-
"""
Created on Sat May 19 20:35:03 2018

This script constructs figures like the ones in the paper

@author: aak228
"""

import networkx as nx
from scipy import sparse
import matplotlib.pyplot as plt
from simulate_contagion import *
from perturb_parameters import *

###############################################################################
'''Figures to systemic effects of XL and proportional systems'''

#construct 2 systems: one XL model and one proportional model
year = 2012
fs_xl, fs_prop = set_comparable_systems(year)

#simulate multiple shocks: 50 1-in-100 and 50 1-in-250 for each model
data250_xl, data250_prop, data100_xl, data100_prop, shocks_250, shocks_100 = simulate_2(fs_xl, fs_prop, num=50)
e = [fs_xl.capital[node] for node in fs_xl.G.nodes()] #capital (equity) values as a vector

#firm-level comparisons (each data point represents a firm in a given model with a given shock)
hist_xl, bin_edges_xl, hist_prop, bin_edges_prop, xl_250, xl_100, prop_250, prop_100 = make_histograms_2(e, shocks_250, shocks_100, data250_xl, data250_prop, data100_xl, data100_prop, meas='equity_change')
hist, bin_edges, pos_wt = compare_firm_equity_changes_histogram(xl_250, xl_100, prop_250, prop_100)
compare_firm_equity_changes(xl_250, xl_100, prop_250, prop_100)

#aggregate comparisons (each data point represents the whole network in a given model with a given shock)
compare_agg_defaults(shocks_250, shocks_100, data250_xl, data250_prop, data100_xl, data100_prop)
compare_agg_uncovered_claims(e, shocks_250, shocks_100, data250_xl, data250_prop, data100_xl, data100_prop)


###############################################################################
'''Figures to perturbations of XL system parameters'''

#calculate static XL layer structure
node_layers = calc_layer_structure(Gprems)

#construct base case to compare with (no perturbation)
G1, Gprems1 = perturb_G_xl(Gprems, node_layers, err_pct=0) #construct the base case
fs_xl = FinancialSystem(Gprems1, G1, state2cos, prems_prim=prems_prim, prems_f_reins=prems_f_reins, capital=capital)

#construct a static shock
sh = random_shock(fs_xl, typ=250)
fs_xl.set_shock(sh)

#simulate systems with perturbations +/- 2.5%, 5%, 10%, 20%
results = perturb_simulations(fs_xl, node_layers, err_pct=0.025, num_trials=50)
results5 = perturb_simulations(fs_xl, node_layers, err_pct=0.05, num_trials=50)
results10 = perturb_simulations(fs_xl, node_layers, err_pct=0.1, num_trials=50)
results20 = perturb_simulations(fs_xl, node_layers, err_pct=0.2, num_trials=50)

#compare resulting equity and default perturbations
max_diffs, d_diffs, mx_e_chng = compare_node_equities(results,fs_xl.sh, 2.5)
max_diffs5, d_diffs5, mx_e_chng5 = compare_node_equities(results5,fs_xl.sh, 5)
max_diffs10, d_diffs10, mx_e_chng10 = compare_node_equities(results10,fs_xl.sh, 10)
max_diffs20, d_diffs20, mx_e_chng20 = compare_node_equities(results20,fs_xl.sh, 20)


###############################################################################
'''Export graph files for visualization'''

def construct_G_for_export(fs):
    GG = nx.DiGraph()
    for node in fs.G.nodes(data=True):
        if fs.G.out_degree(node[0],weight='gamma') > 0:
            reinsurer = 1
        else:
            reinsurer = 0
        GG.add_node(node[0], label=node[1]['label'], reins=reinsurer)
    for edge in fs.G.edges(data=True):
        GG.add_edge(edge[0],edge[1],weight=edge[2]['gamma'])
    return GG

Gexport = construct_G_for_export(fs_xl)
nx.write_gexf(Gexport, 'reinsurance_graph/fs_xl_G_2012.gexf')

Gexport = construct_G_for_export(fs_prop)
nx.write_gexf(Gexport, 'reinsurance_graph/fs_prop_G_2012.gexf')


###############################################################################
'''Figures to compare time dependency of claims'''

#generate one 250-year shock and split into two smaller shocks
sh_1 = random_shock(fs_xl, typ=250)/2
sh_2 = random_shock(fs_xl, typ=250)/2
sh = sh_1 + sh_2


'''simulation 1: all claims come in single period'''
capital_vec = [fs_xl.capital[node] for node in fs_xl.G.nodes()]
fs_xl.set_shock(sh)
L, B, C = solve_push(fs_xl)
p, D = eis_noe.clearing_p(L, capital_vec)
rets1 = equity_change(L, capital_vec, sh, p)

'''simulation 2: half claims come in first period, half claims come in second period'''
capital_vec = [fs_xl.capital[node] for node in fs_xl.G.nodes()]
#first period
fs_xl.set_shock(sh_1)
L_1, B_1, C_1 = solve_push(fs_xl)
p_1, D_1 = eis_noe.clearing_p(L_1, capital_vec)
capital_vec1 = np.multiply(equity_change(L_1, capital_vec, sh_1, p_1), np.array(capital_vec))
capital_vec_pos = np.array([i if i>0 else 0 for i in capital_vec1])

#second period
fs_xl.set_shock(sh_2)
L_2, B_2, C_2 = solve_push(fs_xl)
p_2, D_2 = eis_noe.clearing_p(L_2, capital_vec_pos)

#calculate total equity change over two periods
L1 = L_2.tocsr()*np.ones(len(capital_vec1))
alpha = np.array([p_2[i]/L1[i] if L1[i]>0 else 0 for i in range(len(p_2))])
end_equity = capital_vec1 - p_2 + L_2.transpose().tocsr()*alpha - sh_2
rets2 = np.divide(end_equity, capital_vec)

#plot histogram of change in returns
rets_compare = rets2-rets1
mn = min(rets_compare)
mx = max(rets_compare)
h = Freedman_Diaconis_h(rets_compare, n=len(rets_compare))
num_bins = int(np.ceil((mx-mn)/h))
hist, bin_edges = np.histogram(rets_compare, bins=num_bins, range=(mn,mx), density=False)
plt.hist(rets_compare, bins=num_bins, range=(mn,mx), density=False, log=True)
plt.xlim(mn,mx)
plt.title('Histogram Difference in Firm Returns', fontsize=14)
plt.ylabel('Count (logscale)', fontsize=14)
plt.xlabel('Return 2 Clearings - Return Single Clearing', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('figures/hist_firm_equity_time_diff.eps')
plt.show()
