# -*- coding: utf-8 -*-
"""
Created on Sat May 19 20:35:03 2018

This script constructs figures like the ones in the paper

@author: aak228
"""

import networkx as nx
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



