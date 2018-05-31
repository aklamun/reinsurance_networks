# reinsurance_networks
Code for paper on cascades in reinsurance networks

The .py files implement functionality as follows:
- construct_insurance_graph.py: constructing graphs on premiums ceded from NAIC Schedule F data
- calc_reinsurance_liabilities.py: algorithms to calculate equilibrium liabilities
- eisenberg_noes_contagion_sparse.py: sparse matrix implementation of eisenberg-noe clearing algorithms
- simulate_contagion.py: functions to construct reinsurance networks, simulate shocks, and generate figures
- perturb_parameters.py: functions to perturb parameters of given reinsurance networks within an uncertainty range, and generate figures
- generate_figures.py: script to generate figures as in the paper

reinsurance_contract_data.xlsx contains data collected on reinsurance companies and their contracts (when made public). As mentioned in the paper, this is used to find reasonable simulation parameter values
