# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:11:26 2018

This file constructs graphs of NAICS reinsurance contract data from Schedule F
Those data files are necessary to run the code

@author: aak228
"""

import pandas as pd
import networkx as nx


def get_data(year):
    if int(year) not in [2012, 2013, 2014, 2015, 2016]:
        raise Exception("No data for year {}".format(year))
    else:
        #find the right document number
        num = int(year) % 2010
        
    #Get company data
    df_co = pd.read_excel('P{}012000.xlsx'.format(num),sheetname='P{}012000'.format(num),header=None)
    df_co.columns=['COCODE','SHORT_COMPANY_NAME','FULL_COMPANY_NAME','SURVIVING_COCODE','BUSINESS_TYPE','BUSINESS_TYPE_DESC','BUSINESS_SUB_TYPE','BUSINESS_SUB_TYPE_DESC','FILING_TYPE','FILING_TYPE_DESC','COMPANY_TYPE','COMPANY_TYPE_DESC','COMPANY_SUB_TYPE','COMPANY_SUB_TYPE_DESC','FEIN','STATE_DOMICILE','COMM_BUS_DATE','GROUP_CODE','GROUP_NAME','GROUP_CODE_PRIOR_PERIOD','GROUP_NAME_PRIOR_PERIOD','COMPANY_STATUS','COMPANY_STATUS_DESC','COUNTRY_NAME']
    df_co.index=df_co['COCODE']
    
    #Get reinsurance data
    df = pd.read_excel('P{}012051.xlsx'.format(num),sheetname='P{}012051'.format(num),header=None)
    df = df.drop(21,axis=1)
    df = df.drop(len(df)-1)
    df.columns=['COCODE','LINE_NO','FEIN','NAIC_COCODE','NAM_OF_REINSURER','DOMICIL_JURISDICT','REINS_CONTRACTS','REINS_PREM_CED','PD_LSSES','PD_LAE','KNOWN_CASE_LSS_RES','KNOWN_CASE_LAE_RES','IBNR_LSS_RES',',IBNR_LAE_RES','UNEARN_PREM','CNTG_COMM',',TOT_RCVRBL','CED_BAL_PAYABLE','OTH_AMTS','NET_RCVRBL','FUNDS_REINS_TREAT']
    df = df.apply(pd.to_numeric, errors='ignore')
    df['REINS_PREM_CED'] = df['REINS_PREM_CED'].replace(' ', 0)
    
    #Filter relevant data
    df_filtered = df[(df.LINE_NO % 10000 < 9998)&(pd.to_numeric(df.REINS_PREM_CED) > 0)&(pd.to_numeric(df.NAIC_COCODE) != 0)]
    df_filtered = df_filtered[['COCODE','NAIC_COCODE','FEIN','NAM_OF_REINSURER','REINS_PREM_CED']]
    
    return df_co, df, df_filtered

def get_nodes(df_filtered, df_co):
    nodes = list(set(list(df_filtered['COCODE']) + list(df_filtered['NAIC_COCODE'])))
    node_tuples = []
    for node in nodes:
        if node in df_co['COCODE']:
            tup = (node,df_co.loc[node]['SHORT_COMPANY_NAME'])
        else:
            tup = (node,'No available name')
        node_tuples.append(tup)
    return node_tuples

def create_graph(node_tuples, df_filtered, name, year):
    G = nx.MultiDiGraph()
    for tup in node_tuples:
        G.add_node(tup[0],label=tup[1])
    for i in df_filtered.index:
        start = df_filtered.loc[i]['COCODE']
        end = df_filtered.loc[i]['NAIC_COCODE']
        wt = df_filtered.loc[i]['REINS_PREM_CED']
        G.add_edge(start, end, weight = wt)
    
    # write the graph to .gexf file
    nx.write_gexf(G, '{}{}.gexf'.format(name,year))
    return G

def publicly_traded_companies():
    #Get public company matchings
    df_pub = pd.read_excel('NAME-COCODE.xlsx',header=0)
    df_pub.index=df_pub['Symbol']
    
    #remove cocode NaNs
    df_pub = df_pub[pd.notnull(df_pub['COCODE'])]
    df_pub = df_pub[['Symbol','Name','COCODE']]
    
    cocode_list = []
    for ls in df_pub['COCODE']:
        ls = str(ls).replace(' ','').split(',')
        ls = [int(i) for i in ls]
        cocode_list.append(ls)
    df_pub['COCODE_LIST'] = pd.Series(cocode_list, index=df_pub.index)
    return df_pub[['Symbol','Name','COCODE_LIST']]

def create_publicly_traded_subgraph(G):
    df_pub = publicly_traded_companies()
    #create subgraph of these cocode nodes
    cocodes_all_pub = []
    for ls in df_pub['COCODE_LIST']:
        cocodes_all_pub += ls
    cocodes_all_pub = list(set(cocodes_all_pub))
    H = G.subgraph(cocodes_all_pub)
    return H

def create_public_ticker_graph(H):
    #input H is subgraph of publicly traded nodes
    #note that the dictionary node_mapping will cause error if there are joint ventures
    df_pub = publicly_traded_companies()
    node_mapping = {}
    for sym in df_pub.index:
        ls = df_pub.loc[sym]['COCODE_LIST']
        for co in ls:
            node_mapping[co] = sym
    N = nx.MultiDiGraph()
    for sym in df_pub.index:
        G.add_node(sym,label=df_pub.loc[sym]['Name'])
    for edge in H.edges(data=True):
        N.add_edge(node_mapping[edge[0]],node_mapping[edge[1]],weight=edge[2]['weight'])
    return N
    
    
    
#Construct graphs for 2012
df_co, df, df_filtered = get_data(2012)
node_tuples = get_nodes(df_filtered, df_co)
G = create_graph(node_tuples, df_filtered, 'reinsurance_graph', 2012)
H = create_publicly_traded_subgraph(G)
N = create_public_ticker_graph(H)
nx.write_gexf(H, 'reins_graph2012_publicly_traded.gexf')
nx.write_gexf(N, 'reins_graph2012_public_tickers.gexf')


