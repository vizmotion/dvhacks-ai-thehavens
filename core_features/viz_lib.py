import pandas as pd
import numpy as np
import networkx as nx
import pylab as plt

figure_path = '/Users/hugocontreras/Documents/GitHub/dvhacks-ai-thehavens/figures/'

def get_correlation_network(data,th_corr=0.5):
    """
    get_correlation_network: a funtion to plot the correlation graph from the data
    input:
        data: pandas dataframe with the data
        th_corr: correlation threshold
    output:
        png with the network
    """
    A = np.array((data.corr()**2>th_corr)*1)
    fig = plt.figure(figsize=(10,10))
    G=nx.from_numpy_matrix(A)
    pos=nx.shell_layout(G)
    labels = {}
    for ii in range(len(data.columns)):
        labels[ii] = data.columns[ii]
    nx.draw_networkx_labels(G,pos,labels,font_size=8)
    nx.draw_shell(G)
    plt.savefig(figure_path + '_correlation_network.png')

def similarity_db_column_overlapping(db1,db2):
    """
    similarity_db_column_overlapping: a function to compute Jaccard similarity between column lists
    input:
        db1: database1
        db2: database2
    output:
        Jaccard similarity between feature sets
    """
    return 1.*len(np.intersect1d(db1.columns,db2.columns))/len(np.unique(db1.columns.tolist()+db2.columns.tolist()))

def get_database_network(dDATA,th_sim=0.05):
    """
    get_database_network: a funtion to plot the correlation graph from the databases
    input:
        dDATA: dictionary with pandas dataframes as values and filenames as keys
        th_sim: similarity threshold
    output:
        png with the network
    """
    lfiles = dDATA.keys()
    A = np.zeros((len(lfiles),len(lfiles)))
    for ii in range(len(lfiles)):
    for jj in range(ii):
        sim = similarity_db_column_overlapping(dDATA[lfiles[ii]],dDATA[lfiles[jj]])
        A[ii][jj] = sim
        A[jj][ii] = sim
    AT = (A>th_sim)*1.
    fig = plt.figure(figsize=(10,10))
    G=nx.from_numpy_matrix(AT)
    pos=nx.shell_layout(G)
    labels = {}
    for ii in range(len(lfiles)):
        labels[ii] = lfiles[ii]
    nx.draw_networkx_labels(G,pos,labels,font_size=8)
    nx.draw_shell(G)

    plt.savefig(figure_path + '_database_network.png')
