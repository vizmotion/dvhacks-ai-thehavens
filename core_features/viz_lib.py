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
