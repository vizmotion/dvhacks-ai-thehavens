import pandas as pd
import numpy as np
import networkx as nx
import pylab as plt

#file_path = '/Users/hugocontreras/Documents/GitHub/dvhacks-ai-thehavens/files/'
file_path = ''
def get_connected_components_documentation(data,th_corr=0.5):
    """
    get_connected_components_documentation: a function to write the summary of the
    correlation graph from the data, using the largest connected components
    input:
        data: pandas dataframe with the data
        th_corr: correlation threshold
    output:
        text file with summary of the data
    """
    A = np.array((data.corr()**2>th_corr)*1)
    fig = plt.figure(figsize=(10,10))
    G=nx.from_numpy_matrix(A)
    graphs = list(nx.connected_component_subgraphs(G))
    print graphs[1].nodes
    labels = {}
    for ii in range(len(data.columns)):
        labels[ii] = data.columns[ii]

    fw = open(file_path + 'ConnectedComponents.txt','w')
    line0 = 'The clusters of features with highest dependency are: '
    fw.write(line0 + '\n')
    for ii in range(len(graphs)):
        if len(graphs[ii].nodes)>1:
            str_labels = ''
            for word in [labels[item] for item in graphs[ii].nodes]:
                str_labels = str_labels + word + ', '
            fw.write('Cluster ' + str(ii) + ': ' + str_labels[:-2] + '\n')
    fw.close()
