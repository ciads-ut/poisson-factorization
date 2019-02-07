import os
import sys
import numpy as np
import networkx as nx
from skimage.filters import threshold_otsu
from pf_utils import load_config_vars

# ############  Parameters  ############
threshold_edges = True

# ############  Functions  ############
def node_attributes(n, rkB, phink, network_dictionary):
    result = {}
    result['group'] = str(np.argmax(phink[n,:]))
    result['label'] = network_dictionary[n]
    result['topPhi'] = phink[n,np.argmax(rkB)]
    return result

# ############  Main code  ############
# Load config file
config_fname = 'poisson.config'
if len(sys.argv) > 1:
    config_fname = sys.argv[1]
print('Loading ' + config_fname)
variables = load_config_vars(config_fname)
KB = int(variables['NETWORK_TOPICS'])
model = variables['RUN_MODEL']

# Load network factors
if model == 'ngppf' or model == 'jgppf':
    print('Loading network expected values')
    # Load network
    edges = []
    with open(variables['NETWORK_TRAIN'], 'r') as fin:
        line = fin.readline()
        line_split = line.rstrip('\r\n').split('\t')
        N = int(line_split[0])
        for line in fin:
            line_split = line.rstrip('\r\n').split('\t')
            edges.append((int(line_split[0]), int(line_split[1]), int(line_split[2])))
    rkB = np.zeros(KB)
    phink = np.zeros((N,KB))
    # Load rkB
    fname = variables['OUT_DIR'] + '/' + model.upper() + '/expectedValues/rkB.txt'
    print(fname)
    with open(fname, 'r') as fin:
        for k,line in enumerate(fin):
            rkB[k] = float(line.rstrip('\r\n'))
    # Load phink
    fname = variables['OUT_DIR'] + '/' + model.upper() + '/expectedValues/phink.txt'
    print(fname)
    with open(fname, 'r') as fin:
        for k,line in enumerate(fin):
            vals = line.rstrip('\r\n').split('\t')
            for n in range(N):
                phink[n,k] = float(vals[n])
    # Load the dictionary
    network_dictionary = dict()
    if variables.get('NETWORK_DICTIONARY', ''):
        with open(variables['NETWORK_DICTIONARY'], 'r') as fin:
            for line in fin:
                line_split = line.rstrip('\r\n').split('\t')
                network_dictionary[int(line_split[0])] = line_split[1]
    else:
        for n in range(N):
            network_dictionary[n] = str(n)
    # Create a graph
    G = nx.Graph()
    threshold = threshold_otsu(np.array([edge[2] for edge in edges]))
    for edge in edges:
        n = edge[0]
        m = edge[1]
        count = edge[2]
        if n not in G.nodes():
            G.add_node(n, **node_attributes(n, rkB, phink, network_dictionary))
        if m not in G.nodes():
            G.add_node(m, **node_attributes(m, rkB, phink, network_dictionary))
        if (threshold_edges and count >= threshold) or not threshold_edges:
            G.add_edge(n, m, weight=count)
    # Save the graph in GML format
    gml_dir = variables['OUT_DIR'] + '/' + model.upper() + '/GML'
    outname = 'pf_network.gml'
    if not os.path.isdir(gml_dir):
        os.mkdir(gml_dir)
    if not os.path.exists(gml_dir + '/' + outname):
        nx.write_gml(G, gml_dir + '/' + outname)
    else:
        print('Warning: ' + gml_dir + '/' + outname + ' already exists')

            
