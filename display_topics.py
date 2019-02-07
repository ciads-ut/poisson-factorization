import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib
from pf_utils import load_config_vars, P_nk, P_dk, P_wk

# ############  Parameters  ############
display_topics = 15
display_words = 12
display_groups = 2

# ############  Main code  ############
# Load config file
config_fname = 'poisson.config'
if len(sys.argv) > 1:
    config_fname = sys.argv[1]
print('Loading ' + config_fname)
variables = load_config_vars(config_fname)
KB = int(variables['NETWORK_TOPICS'])
KY = int(variables['CORPUS_TOPICS'])
model = variables['RUN_MODEL']
print('Displaying results for ' + model + '\n')

# Show topics
if model == 'cgppf' or model == 'jgppf':
    print('Loading corpus expected values')
    # Get corpus dimensions
    with open(variables['CORPUS_TRAIN'], 'r') as fin:
        line = fin.readline()
        line_split = line.rstrip('\r\n').split('\t')
        D = int(line_split[0])
        V = int(line_split[1])
    rkY = np.zeros(KY)
    betawk = np.zeros((V,KY))
    thetadk = np.zeros((D,KY))
    # Load rkY
    fname = variables['OUT_DIR'] + '/' + model.upper() + '/expectedValues/rkY.txt'
    print(fname)
    with open(fname, 'r') as fin:
        for k,line in enumerate(fin):
            rkY[k] = float(line.rstrip('\r\n'))
    # Load betawk
    fname = variables['OUT_DIR'] + '/' + model.upper() + '/expectedValues/betawk.txt'
    print(fname)
    with open(fname, 'r') as fin:
        for k,line in enumerate(fin):
            vals = line.rstrip('\r\n').split('\t')
            for w in range(V):
                betawk[w,k] = float(vals[w])
    # Load thetadk
    fname = variables['OUT_DIR'] + '/' + model.upper() + '/expectedValues/thetadk.txt'
    print(fname)
    with open(fname, 'r') as fin:
        for k,line in enumerate(fin):
            vals = line.rstrip('\r\n').split('\t')
            for d in range(D):
                thetadk[d,k] = float(vals[d])
    # Load the dictionary
    corpus_dictionary = dict()
    if variables.get('CORPUS_DICTIONARY', ''):
        with open(variables['CORPUS_DICTIONARY'], 'r') as fin:
            for line in fin:
                line_split = line.rstrip('\r\n').split('\t')
                corpus_dictionary[int(line_split[0])] = line_split[1]
    else:
        for w in range(V):
            corpus_dictionary[w] = str(w)
    # Show the top words per topic
    term_topic_joint = P_wk(rkY, thetadk, betawk)
    topic_dist = np.sum(term_topic_joint, axis=0)
    top_topics = np.argsort(topic_dist)
    print('\nTop words per corpus topic:')
    for k in range(display_topics):
        topic = top_topics[-k-1]
        top_words = np.argsort(term_topic_joint[:,topic])
        strout = 'Topic ' + str(topic) + ' (' + str(topic_dist[topic]) + '):'
        for w in range(display_words):
            strout += ' ' + corpus_dictionary[top_words[-w-1]]
        print(strout)
    print(' ')

# Show network factors
if model == 'ngppf' or model == 'jgppf':
    print('Loading network expected values')
    # Get network dimensions
    with open(variables['NETWORK_TRAIN'], 'r') as fin:
        line = fin.readline()
        line_split = line.rstrip('\r\n').split('\t')
        N = int(line_split[0])
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
    # Display the top authors per group
    term_topic_joint = P_nk(rkB, phink)
    topic_dist = np.sum(term_topic_joint, axis=0)
    top_topics = np.argsort(topic_dist)
    print('\nTop authors per network group:')
    for k in range(display_groups):
        topic = top_topics[-k-1]
        top_words = np.argsort(term_topic_joint[:,topic])
        strout = 'Group ' + str(topic) + ' (' + str(topic_dist[topic]) + '):'
        for w in range(display_words):
            strout += ' ' + network_dictionary[top_words[-w-1]]
        print(strout)
    print(' ')
    # Display the matrices
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(rkB)
    plt.xlabel('Factors (kB)')
    plt.title('r_kB')
    plt.subplot(1,2,2)
    plt.imshow(phink, cmap='jet') # Groups (network)
    plt.xlabel('Factors (kB)')
    plt.ylabel('Network vertices (n)')
    plt.title('phi_nk')
    plt.colorbar(shrink=0.7)
    print(' ')

# Show joint results
if model == 'jgppf':
    print('Loading joint modeling expected values')
    psiwk = np.zeros((V,KB))
    Znd = np.zeros((N,D))
    # Load psiwk
    fname = variables['OUT_DIR'] + '/' + model.upper() + '/expectedValues/psiwk.txt'
    print(fname)
    with open(fname, 'r') as fin:
        for k,line in enumerate(fin):
            vals = line.rstrip('\r\n').split('\t')
            for w in range(V):
                psiwk[w,k] = float(vals[w])
    # Load Z
    fname = variables['AUTHORS_TRAIN']
    print(fname)
    with open(fname, 'r') as fin:
        for line in fin:
            vals = line.rstrip('\r\n').split('\t')
            author_id = int(vals[0])
            doc_id = int(vals[1])
            Znd[author_id, doc_id] = 1
    # Display the top words per group
    Zphidk = np.array(np.matrix(Znd.T) * np.matrix(phink))
    term_topic_joint = P_wk(rkB, Zphidk, psiwk)
    topic_dist = np.sum(term_topic_joint, axis=0)
    top_topics = np.argsort(topic_dist)
    print('\nTop words per network group:')
    for k in range(display_groups):
        topic = top_topics[-k-1]
        top_words = np.argsort(term_topic_joint[:,topic])
        strout = 'Group ' + str(topic) + ' (' + str(topic_dist[topic]) + '):'
        for w in range(display_words):
            strout += ' ' + corpus_dictionary[top_words[-w-1]]
        print(strout)
    print(' ')

plt.show()


