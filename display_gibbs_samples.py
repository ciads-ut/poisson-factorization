import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pf_utils import load_config_vars

# ############  Main code  ############
# Load config file
config_fname = 'poisson.config'
if len(sys.argv) > 1:
    config_fname = sys.argv[1]
print('Loading ' + config_fname)
variables = load_config_vars(config_fname)
model = variables['RUN_MODEL']
KB = int(variables['NETWORK_TOPICS'])
in_dir = variables['OUT_DIR'] + '/' + model.upper() + '/iterations'
print('Displaying samples for ' + model.upper())
# Show network factors
if model == 'jgppf' or model == 'ngppf':
    print('Loading network iterations')
    # Get network dimensions
    with open(variables['NETWORK_TRAIN'], 'r') as fin:
        line = fin.readline()
        line_split = line.split('\t')
        N = int(line_split[0])
    # Load the phink samples
    phink_fnames = [fname for fname in os.listdir(in_dir) if fname[0:5]=='phink' and fname[-4:]=='.txt']
    phink_list = []
    for fname in phink_fnames:
        print(in_dir + '/' + fname)
        phink = np.zeros((N,KB))
        with open(in_dir + '/' + fname, 'r') as fin:
            for k,line in enumerate(fin):
                vals = line.rstrip('\r\n').split('\t')
                for n in range(N):
                    phink[n,k] = float(vals[n])
        sample_num = int(fname[len('phink-itr'):-len('.txt')])
        phink_list.append((sample_num, phink))
    # Load the rkB values
    rkB_fnames = [fname for fname in os.listdir(in_dir) if fname[0:3]=='rkB' and fname[-4:]=='.txt']
    rkB_list = []
    for fname in rkB_fnames:
        print(in_dir + '/' + fname)
        rkB = np.zeros(KB)
        with open(in_dir + '/' + fname, 'r') as fin:
            for k,line in enumerate(fin):
                rkB[k] = float(line)
        sample_num = int(fname[len('rkB-itr'):-len('.txt')])
        rkB_list.append((sample_num, rkB))
    # Display the samples
    num_samples = len(phink_list)
    phink_list.sort()
    rkB_list.sort()
    plt.ion() # enable interactive plotting
    for i in range(num_samples):
        plt.clf() # clears the figure (both plots) but leaves the window open
        plt.subplot(1,2,1)
        plt.plot(rkB_list[i][1])
        plt.xlabel('Group')
        plt.title('r_kB iter=' + str(rkB_list[i][0]))
        plt.subplot(1,2,2)
        plt.imshow(phink_list[i][1])
        plt.xlabel('Group')
        plt.ylabel('Senator')
        plt.title('phi_nk')
        plt.colorbar(shrink=0.7)
        plt.pause(0.5)
    plt.pause(10)
else:
    print('No network factors')
