import os
import sys
import numpy as np
import pyLDAvis
from shutil import copyfile
from pf_utils import load_config_vars, P_nk, P_dk, P_wk

# ############  Parameters  ############
corpus_n_terms = 30
network_n_terms = 50
LDAvis_htmljs_path = '../LDAvis/inst/htmljs'

# ############  Functions  ############   
def generate_index_html(variables, path, json_names, config_fname):
    # First determine which configuration variables to display
    config1 = []
    config2 = []
    config3 = []
    links = []
    if variables['RUN_MODEL'].lower() == 'jgppf':
        header = 'Joint Gamma Process Poisson Factorization (J-GPPF)'
        config1.append('NETWORK_TRAIN')
        config1.append('CORPUS_TRAIN')
        config1.append('AUTHORS_TRAIN')
        if variables.get('NETWORK_HELDOUT', ''): config1.append('NETWORK_HELDOUT')
        if variables.get('CORPUS_HELDOUT', ''): config1.append('CORPUS_HELDOUT')
        config2.append('NETWORK_TOPICS')
        config2.append('CORPUS_TOPICS')
        if variables.get('NETWORK_DICTIONARY', ''): config3.append('NETWORK_DICTIONARY')
        if variables.get('CORPUS_DICTIONARY', ''): config3.append('CORPUS_DICTIONARY')
    elif variables['RUN_MODEL'].lower() == 'ngppf':
        header = 'Network-only Gamma Process Poisson Factorization (N-GPPF)'
        if variables.get('NETWORK_HELDOUT', ''): config1.append('NETWORK_HELDOUT')
        config1.append('NETWORK_TRAIN')
        config2.append('NETWORK_TOPICS')
        if variables.get('NETWORK_DICTIONARY', ''): config3.append('NETWORK_DICTIONARY')
    elif variables['RUN_MODEL'].lower() == 'cgppf':
        header = 'Corpus-only Gamma Process Poisson Factorization (C-GPPF)'
        config1.append('CORPUS_TRAIN')
        if variables.get('CORPUS_HELDOUT', ''): config1.append('CORPUS_HELDOUT')
        config2.append('CORPUS_TOPICS')
        if variables.get('CORPUS_DICTIONARY', ''): config3.append('CORPUS_DICTIONARY')
    config1.append('OUT_DIR')
    config2.append('BURNIN_ITER')
    config2.append('COLLECT_ITER')
    config2.append('COUNT_FLAG')
    config2.append('EPSILON')
    links = ['<p><a href="' + json_name + '.html">' + json_name[0].upper() + json_name[1:].lower() + ' Factors</a></p>' for json_name in json_names]
    # Save the variables and links to html
    with open(path + '/index.html', 'w') as fout:
        fout.write('<!DOCTYPE html>\n')
        fout.write('<html>\n')
        fout.write('  <head>\n')
        fout.write('    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">\n')
        fout.write('    <title>LDAvis</title>\n')
        fout.write('    <link rel="stylesheet" type="text/css" href="lda.css">\n')
        fout.write('  </head>\n\n')
        fout.write('  <body>\n')
        fout.write('    <h1>' + header + '</h1>\n')
        fout.write('    <h3>Configuration</h3>\n')
        fout.write('    <p>Source: ' + config_fname + '</p>\n')
        fout.write('    <p>\n')
        for var in config1:
            fout.write('      ' + var + ': ' + variables[var] + '<br>\n')
        fout.write('    </p>\n')
        fout.write('    <p>\n')
        for var in config2:
            fout.write('      ' + var + ': ' + variables[var] + '<br>\n')
        fout.write('    </p>\n')
        fout.write('    <p>\n')
        for var in config3:
            fout.write('      ' + var + ': ' + variables[var] + '<br>\n')
        fout.write('    </p>\n')
        fout.write('    <h3>LDAvis Results</h3>\n')
        for link in links:
            fout.write('    ' + link + '\n')
        fout.write('  </body>\n\n')
        fout.write('</html>\n')
    
def generate_LDAvis_html(json_name, path):
    header = json_name[0].upper() + json_name[1:].lower() + ' Factors'
    with open(path + '/' + json_name + '.html', 'w') as fout:
        fout.write('<!DOCTYPE html>\n')
        fout.write('<html>\n')
        fout.write('  <head>\n')
        fout.write('    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">\n')
        fout.write('    <title>' + header + '</title>\n')
        fout.write('    <script src="d3.v3.js"></script>\n')
        fout.write('    <script src="ldavis.js"></script>\n')
        fout.write('    <link rel="stylesheet" type="text/css" href="lda.css">\n')
        fout.write('  </head>\n\n')
        fout.write('  <body>\n')
        fout.write('    <h1>' + header + '</h1>\n')
        fout.write('    <div id = "lda"></div>\n')
        fout.write('    <script>\n')
        fout.write('      var vis = new LDAvis("#lda", "' + json_name + '.json");\n')
        fout.write('    </script>\n')
        fout.write('  </body>\n\n')
        fout.write('</html>\n')

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
print('Generating LDAvis results for ' + model)

# Load corpus factors
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

# Load network factors
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

# Generate LDAvis json files
json_names = []
pyLDAvis_dir = variables['OUT_DIR'] + '/' + model.upper() + '/LDAvis'
if not os.path.isdir(pyLDAvis_dir):
    os.mkdir(pyLDAvis_dir)

if model == 'cgppf' or model == 'jgppf':
    # Generate the corpus files
    json_name = 'corpus'
    json_names.append(json_name)
    if not os.path.exists(pyLDAvis_dir + '/' + json_name + '.json'):
        print('Generating ' + json_name + '.json')
        # Topic-term probabilities
        topic_term_dists = P_wk(rkY, thetadk, betawk)
        topic_term_dists = np.array(np.matrix(topic_term_dists) * np.matrix(np.diag(np.sum(topic_term_dists, axis=0)**-1))).T
        # Document-topic probabilities
        doc_topic_dists = P_dk(rkY, thetadk, betawk)
        doc_topic_dists = np.array(np.matrix(np.diag(np.sum(doc_topic_dists, axis=1)**-1)) * np.matrix(doc_topic_dists))
        # Document lengths and term frequencies
        doc_lengths = np.zeros(D)
        term_frequency = np.zeros(V)
        with open(variables['CORPUS_TRAIN'], 'r') as fin:
            for num,line in enumerate(fin):
                if num > 0:
                    vals = line.rstrip('\r\n').split('\t')
                    doc_id = int(vals[0])
                    word_id = int(vals[1])
                    word_count = int(vals[2])
                    doc_lengths[doc_id] += word_count
                    term_frequency[word_id] += word_count
        # Dictionary terms
        vocab = [corpus_dictionary[word_id] for word_id in range(V)]
        # Generate the JSON and html
        prepared_data = pyLDAvis.prepare(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency, R=corpus_n_terms)
        pyLDAvis.save_json(prepared_data, pyLDAvis_dir + '/' + json_name + '.json')
        print('Generating ' + json_name + '.html')
        generate_LDAvis_html(json_name, pyLDAvis_dir)
    else:
        print('Warning: ' + pyLDAvis_dir + '/' + json_name + '.json already exists')

if model == 'ngppf' or model == 'jgppf':
    # Generate the network files
    json_name = 'network'
    json_names.append(json_name)
    if not os.path.exists(pyLDAvis_dir + '/' + json_name + '.json'):
        print('Generating ' + json_name + '.json')
        # Topic-term probabilities
        topic_term_dists = P_nk(rkB, phink)
        topic_term_dists = np.array(np.matrix(topic_term_dists) * np.matrix(np.diag(np.sum(topic_term_dists, axis=0)**-1))).T
        # Document-topic probabilities
        doc_topic_dists = P_nk(rkB, phink)
        doc_topic_dists = np.array(np.matrix(np.diag(np.sum(doc_topic_dists, axis=1)**-1)) * np.matrix(doc_topic_dists))
        # Document lengths and term frequencies
        doc_lengths = np.zeros(N)
        term_frequency = np.zeros(N)
        with open(variables['NETWORK_TRAIN'], 'r') as fin:
            for num,line in enumerate(fin):
                if num > 0:
                    vals = line.rstrip('\r\n').split('\t')
                    doc_id = int(vals[0])
                    word_id = int(vals[1])
                    word_count = int(vals[2])
                    doc_lengths[doc_id] += word_count
                    term_frequency[word_id] += word_count
        # Dictionary terms
        vocab = [network_dictionary[word_id] for word_id in range(N)]
        # Generate the JSON and html
        prepared_data = pyLDAvis.prepare(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency, R=network_n_terms)
        pyLDAvis.save_json(prepared_data, pyLDAvis_dir + '/' + json_name + '.json')
        print('Generating ' + json_name + '.html')
        generate_LDAvis_html(json_name, pyLDAvis_dir)
    else:
        print('Warning: ' + pyLDAvis_dir + '/' + json_name + '.json already exists')

# Generate LDAvis index file
if not os.path.exists(pyLDAvis_dir + '/index.html'):
    print('Generating index.html')
    generate_index_html(variables, pyLDAvis_dir, json_names, config_fname)
else:
    print('Warning: ' + pyLDAvis_dir + '/index.html already exists')
# Copy over d3js files
for htmljs_fname in ['d3.v3.js', 'lda.css', 'ldavis.js']:
    if not os.path.exists(pyLDAvis_dir + '/' + htmljs_fname):
        if os.path.exists(LDAvis_htmljs_path + '/' + htmljs_fname):
            copyfile(LDAvis_htmljs_path + '/' + htmljs_fname, pyLDAvis_dir + '/' + htmljs_fname)
        else:
            print('Error: cannot find ' + htmljs_fname)


