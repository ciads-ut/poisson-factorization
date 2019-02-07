import numpy as np

# ############  Functions  ############
def load_config_vars(config_fname):
    variables = dict()
    with open(config_fname, 'r') as fin:
        for line in fin:
            if '#' in line:
                line = line[0:line.find('#')]
            if '=' in line:		
                line_split = line.rstrip('\r\n').split('=')
                key = line_split[0]
                value = line_split[1]
                if '"' in value:
                    value = value[value.find('"')+1:value.find('"', value.find('"')+1)]
                else:
                    value = value.strip()
                variables[key] = value
    return variables

# Get P(n,kB) by marginalizing out m from P(n,m,kB) ~ rho[kB] * phi[n,kB] * phi[m,kB]
def P_nk(rkB, phink):
    P = np.zeros(phink.shape)
    # Sum over m
    N = phink.shape[0]
    for n in range(N):
        P += np.array(np.matrix(phink) * np.matrix(np.diag(rkB * phink[n,:])))
    P = P / np.sum(P)
    return P

# Get P(d,kY) by marginalizing out w from P(d,w,kY) ~ r[kY] * theta[d,kY] * beta[w,kY]
def P_dk(rkY, thetadk, betawk):
    P = np.zeros(thetadk.shape)
    # Sum over w
    V = betawk.shape[0]
    for w in range(V):
        P += np.array(np.matrix(thetadk) * np.matrix(np.diag(rkY * betawk[w,:])))
    P = P / np.sum(P)
    return P

# Get P(w,kY) by marginalizing out d from P(d,w,kY) ~ r[kY] * theta[d,kY] * beta[w,kY]
def P_wk(rkY, thetadk, betawk):
    P = np.zeros(betawk.shape)
    # Sum over d
    D = thetadk.shape[0]
    for d in range(D):
        P += np.array(np.matrix(betawk) * np.matrix(np.diag(rkY * thetadk[d,:])))
    P = P / np.sum(P)
    return P
