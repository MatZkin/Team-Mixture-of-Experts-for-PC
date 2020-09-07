import numpy as np


K=2
P=10


def buildDataDisc(n,randomness,res,flip):
    if(not randomness):
        np.random.seed(1)
    #Generate random
    if res==0:
        sigma = np.round(np.random.rand(n,K))
    else:
        sigma =np.round(np.random.rand(n,K)*(res+1)-0.5)/res
    for i in range(n):
        sigma[i, :] = np.sort(sigma[i, :])
    if(flip):
        sigma = np.fliplr(sigma)
    #Number of training samples to be generated
    #Noise model of "Team Deep Neural Networks for Interference Channels"
    G=np.square(np.random.normal(0, 1, (n, K,K)))+np.square(np.random.normal(0, 1, (n, K,K))) #True channel matrix
    G_hat=[]
    #Noise variance
    #User 1 CSI noise variance
    for i in range(K):
        Sigma_i = np.ones((n, K, K)) * sigma[:,i,np.newaxis,np.newaxis]
        OneMat=np.ones((n, K, K))
        SigmaBar_i=np.sqrt(OneMat-np.square(Sigma_i))
        Delta=np.square(np.random.normal(0, 1, (n, K,K)))+np.square(np.random.normal(0, 1, (n, K,K)))
        #User 1 CSI
        G_i=np.multiply(SigmaBar_i,G)+np.multiply(Sigma_i,Delta)
        G_hat.append(G_i)
    return (G,G_hat,sigma)

def buildDataCont(n,randomness,flip):
    if(not randomness):
        np.random.seed(1)
    #Generate random CSI quality levels
    sigma =np.random.rand(n,K)
    for i in range(n):
        sigma[i, :] = np.sort(sigma[i, :])
    if(flip):
        sigma = np.fliplr(sigma)
    #Number of training samples to be generated
    #Noise model of "Team Deep Neural Networks for Interference Channels"
    G=np.square(np.random.normal(0, 1, (n, K,K)))+np.square(np.random.normal(0, 1, (n, K,K))) #True channel matrix
    G_hat=[]
    #Noise variance
    #User 1 CSI noise variance
    for i in range(K):
        Sigma_i = np.ones((n, K, K)) * sigma[:,i,np.newaxis,np.newaxis]
        OneMat=np.ones((n, K, K))
        SigmaBar_i=np.sqrt(OneMat-np.square(Sigma_i))
        Delta=np.square(np.random.normal(0, 1, (n, K,K)))+np.square(np.random.normal(0, 1, (n, K,K)))
        #User 1 CSI
        G_i=np.multiply(SigmaBar_i,G)+np.multiply(Sigma_i,Delta)
        G_hat.append(G_i)
    sigma = sigma
    return (G,G_hat,sigma)


def buildDataSigma(n,sigmaVal):
    #Generate random
    np.random.seed(1)
    sigma =np.ones((n,K))
    for i in range(K):
        sigma[:,i]=sigma[:,i]*sigmaVal[i]
    #Number of training samples to be generated
    #Noise model of "Team Deep Neural Networks for Interference Channels"
    G=np.square(np.random.normal(0, 1, (n, K,K)))+np.square(np.random.normal(0, 1, (n, K,K))) #True channel matrix
    G_hat=[]
    #Noise variance
    #User 1 CSI noise variance
    for i in range(K):
        Sigma_i = np.ones((n, K, K)) * sigma[:,i,np.newaxis,np.newaxis]
        OneMat=np.ones((n, K, K))
        SigmaBar_i=np.sqrt(OneMat-np.square(Sigma_i))
        Delta=np.square(np.random.normal(0, 1, (n, K,K)))+np.square(np.random.normal(0, 1, (n, K,K)))
        #User 1 CSI
        G_i=np.multiply(SigmaBar_i,G)+np.multiply(Sigma_i,Delta)
        G_hat.append(G_i)
    sigma=sigma+np.random.normal(0,0,(n,K))
    return (G,G_hat,sigma)

def dataLabeling(G):
    p_opt = np.zeros([G.shape[0], K])
    for k in range(G.shape[0]):
        p_opt[k, :] = np.squeeze(NP_WMMSE_power_control(G[k, :, :], P))
    return p_opt

def NP_WMMSE_power_control(H_NP, P_NP):
    # H: H[i,j] means |h_{i,j}| which is absolute of channel state from Tx i to Rx j
    # note that |h|^2 is the channel gain
    # var_noise : noise power
    # Pmax : maximum transmit power
    vnew = 0
    var_noise = 1.0
    b = np.sqrt(P_NP) * np.ones((K, 1))  # initialize
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = np.abs(H_NP[i, i]) * b[i] / (np.matmul(np.square(np.abs(H_NP[i, :])), np.square(b)) + var_noise)
        w[i] = 1 / (1 - f[i] * b[i] * np.abs(H_NP[i, i]))
        vnew = vnew + np.log(w[i])

    VV = np.zeros(100)  # within 100 iteration solution usually converges
    for iter in range(100):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * np.abs(H_NP[i, i]) / sum(w * np.square(f) * np.square(np.abs(H_NP[:, i])))
            b[i] = np.minimum(btmp, np.sqrt(P_NP)) + np.maximum(btmp, 0) - btmp
        vnew = 0

        for i in range(K):
            f[i] = np.abs(H_NP[i, i]) * b[i] / (np.matmul(np.square(np.abs(H_NP[i, :])), np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * np.abs(H_NP[i, i]) + 1e-12)
            vnew = vnew + np.log(w[i])

        VV[iter] = vnew
        if vnew - vold <= 1e-3:  # break the loop if the solution is converged
            break
    p_opt = np.square(b)
    return p_opt