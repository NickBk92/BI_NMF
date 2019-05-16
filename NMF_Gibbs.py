import numpy as np
import scipy.stats
from scipy.io import loadmat,savemat
import matplotlib
import scipy.stats
import os

#
ind = 0
vi = 1024
tau = 100



foldername = "Gibbs_(1010k510)_Is/"
 os.mkdir(foldername)
sqrt_error = np.zeros(12)
entropy_error = np.zeros(12)
KL_I = np.zeros(17)
for I in range(10,70,5):

    print("I = "+str(I))
    a_tm = 10 * np.ones((vi, I))
    b_tm = 10 * np.ones((vi, I))
    a_ve = 0.5 * np.ones((I, tau))
    b_ve = 10 * np.ones((I, tau))
    #Initialize
    T = np.random.gamma(a_tm, b_tm)
    V = np.random.gamma(a_ve, b_ve)
    faces = np.zeros((vi,tau))
    X = loadmat('faces32x400.mat')
    for k,v in X.items():
        if k == 'faces_new':
            for i in range(tau):
                for j in range(vi):
                    faces[j][i] = v[j][i]

    vi = np.shape(faces)[0]
    tau = np.shape(faces)[1]
    n=2000 #maxiter
    theta_I = []
    P=np.zeros((vi, I, tau))
    TV = np.dot(T,V)
    S = np.zeros((vi, I, tau))
    sqrt_error_I = np.zeros(n)
    entropy_error_I = np.zeros(n)

    for loop in range(n):
        print("....iter = "+str(loop))
        TV = np.dot(T,V)
        for t in range(tau):
            for v in range(vi):
                for i in range(I):
                    P[v,i,t] = T[v][i] * V[i][t] / TV[v][t]
                S[v,:,t] = np.random.multinomial(faces[v][t],P[v,:,t]/np.sum(P[v,:,t]))
        sigT = np.sum(S,axis=2)
        sigV = np.sum(S,axis=0)

        a_t = a_tm + sigT
        b_t = 1/(a_tm/b_tm + np.dot(np.ones((vi,1)),np.transpose(np.sum(V,axis=1)).reshape(1,I) ))
        T_temp = np.random.gamma(a_t, b_t)

        a_v = a_ve + sigV
        b_v = 1/(a_ve/b_ve + np.dot(np.transpose(np.sum(T,axis=0)).reshape(I,1),np.ones((1,tau))))
        V = np.random.gamma(a_v, b_v)
        T = T_temp

        TV = np.dot(T, V)
        # sqrt_error_I[loop] = np.sqrt(np.sqrt(np.sum(np.square(faces - TV))))
        # print("       sqrt error = "+str(sqrt_error_I[loop]))
        # entropy_error_I[loop] = np.sum(scipy.stats.entropy(faces, TV))
        # print("       entropy error = " + str(entropy_error_I[loop]))
    TV = np.dot(T, V)
    for v in range(vi):
        for t in range(tau):
            KL_I[ind] = KL_I[ind] + faces[v][t] * np.log(TV[v][t] / faces[v][t]) - TV[v][t] + faces[v][t]
    KL_I[ind] = - KL_I[ind]
    print("       KL divergance = " + str(KL_I[ind]))
    # theta_I.append([a_t,b_t,a_v,b_v])
    # savemat(foldername+'theta_I_'+str(I),{'theta':theta_I})
    sqrt_error[ind] = np.sqrt(np.sqrt(np.sum(np.square(faces-TV))))
    print("             sqrt error = " + str(sqrt_error[ind]))
    entropy_error[ind] = np.sum(scipy.stats.entropy(faces, TV))
    print("             entro error = " + str(entropy_error[ind]))
    ind += 1
savemat(foldername+'sqrt_error',{'sqrt_error':sqrt_error})
savemat(foldername+'entropy_error',{'entropy_error':entropy_error})
savemat(foldername+'KL_error',{'KL_error':KL_I})



