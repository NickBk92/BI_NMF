import numpy as np
import scipy.stats
from scipy.io import loadmat,savemat
import matplotlib
import scipy.stats
import os
foldername = "EM_(k3101010)/"
 os.mkdir(foldername)

vi = 1024
tau = 50
I = 15
a_tm = 100 * np.ones((vi, I))
b_tm = 0.8* np.ones((vi, I))
a_ve = 0.5 * np.ones((I, tau))
b_ve = 5 * np.ones((I, tau))
T = np.random.gamma(a_tm, b_tm)
V = np.random.gamma(a_ve, b_ve)
faces = np.zeros((vi,tau))
X = loadmat('faces32x400.mat')
for k,v in X.items():
    if k == 'faces_new':
        for i in range(tau):
            for j in range(vi):
                faces[j][i] = v[j][i]

# faces = np.transpose(faces)
n=50
# estimate = np.dot(T,V)
sqrt_error_I = np.zeros(n)
entropy_error_I = np.zeros(n)
KL = np.zeros(n)

for itter in range(n):
    T_new = T.copy()
    # s3 = []
    # for i in range(I):
    #     s3.append(sum(V[i, :]))
    for v in range(vi):
        for i in range(I):
            s2 = 0
            for t in range(tau):
                s1 = 0
                for ip in range(I):
                    s1 += T[v,ip]*V[ip,t]
                s2 = s2 + (faces[v,t]*V[i,t])/s1
            s3 = sum(V[i,:])
            coef = s2/s3
            T_new[v,i] = 0.3*T[v,i]+0.7*T[v,i]*coef
            if T_new[v,i] > 255: T_new[v,i] = 125
    V_new = V.copy()

    for t in range(tau):
        for i in range(I):
            s2 = 0
            for v in range(vi):
                s1 = 0


                for ip in range(I):
                    s1 += T[v,ip]*V[ip,t]
                s2 += (faces[v,t]*T[v,i])/s1
            s3 = sum(T[i,:])
            coef = s2/s3
            V_new[i,t] = 0.6*V[i,t]+0.4*V[i,t]*coef
    T = T_new.copy()
    # V = V_new.copy()
    print(itter)
    TV = np.dot(T, V)
    sqrt_error_I[itter] = np.sqrt(np.sqrt(np.sum(np.square(faces - TV))))
    print("       sqrt error = " + str(sqrt_error_I[itter]))
    entropy_error_I[itter] = np.sum(scipy.stats.entropy(faces, TV))
    print("       entropy error = " + str(entropy_error_I[itter]))
    for v in range(vi):
        for t in range(tau):
            KL[itter] = KL[itter] + faces[v][t] * np.log(TV[v][t] / faces[v][t]) - TV[v][t] + faces[v][t]
    KL[itter] = - KL[itter]
    print("       KL divergance = " + str(KL[itter]))
savemat(foldername + 'EM_sqrt_error_I_' + str(I), {'EM_sqrt_error_I_' + str(I): sqrt_error_I})
savemat(foldername + 'EM_KL_Diverg_I_' + str(I), {'EM_KL_Diverg_I_' + str(I): KL})
savemat(foldername + 'EM_entropy_error_I_' + str(I), {'EM_entropy_error_I_' + str(I): entropy_error_I})
savemat(foldername+"EM_I_"+str(I),{"EM_T":T,"EM_V":V})



