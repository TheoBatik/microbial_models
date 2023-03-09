'''' Was: 20210922 Znad Gluconic Acid Gluconic acid production by A. niger'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from lmfit import Parameters, fit_report, minimize


tx = np.array([0, 2, 5, 7, 8, 10, 12, 24, 29, 34, 48, 53, 58])
Xy = np.array([0.55, 0.56, 0.74, 0.99, 1.00, 1.53, 1.88, 3.41, 6.68, 7.88, 16.22, 16.95, 17.32])

X0 = Xy[0] #0.04 #g/L
S0 = 10 #g/L # check glycerol
P0 = 0 #g/

umax = 0.2 #/h
Ks = 18.35 #19.8 #g/L
Yxs = 0.212 # 0.047 #0.71#0.0732

params = Parameters()
params.add('umax', value= umax, min=0, vary=False)
params.add('Ks', value= Ks, min=0, vary=True)
params.add('Yxs', value= Yxs, min=0, vary=True)


def CONTOIS(f,t, umax, Ks, Yps, Yxs):
    X = f[0]
    S = f[1]
    P = f[2]

    u = umax*(S/(Ks*X+S))
    #u = umax*(S/(Ks+S))
    ddt0 = u*X #dXdt
    ddt1 = -ddt0/Yxs   #dSdt
    ddt2 = -ddt1*Yps

    ddt = [ddt0, ddt1, ddt2]
    return ddt

def MONOD(f, t, umax, Ks, Yxs):
    X = f[0]
    S = f[1]

    u = umax*(S/(Ks+S))
    ddt0 = u*X           # dXdt
    ddt1 = -ddt0/Yxs    # dSdt     strictly growth associated growth, cell maintenance

    ddt = [ddt0, ddt1]
    return ddt

def regress(params):
    umax = params['umax'].value
    Ks = params['Ks'].value
    Yxs = params['Yxs'].value

    c = odeint(MONOD, b0,tx, args=(umax, Ks, Yxs))
    cX = c[:, 0]
    I = (Xy - cX)**2
    # I = Py - cP
    #weight = [1, 1, 10, 10, 10, 10, 20, 20, 1, 1, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 10]
    weight = [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 5, 5]
    I = ((Xy - cX) * weight) ** 2
    return I

b0 = [X0, S0]
t = np.linspace(1e-5,tx[-1],151)

METHOD = 'Nelder'
result = minimize(regress,params, method=METHOD)
result.params.pretty_print()
#print(fit_report(result))

fit_data = 1

if(fit_data == 1):
    umax = result.params['umax'].value
    Ks = result.params['Ks'].value
    Yxs = result.params['Yxs'].value


g = odeint(MONOD,b0,t, args=(umax, Ks, Yxs))
cX = g[:,0]
cS = g[:,1]
print(len(tx))

plt.figure()
plt.plot(t,cX,'--r')
#plt.plot(t,cXM,'--g')
plt.plot(tx, Xy, 'o')
plt.xlabel('Time (hours)')
plt.ylabel('Biomass Concentration (g/L)')
plt.legend(['Monod', 'Experimental'])


plt.figure()
plt.plot(t,cS,'--r')
#plt.plot(t,cSM,'--g')
#plt.plot(tx, Sy, 'o')
plt.xlabel('Time (hours)')
plt.ylabel('Glycerol Concentration (g/L)')
#plt.legend(['Contois', 'Monod', 'Experimental'])

plt.show()

