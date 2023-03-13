'''' 
Microbial growth model for PC8R7N2.1 
including inhibition dynamics based on Haldane's equation
(greater yield)
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from lmfit import Parameters, fit_report, minimize


#tx = np.array([0, 12, 24, 30, 36, 48, 54, 60, 65, 69, 72, 78, 80, 84, 99.5, 105.5, 108, 121, 128.5, 132, 144])
#Xy = np.array([0.88, 0.88, 0.85, 0.83, 0.81, 0.82, 0.88, 0.99, 1.55, 2.48, 3.66, 4.01, 5.40, 5.63, 12.74, 10.62, 11.59, 14.06, 14.67, 23.77, 22.07])

txb4 = np.array([ 60, 65, 69, 72, 78, 80, 84, 99.5, 105.5, 108, 121, 128.5, 132, 144])
tx = txb4-54
Xy = np.array([ 0.99, 1.55, 2.48, 3.66, 4.01, 5.40, 5.63, 12.74, 10.62, 11.59, 14.06, 14.67, 23.77, 22.07])
#Sy = np.array([150.539, 144.564, 142.573, 135.602, 132.614, 126.141, 127.137, 123.154, 117.676, 114.689, 109.71, 100.747, 88.2988, 83.8174, 84.3154, 73.8589, 60.9129, 49.4606, 45.9751])
#Py = np.array([0, 1.83406, 4.58515, 7.64192, 9.47598, 17.1179, 19.2576, 17.4236, 26.8996, 30.262, 35.7642, 39.738, 43.4061, 48.6026, 51.3537, 56.2445, 57.7729, 60.8297, 68.7773])

X0 = 0.88 #0.04 #g/L
S0 = 10 #g/L # check glycerol
P0 = 0 #g/

umax = 0.18 #/h
Ks = 18.35 #19.8 #g/L
Yxs = 2 # 0.047 #0.71#0.0732

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
    #weight = [10, 10, 100, 50, 20, 20, 10, 1, 1, 10, 10, 19, 10, 10, 1, 1, 1]
    #weight = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    I = ((Xy - cX) ) ** 2 # * weight) ** 2
    return I


b0 = [X0, S0]
t = np.linspace(1e-5,tx[-1],151)

METHOD = 'Nelder'
result = minimize(regress, params, method=METHOD)
result.params.pretty_print()
#print(fit_report(result))

fit_data = 1

if(fit_data == 1):
    umax = result.params['umax'].value
    Ks = result.params['Ks'].value
    Yxs = result.params['Yxs'].value


g = odeint(MONOD, b0,t, args=(umax, Ks, Yxs))
cX = g[:,0]
cS = g[:,1]
print(len(tx))


# plt.figure()
# plt.plot(t,cX,'--g')
# #plt.plot(t,cXM,'--g')
# plt.plot(tx, Xy, 'o')
# plt.xlabel('Time (hours)')
# plt.ylabel('Biomass Concentration (g/L)')
# plt.legend(['Monod', 'Experimental'])


# plt.figure()
# plt.plot(t,cS,'--g')
# #plt.plot(t,cSM,'--g')
# #plt.plot(tx, Sy, 'o')
# plt.xlabel('Time (hours)')
# plt.ylabel('Glycerol Concentration (g/L)')
# #plt.legend(['Contois', 'Monod', 'Experimental'])

# plt.show()

#######################################################################################

# Plot inhibition curves

from inhibition import plot_inhibition_curves, haldane

xvline = 48
times = sorted( np.concatenate( ([xvline], np.linspace(1e-5, 215)) ) )
Kis = [1, 1.7, 2.2, 3]
args = (umax, Ks, Yxs)

g = odeint(MONOD, b0, times, args=args)
zero_inhib = g[:,0] # Biomass concentration
mic_name = 'PC8R7N2.1'

plot_inhibition_curves(
    times,
    b0,
    Kis,
    args,
    zero_inhib,
    haldane,
    mic_name,
    xvline
)



# Plot on all variables 
# Params 