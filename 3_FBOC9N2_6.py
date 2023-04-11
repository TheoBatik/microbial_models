'''' 
Microbial growth model for FBOC9N2.6
including inhibition dynamics based on Haldane's equation
'''

##############################################################################

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from lmfit import Parameters, fit_report, minimize
from control import fit_report_toggle, show_fig
from inhibition import plot_inhibition_curves, haldane_with_products, monod

##############################################################################

mic_name = 'FBOC9N2.6'
print( '\n'*2, 'Summary of params used for species ', mic_name)

tx = np.array([0, 2, 5, 7, 8, 10, 12, 24, 29, 34, 48, 53, 58])
Xy = np.array([0.55, 0.56, 0.74, 0.99, 1.00, 1.53, 1.88, 3.41, 6.68, 7.88, 16.22, 16.95, 17.32])

X0 = Xy[0] #0.04 #g/L
S0 = 10 #g/L # check glycerol
P0 = 0 #g/
b0 = [X0, S0, P0]

umax = 0.2 #/h
Ks = 18.35 #19.8 #g/L
Yxs = 0.212 # 0.047 #0.71#0.0732
Yps = 0.6

params = Parameters()
params.add('umax', value= umax, min=0, vary=False)
params.add('Ks', value= Ks, min=0, vary=True)
params.add('Yxs', value= Yxs, min=0, vary=True)
params.add('Yps', value= Yps, min=0, vary=True)

##############################################################################

def regress(params):
    umax = params['umax'].value
    Ks = params['Ks'].value
    Yps = params['Yps'].value
    Yxs = params['Yxs'].value

    c = odeint(monod, b0, tx, args=(umax, Ks, Yps, Yxs))
    cX = c[:, 0]

    I = (Xy - cX)**2
    # I = Py - cP
    #weight = [1, 1, 10, 10, 10, 10, 20, 20, 1, 1, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 10]
    # weight = [1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 5, 5]
    # I = ((Xy - cX) * weight) ** 2
    
    return I


METHOD = 'Nelder'
result = minimize(regress, params, method=METHOD)
result.params.pretty_print()
if fit_report_toggle:
    print(fit_report(result))

fit_data = 1

if(fit_data == 1):
    umax = result.params['umax'].value
    Ks = result.params['Ks'].value
    Yxs = result.params['Yxs'].value
    Yps = result.params['Yps'].value



# plt.figure()
# plt.plot(t,cX,'--r')
# #plt.plot(t,cXM,'--g')
# plt.plot(tx, Xy, 'o')
# plt.xlabel('Time (hours)')
# plt.ylabel('Biomass Concentration (g/L)')
# plt.legend(['Monod', 'Experimental'])


# plt.figure()
# plt.plot(t,cS,'--r')
# #plt.plot(t,cSM,'--g')
# #plt.plot(tx, Sy, 'o')
# plt.xlabel('Time (hours)')
# plt.ylabel('Glycerol Concentration (g/L)')
# #plt.legend(['Contois', 'Monod', 'Experimental'])

# plt.show()

##############################################################################


# Plot inhibition curves

xvline = 48
times = sorted( np.concatenate( ([xvline], np.linspace(1e-5, 150)) ) )
Kis = [2.8, 3.18, 4.9]
args = (umax, Ks, Yps, Yxs)

g = odeint(monod, b0, times, args=args)
cX_no_inhib = g[:,0] # Biomass concentration
cS_no_inhib = g[:,1] # Substrate concentration
cP_no_inhib = g[:,2] # Product concentration

# Plot inhibition curves
plot_inhibition_curves(
    times,
    b0,
    Kis,
    args,
    haldane_with_products,
    mic_name,
    cX_no_inhib=cX_no_inhib,
    cS_no_inhib=cS_no_inhib,
    cP_no_inhib=cP_no_inhib,
    # xvline=xvline,
    show_fig=show_fig,
    cX_measured=Xy,
    # cS_measured=Sy,
    measurement_times=tx
)

# Plot zero inhibition curves
Kis = []
plot_inhibition_curves(
    times,
    b0,
    Kis,
    args,
    haldane_with_products,
    mic_name,
    cX_no_inhib=cX_no_inhib,
    cS_no_inhib=cS_no_inhib,
    cP_no_inhib=cP_no_inhib,
    # xvline=xvline,
    show_fig=show_fig,
    cX_measured=Xy,
    # cS_measured=Sy,
    measurement_times=tx
)


print( 'Initial states (X, S, P)', b0)
print('Ks used', Ks)
print('umax used', umax)
print('Yps used', Yps)
print('Yxs used', Yxs)