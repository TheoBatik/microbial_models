'''' 
Microbial growth model for A. Niger
including inhibition dynamics based on Haldane's equation
'''
#PC8R7N2.1
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from lmfit import Parameters, fit_report, minimize
from control import fit_report_toggle, show_fig
from inhibition import plot_inhibition_curves, haldane_with_products, monod

##############################################################################

mic_name = 'A. niger'
print( '\n'*2, 'Summary of params used for species ', mic_name)

X0 = 0.88 #0.04 #g/L
S0 = 10 #g/L # check glycerol
P0 = 0 #g/
b0 = [X0, S0, P0]

tx = np.array([0, 12, 24, 30, 36, 48, 54, 60, 65, 69, 72, 78, 80, 84, 99.5, 105.5, 108, 121, 128.5, 132, 144])
Xy = np.array([0.88, 0.88, 0.85, 0.83, 0.81, 0.82, 0.88, 0.99, 1.55, 2.48, 3.66, 4.01, 5.40, 5.63, 12.74, 10.62, 11.59, 14.06, 14.67, 23.77, 22.07])

#txb4 = np.array([ 60, 65, 69, 72, 78, 80, 84, 99.5, 105.5, 108, 121, 128.5, 132, 144])
#tx = txb4-54
#Xy = np.array([ 0.99, 1.55, 2.48, 3.66, 4.01, 5.40, 5.63, 12.74, 10.62, 11.59, 14.06, 14.67, 23.77, 22.07])
#Sy = np.array([150.539, 144.564, 142.573, 135.602, 132.614, 126.141, 127.137, 123.154, 117.676, 114.689, 109.71, 100.747, 88.2988, 83.8174, 84.3154, 73.8589, 60.9129, 49.4606, 45.9751])
#Py = np.array([0, 1.83406, 4.58515, 7.64192, 9.47598, 17.1179, 19.2576, 17.4236, 26.8996, 30.262, 35.7642, 39.738, 43.4061, 48.6026, 51.3537, 56.2445, 57.7729, 60.8297, 68.7773])

umax = 0.18 #/h
Ks = 18.35 #19.8 #g/L
Yxs = 1.48 # 0.047 #0.71#0.0732
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

    c = odeint(monod, b0,tx, args=(umax, Ks, Yps, Yxs))
    cX = c[:, 0]
    I = (Xy - cX)**2
    # I = Py - cP
    # weight = [0, 0, 0, 1, 5, 5, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # #weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 19, 0, 0]
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
# plt.plot(t,cX,'--b')
# #plt.plot(t,cXM,'--g')
# plt.plot(tx, Xy, 'o')
# plt.xlabel('Time (days)')
# plt.ylabel('Biomass Concentration (g/L)')
# plt.legend(['Monod', 'Experimental'])

# plt.figure()
# plt.plot(t,cS,'--b')
# #plt.plot(t,cSM,'--g')
# #plt.plot(tx, Sy, 'o')
# plt.xlabel('Time (days)')
# plt.ylabel('Glucose Concentration (g/L)')
#plt.legend(['Contois', 'Monod', 'Experimental'])


# plt.show()


# Plot inhibition curves

xvline = 48
times = sorted( np.concatenate( ([xvline], np.linspace(1e-5, 500)) ) ) # end 500
Kis = [2, 3, 5, 10]
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
    xvline=xvline,
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