'''' 
Microbial growth model for mixed thermophylic culture
including inhibition dynamics based on Haldane's equation
(greater yield)
'''

mic_name = 'mixed thermophylic culture'
print( '\n'*2, 'Summary of params used for species ', mic_name)


# Imports
from inhibition import load_csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from lmfit import Parameters, fit_report, minimize
from inhibition import plot_inhibition_curves, haldane_with_products
from control import show_fig

#######################################################################################

# Import raw data

# Measured data
measured_data, header = load_csv( 'mixed_thermophylic_biomass_over_time')
print('\nRaw measured data')
print(header)
print(measured_data)

# Extract states
states_m = measured_data[:, 1] # states measured
state_names = header[1]
print('\nRaw extracted states')
print(state_names, '\n', states_m)

# Convert biomass concentration (cells/ml) to (cells/L)
states_m = states_m / 1000
state_names = 'Biomass Concentration (cells/L)'
print('\nProcessed measured states')
print(state_names, '\n', states_m)

# Extract times at which to evalutate the solution of the ODE system
times_m =  measured_data[:, 0]
print('\nMeasurement times')
print(header[0], times_m)

# Set initial states
print('\nInitial measured states')
initial_states = [states_m[0], 20, 0] # 20g/L substrate in Luria broth + 5 g glycine
print(initial_states)


#######################################################################################

# Build model and define regression function

# Define model for parameter fitting 
def monod(f,t, umax, Ks, Yps, Yxs):
    X = f[0]
    S = f[1]
    P = f[2]

    u = umax * (S / (Ks + S))
    dXdt = u * X 
    dSdt = -dXdt / Yxs
    dPdt = (-dSdt) * Yps  

    dfdt = [dXdt, dSdt, dPdt]
    return dfdt


# Set model params
umax = 0.18 #/h
Ks = 18.35 # #g/L
Yxs = 2 
Yps = 0.5445

params = Parameters()
params.add('umax', value= umax, min=0, vary=True)
params.add('Ks', value= Ks, min=0, vary=True)
params.add('Yxs', value= Yxs, min=0, vary=True)
params.add('Yps', value= Yxs, min=0, vary=True)


# Define regression
def regress( params ):

    # Unpack params
    umax = params['umax'].value
    Ks = params['Ks'].value
    Yps = params['Yps'].value
    Yxs = params['Yxs'].value

    # Model prediction
    c = odeint(monod, initial_states, times_m, args=(umax, Ks, Yps, Yxs))
    cX = c[:, 0]
    # cS = c[:, 1]
    cP = c[:, 2]
    del c

    # Compute error
    I = (states_m - cX)**2

    return I


#######################################################################################

# Fit model parameters to measured data

# Minimise
method = 'Nelder'
result = minimize(regress, params, method=method)
result.params.pretty_print()

# Redefine fitted model params
umax = result.params['umax'].value
Ks = result.params['Ks'].value
Yxs = result.params['Yxs'].value
Yps = result.params['Yps'].value

#######################################################################################

# Plot inhibition curves

xvline = 24
times_p = sorted( np.concatenate( ([xvline], np.linspace(1e-5, 285, 400)) ) )
Kis = np.asarray([2, 3, 5, 10, ]) * 1 / 10000
args = (umax, Ks, Yps, Yxs)

c_monod = odeint(monod, initial_states, times_p, args=args)
cX_no_inhib = c_monod[:,0]  # Biomass concentration
cS_no_inhib = c_monod[:,1] # Substrate concentration

plot_inhibition_curves(
    times_p,
    initial_states,
    Kis,
    args,
    haldane_with_products,
    mic_name,
    cX_no_inhib=cX_no_inhib,
    cS_no_inhib=cS_no_inhib,
    xvline=xvline,
    show_fig=show_fig,
    cX_measured=states_m[:,0],
    # cS_measured=[:,1],
    measurement_times=times_m,
    cells=True,
    scale_cX=None#1e8
)