'''' 
Microbial growth model for P. brassicacearum
including inhibition dynamics based on Haldane's equation
'''

mic_name = 'P. brassicacearum'
print( '\n'*2, 'Summary of params used for species ', mic_name)


# Imports
from inhibition import load_csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from lmfit import Parameters, fit_report, minimize
from inhibition import plot_inhibition_curves, haldane_with_products
from control import show_fig
from control import fit_report_toggle

#######################################################################################

# Import Wits dataset 3 to fit model parameters:
# Inlcude, biomass optimal density and Cyanide concentration over time
# Extract required variables from measured data and carry out conversion

# Load measure data
measured_data, header = load_csv( 'p_brassicacearum_wits_dataset_3')
print('\nRaw measured data')
print(measured_data)

# Extract states
states_m = measured_data[:, 1:3] # states measured
state_names = header[1:3]
print('\nRaw extracted states')
print(state_names, '\n', states_m)

# Convert optimal density to biomass concentration (g/L) and mg to g/L
conversion_factor_OD = 0.24
states_m[:, 0] = states_m[:, 0] * conversion_factor_OD
states_m[:, 1] = states_m[:, 1] / 1000 # * (1000 ml / L)
state_names[0] = 'Biomass Concentration (g/L)'
state_names[1] = 'Cyanide Concentration (g/L)'

# Extract times at which to evalutate the solution of the ODE system
times_m =  measured_data[:, 0]
print('\nMeasurement times')
print(header[0], times_m)

# Set initial states
innoculum_size_0 = 1.3e8
conversion_factor_IS = 10e-10 #  # grams/cell
cX_0 =  innoculum_size_0 * conversion_factor_IS
states_m[0, 0] = cX_0
print('\nProcessed measured states')
print(state_names, '\n', states_m)
print('\nInitial measured states')
initial_states = [ cX_0, 5, states_m[0, 1] ] # 5 g glycine
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

# Remove last three data points from the regression
weight = np.ones( len(times_m) )
for i in range(-3, 0):
    print(i)
    weight[i] = 0

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
    I = (states_m[:, 0] - cX)**2 + ((states_m[:, 1] - cP) * weight )**2

    return I


#######################################################################################

# Fit model parameters to measured data

# Minimise
method = 'Nelder'
result = minimize(regress, params, method=method)
result.params.pretty_print()
if fit_report_toggle:
    print(fit_report(result))

# Redefine fitted model params
umax = result.params['umax'].value
Ks = result.params['Ks'].value
Yxs = result.params['Yxs'].value
Yps = result.params['Yps'].value


#######################################################################################

# Plot inhibition curves

xvline = 24
times_p = sorted( np.concatenate( ([xvline], np.linspace(1e-5, 150, 400)) ) )
Kis = [2, 3, 5, 10]
args = (umax, Ks, Yps, Yxs)

c_monod = odeint(monod, initial_states, times_p, args=args)
cX_no_inhib = c_monod[:,0]  # Biomass concentration
cS_no_inhib = c_monod[:,1] # Substrate concentration
cP_no_inhib = c_monod[:,2] # Substrate concentration

plot_inhibition_curves(
    times_p,
    initial_states,
    Kis,
    args,
    haldane_with_products,
    mic_name,
    cX_no_inhib=cX_no_inhib,
    cS_no_inhib=cS_no_inhib,
    cP_no_inhib=cP_no_inhib,
    xvline=xvline,
    show_fig=show_fig,
    cX_measured=states_m[:,0],
    cP_measured=states_m[:,1],
    measurement_times=times_m,
    # cells=True,
    # scale_cX=None#1e8
    # cX_label_y='Biomass Concentration (cells/L)'
)


#######################################################################################


plot_inhibition_curves(
    times_p,
    initial_states,
    [],
    args,
    haldane_with_products,
    mic_name,
    cX_no_inhib=cX_no_inhib,
    cS_no_inhib=cS_no_inhib,
    cP_no_inhib=cP_no_inhib,
    xvline=xvline,
    show_fig=show_fig,
    cX_measured=states_m[:,0],
    cP_measured=states_m[:,1],
    measurement_times=times_m,
    # cells=True,
    # scale_cX=None#1e8
    # cX_label_y='Biomass Concentration (cells/L)'
)


#######################################################################################

# Ad-hoc 

# times = np.linspace(times_m[0], 85, 400) # times_m[-1]

# c_monod = odeint(monod, initial_states, times, args=args)
# zero_inhib = c_monod[:,0] # Biomass concentration
# plt.figure()
# plt.plot(
#     times_m,
#     states_m[:,0], 
#     'o',
#     label='Measured',
#     ms=5
# )
# plt.plot(
#     times,
#     zero_inhib, 
#     '-',
#     label='Predicted',
#     # linewidth=1
# )
# plt.xlabel('Time (hours)')
# plt.ylabel( 'Biomass Concentration (g/L)' )
# title = 'Biomass concentrations over time for ' + mic_name + ': comparison of measured data with Monod model prediction'
# plt.title( title, loc='center', wrap=True )
# plt.legend()
# if show_fig:
#     plt.show()


# c_monod = odeint(monod, initial_states, times, args=args)
# zero_inhib = c_monod[:,2] # Product concentration
# plt.figure()
# plt.plot(
#     times_m,
#     states_m[:,1], 
#     'o',
#     label='Measured',
#     ms=5
# )
# plt.plot(
#     times,
#     zero_inhib, 
#     '-',
#     label='Predicted',
#     # linewidth=1
# )
# plt.xlabel('Time (hours)')
# plt.ylabel( 'Product Concentration (g/L)' )
# title = 'Product concentrations over time for ' + mic_name + ': comparison of measured data with Monod model prediction'
# plt.title( title, loc='center', wrap=True )
# plt.legend()
# if show_fig:
#     plt.show()