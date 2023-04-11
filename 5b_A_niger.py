'''' 
Microbial growth model for A. Niger
including inhibition dynamics based on Haldane's equation
'''

##############################################################################

mic_name = 'A. niger'
print( '\n'*2, 'Summary of params used for species ', mic_name)


# Imports
from inhibition import load_csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from lmfit import Parameters, fit_report, minimize
from inhibition import plot_inhibition_curves, haldane_3_products
from control import show_fig
from control import fit_report_toggle

#######################################################################################

# Import dataset to fit model parameters:
# Inlcude, biomass optimal density and Cyanide concentration over time
# Extract required variables from measured data and carry out conversion

# Load measure data
measured_data, header = load_csv( 'CETIM - A niger data 1')
print('\nRaw measured data')
print(header, measured_data)

# Extract states
states_m = measured_data[:, 1:4] # states measured
state_names = header[1:4]
print('\nRaw extracted states')
print(state_names, '\n', states_m)

# Extract times at which to evalutate the solution of the ODE system
times_m =  measured_data[:, 0]
print('\nMeasurement times')
print(header[0], times_m)

# Data cleaning
times_m = times_m[3:-1] - times_m[3]
states_m = states_m[3:-1,:]

# Set initial states
innoculum_size_0 = 1e5 #1.3e8
conversion_factor_IS = 1e-8 #  # grams/cell
cX_0 =  innoculum_size_0 * conversion_factor_IS
print('\nInitial measured states')
initial_states = [ cX_0, 25, *states_m[0,:] ] # 5 g glycine
print(initial_states)

# Data cleaning
# for ax in range(0,1):
#     states_m = np.delete( states_m, [1, 2], ax )
#     times_m = np.delete( times_m, [1, 2], ax )

#######################################################################################

# Build model and define regression function

# Define model for parameter fitting 
# def monod(f, t, umax, Ks, Yps, Yxs):
#     X = f[0]
#     S = f[1]
#     P = f[2]

#     u = umax * (S / (Ks + S))
#     dXdt = u * X 
#     dSdt = -dXdt / Yxs
#     dPdt = (-dSdt) * Yps  

#     dfdt = [dXdt, dSdt, dPdt]
#     return dfdt


def monod( f, t, *args ):
    ''' 
    System of differential equations for:
    1) Biomass production, x (Monod dynamics assumed)
    2) Substrate consumption, s
    3) Organic acid production, p
        pgl -> gluconic acid
        pox -> oxalic acid
        pci -> citric acid
    '''
    # Element-wise unpacking of vectorised solution, f 
    x = f[0]
    s = f[1]

    if s <= 0:
        return np.zeros(5)
    else:
        # Biomass production rate
        dxdt = args[0]*( s / (args[1] + s) ) * x

        # Substrate consumption rate
        dsdt = - args[2] * dxdt # - args[3] * x

        # Acid production rates
        dpdt = [ - args[i] * dsdt for i in [3, 4, 5] ]
        
        # Return ODE system
        return [dxdt, dsdt, *dpdt]


# Set model params
umax = 0.18 #/h
Ks = 62.24 # #g/L
Yxs = 8.51
Yps_gluc_1 = 0.003
# Yps_gluc_2 = 0.4
Yps_oxal_1 = 0.4
# Yps_oxal_2 = 0.2
Yps_citr_1 = 0.06
# Yps_citr_2 = 0.02

params = Parameters()
params.add(name='umax', value= umax, min=0, vary=False)
params.add(name='Ks', value= Ks, min=0, vary=False)
params.add(name='Yxs', value= Yxs, min=0, vary=True)
params.add(name='Yps_gluc_1', value=Yps_gluc_1, vary=True)
# params.add(name='Yps_gluc_2', value=Yps_gluc_2, min=0, vary=True)
params.add(name='Yps_oxal_1', value=Yps_oxal_1, min=0, vary=True)
# params.add(name='Yps_oxal_2', value=Yps_oxal_2, min=0, vary=True)
params.add(name='Yps_citr_1', value=Yps_citr_1, min=0, vary=True)
# params.add(name='Yps_citr_2', value=Yps_citr_2, min=0, vary=True)


# Define regression
def regress( params ):

    # Unpack params
    umax = params['umax'].value
    Ks = params['Ks'].value
    Yxs = params['Yxs'].value
    Yps_gluc_1 = params['Yps_gluc_1'].value
    # Yps_gluc_2 = params['Yps_gluc_2'].value
    Yps_oxal_1 = params['Yps_oxal_1'].value
    # Yps_oxal_2 = params['Yps_oxal_2'].value
    Yps_citr_1 = params['Yps_citr_1'].value
    # Yps_citr_2 = params['Yps_citr_2'].value
    args = ( umax, Ks, Yxs, Yps_gluc_1, Yps_oxal_1, Yps_citr_1 )

    # Model prediction
    c = odeint(monod, initial_states, times_m, args=args)
    cX = c[:, 0]
    # cS = c[:, 1]
    cP0 = c[:, -3] # Gluconic
    cP1 = c[:, -2] # Oxalic
    cP2 = c[:, -1] # Citric

    del c

    weight = [1, 1, 10000, 10000, 10000]
    # Compute error
    I = ( states_m[:, 0] - cP0 )**2 + ( states_m[:, 1] - cP1 )**2 + (( states_m[:, 2] - cP2)*weight )**2

    return I


# #######################################################################################

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
Yps_gluc_1 = params['Yps_gluc_1'].value
# Yps_gluc_2 = params['Yps_gluc_2'].value
Yps_oxal_1 = params['Yps_oxal_1'].value
# Yps_oxal_2 = params['Yps_oxal_2'].value
Yps_citr_1 = params['Yps_citr_1'].value
# Yps_citr_2 = params['Yps_citr_2'].value
# args = (umax, Ks, Yxs, Yps_gluc_1, Yps_gluc_2, Yps_oxal_1, Yps_oxal_2, Yps_citr_1, Yps_citr_2)
args = (umax, Ks, Yxs, Yps_gluc_1, Yps_oxal_1, Yps_citr_1)

#######################################################################################

# Plot inhibition curves

xvline = 24
times_p = sorted( np.concatenate( ([xvline], np.linspace(1e-5, 300, 400)) ) )
Kis = np.array( [12.2] ) # [2, 3, 5, 10])

c_monod = odeint(monod, initial_states, times_p, args=args)
cX_no_inhib = c_monod[:,0]  # Biomass concentration
cS_no_inhib = c_monod[:,1] # Substrate concentration
cP_no_inhib_1 = c_monod[:,2] # Product concentration
cP_no_inhib_2 = c_monod[:,3] # Product concentration
cP_no_inhib_3 = c_monod[:,4] # Product concentration


mic_name_1 = mic_name + ' (gluconic acid)'
mic_name_2 = mic_name + ' (oxalic acid)'
mic_name_3 = mic_name + ' (citric acid)'

# Plot biomass and sub. no inhibition curves
plot_inhibition_curves(
    times_p,
    initial_states,
    [],
    args,
    haldane_3_products,
    mic_name,
    cX_no_inhib=cX_no_inhib,
    cS_no_inhib=cS_no_inhib,
    # cP_no_inhib=cP_no_inhib_1,
    # xvline=xvline,
    show_fig=show_fig,
    # cX_measured=Xy,
    # cS_measured=Sy,
    # cP_measured=states_m[:,0],
    # measurement_times=times_m
)


# Plot product no inhibition curve 1
plot_inhibition_curves(
    times_p,
    initial_states,
    [],
    args,
    haldane_3_products,
    mic_name_1,
    # cX_no_inhib=cX_no_inhib,
    # cS_no_inhib=cS_no_inhib,
    cP_no_inhib=cP_no_inhib_1,
    # xvline=xvline,
    show_fig=show_fig,
    # cX_measured=Xy,
    # cS_measured=Sy,
    cP_measured=states_m[:,0],
    measurement_times=times_m,
    cP_index=2
)


# Plot product no inhibition curve 2
plot_inhibition_curves(
    times_p,
    initial_states,
    [],
    args,
    haldane_3_products,
    mic_name_2,
    # cX_no_inhib=cX_no_inhib,
    # cS_no_inhib=cS_no_inhib,
    cP_no_inhib=cP_no_inhib_2,
    # xvline=xvline,
    show_fig=show_fig,
    # cX_measured=Xy,
    # cS_measured=Sy,
    cP_measured=states_m[:,1],
    measurement_times=times_m,
    cP_index=3
)


# Plot product no inhibition curve 3
plot_inhibition_curves(
    times_p,
    initial_states,
    [],
    args,
    haldane_3_products,
    mic_name_3,
    # cX_no_inhib=cX_no_inhib,
    # cS_no_inhib=cS_no_inhib,
    cP_no_inhib=cP_no_inhib_3,
    # xvline=xvline,
    show_fig=show_fig,
    # cX_measured=Xy,
    # cS_measured=Sy,
    cP_measured=states_m[:,2],
    measurement_times=times_m,
    cP_index=4
)


#################################################################################


# Plot biomass and sub. inhibition curves
plot_inhibition_curves(
    times_p,
    initial_states,
    Kis,
    args,
    haldane_3_products,
    mic_name,
    cX_no_inhib=cX_no_inhib,
    cS_no_inhib=cS_no_inhib,
    # cP_no_inhib=cP_no_inhib_1,
    # xvline=xvline,
    show_fig=show_fig,
    # cX_measured=Xy,
    # cS_measured=Sy,
    # cP_measured=states_m[:,0],
    # measurement_times=times_m
)


# Plot product inhibition curve 1
plot_inhibition_curves(
    times_p,
    initial_states,
    Kis,
    args,
    haldane_3_products,
    mic_name_1,
    # cX_no_inhib=cX_no_inhib,
    # cS_no_inhib=cS_no_inhib,
    cP_no_inhib=cP_no_inhib_1,
    # xvline=xvline,
    show_fig=show_fig,
    # cX_measured=Xy,
    # cS_measured=Sy,
    cP_measured=states_m[:,0],
    measurement_times=times_m,
    cP_index=2
)


# Plot product inhibition curve 2
plot_inhibition_curves(
    times_p,
    initial_states,
    Kis,
    args,
    haldane_3_products,
    mic_name_2,
    # cX_no_inhib=cX_no_inhib,
    # cS_no_inhib=cS_no_inhib,
    cP_no_inhib=cP_no_inhib_2,
    # xvline=xvline,
    show_fig=show_fig,
    # cX_measured=Xy,
    # cS_measured=Sy,
    cP_measured=states_m[:,1],
    measurement_times=times_m,
    cP_index=3
)


# Plot product inhibition curve 3
plot_inhibition_curves(
    times_p,
    initial_states,
    Kis,
    args,
    haldane_3_products,
    mic_name_3,
    # cX_no_inhib=cX_no_inhib,
    # cS_no_inhib=cS_no_inhib,
    cP_no_inhib=cP_no_inhib_3,
    # xvline=xvline,
    show_fig=show_fig,
    # cX_measured=Xy,
    # cS_measured=Sy,
    cP_measured=states_m[:,2],
    measurement_times=times_m,
    cP_index=4
)