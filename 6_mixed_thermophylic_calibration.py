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
from control import fit_report_toggle
from scipy.optimize import basinhopping

#########################################################################

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

# Convert biomass concentration (10^8 cells/ml) to (g/L)
states_m = ( states_m ) * 1e8 / 1e3 # (grams/cell) * (order of cells) / (ml/L)
state_names = 'Biomass Concentration (g/L)'
print('\nProcessed measured states')
print(state_names, '\n', states_m)

# Extract times at which to evalutate the solution of the ODE system
times_m =  measured_data[:, 0]
print('\nMeasurement times')
print(header[0], times_m)

# Set initial states
print('\nInitial measured states')
initial_states = [states_m[0], 5, 0] 
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
umax = 0.018 #0.01 #/h
Ks = 0.5336 #g/L
Yxs = 0.005064
Yps = 0.5445

params = Parameters()
params.add('umax', value= umax, min=0, vary=False)
params.add('Ks', value= Ks, min=0, vary=True)
params.add('Yxs', value= Yxs, min=0, vary=True)
params.add('Yps', value= Yxs, min=0, vary=True)


conversions = np.linspace(1e10, 1e11, 20)

for c in [1e20]: #conversions:
    
    save_folder = 'plots/test/' + '{:.2e}'.format(c) + ' '
    states_mc = states_m / c
    print( '\nstates_mc\n', states_mc )

    initial_states_c = [states_mc[0], 6, 0] 

    #######################################################################################

    # Define regression on norm
    norm_states_m = np.linalg.norm(states_mc)
    def regress( params ):

        # Unpack params
        umax = params['umax'].value
        Ks = params['Ks'].value
        Yps = params['Yps'].value
        Yxs = params['Yxs'].value

        # Model prediction
        c = odeint(monod, initial_states_c, times_m, args=(umax, Ks, Yps, Yxs))
        cX = np.linalg.norm(c[:, 0])
        # cS = c[:, 1]
        # cP = c[:, 2]
        del c

        # Compute error
        I = (norm_states_m - cX)**2

        return I

    #########################################################################

    # # Define regression
    # def regress( params ):

    #     # Unpack params
    #     umax = params['umax'].value
    #     Ks = params['Ks'].value
    #     Yps = params['Yps'].value
    #     Yxs = params['Yxs'].value

    #     # Model prediction
    #     c = odeint(monod, initial_states_c, times_m, args=(umax, Ks, Yps, Yxs))
    #     cX = c[:, 0]
    #     # cS = c[:, 1]
    #     # cP = c[:, 2]
    #     del c

    #     # Compute error
    #     I = (states_mc - cX)**2

    #     return I


##########################################################################

# # # Fit using basinhopping (global + local)
# # def regress_bh(params):
# #     Vmax = params[0]
# #     Km = params[1]
# #     Yps = params[2]
# #     Yxs = params[3]

# #     c = odeint(monod, initial_states_c, times_m, args=(Vmax, Km, Yps, Yxs))
# #     cX = c[:, 0] # np.linalg.norm(
# #     # print('states_measured =', states_m)
# #     # print('cX =', cX)
# #     discrepency = ((states_m - cX))
# #     I = np.sum( np.multiply( discrepency, discrepency ) )

# #     return I

# # params_bh = [
# #     params['umax'].value,
# #     params['Ks'].value,
# #     params['Yps'].value,
# #     params['Yxs'].value
# # ]

# # print(params_bh)

# # # Basinhopping!
# # b0 = (0, 0.2)
# # b1 = (50, 120)
# # b2 = (0, 100)
# # b3 = (0, 100)
# # bounds = [b0, b1, b2, b3]
# # minimizer_kwargs = {"bounds": bounds} # "method":"L-BFGS-B", }
# # result_bh = basinhopping(
# #     regress_bh, 
# #     params_bh, 
# #     minimizer_kwargs=minimizer_kwargs, 
# #     niter=1000, 
# #     T=10.0, 
# #     stepsize=20, 
# #     disp=True
# # )
# # print(result_bh)

# # # Unpack params
# # params = result_bh.x
# # umax = params[0]
# # Ks = params[1]
# # Yps = params[2]
# # Yxs = params[3]

# # print('Ks used', Ks)
# # print('umax used', umax)
# # print('Yps used', Yps)
# # print('Yxs used', Yxs)

# #######################################################################################

    # Fit model parameters locally to measured data

    # Minimise
    method = 'Nelder'
    result = minimize(regress, params, method=method)
    if fit_report_toggle:
        print(fit_report(result))
    result.params.pretty_print()

    # Redefine fitted model params
    umax = result.params['umax'].value
    Ks = result.params['Ks'].value
    Yxs = result.params['Yxs'].value
    Yps = result.params['Yps'].value

    #######################################################################################

    # Plot inhibition curves

    xvline = 24
    times_p = sorted( np.concatenate( ([xvline], np.linspace(1e-5, 1750, 600)) ) )
    # times_p = sorted( np.concatenate( ([xvline], np.linspace(1e-5, times_m[-1], 600)) ) )

    Kis = np.asarray([2, 3, 5, 10, ])
    args = (umax, Ks, Yps, Yxs)

    c_monod = odeint(monod, initial_states_c, times_p, args=args)
    cX_no_inhib = c_monod[:,0]  # Biomass concentration
    cS_no_inhib = c_monod[:,1] # Substrate concentration
    cP_no_inhib = c_monod[:,2] # Substrate concentration

    # plot_inhibition_curves(
    #     times_p,
    #     initial_states_c,
    #     Kis,
    #     args,
    #     haldane_with_products,
    #     mic_name,
    #     cX_no_inhib=cX_no_inhib,
    #     cS_no_inhib=cS_no_inhib,
    #     cP_no_inhib=cP_no_inhib,
    #     xvline=xvline,
    #     show_fig=show_fig,
    #     cX_measured=states_m,
    #     measurement_times=times_m,
    #     # cells=True,
    #     # scale_cX=None#1e8
    #     # cX_label_y='Biomass Concentration ($10^{8}$ cells/L)'
    # )


    # Plot zero inhib curves
    times_p = sorted( np.concatenate( ([xvline], np.linspace(1e-5, times_m[-1], 600)) ) )
    plot_inhibition_curves(
        times_p,
        initial_states_c,
        [],
        args,
        haldane_with_products,
        mic_name,
        cX_no_inhib=cX_no_inhib,
        # cS_no_inhib=cS_no_inhib,
        # cP_no_inhib=cP_no_inhib,
        xvline=xvline,
        show_fig=show_fig,
        cX_measured=states_mc,
        # cS_measured=[:,1],
        measurement_times=times_m,
        save_folder=save_folder,
        # cells=True,
        # scale_cX=None#1e8
        # cX_label_y='{:.2e}'.format(c)
    )