import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint



def haldane(f, t, umax, Ks, Yxs, ki):
    X = f[0]
    S = f[1]

    u = umax*(S/(Ks+S+(S**2)/ki))
    ddt0 = u*X           # dXdt
    ddt1 = -ddt0/Yxs    # dSdt     strictly growth associated growth, cell maintenance

    ddt = [ddt0, ddt1]
    return ddt


def plot_inhibition_curves(
    eval_times, # model evaluation times
    params, # see below
    zero_inhib, # function for zero inhibition (e.g. MONOD)
    ):

    # Unpack parameters
    kis = params['inhibs'] # list of inhibition constant values 
    umax = params['umax'] # limiting rate constant
    Ks = params['Ks'] # Michaelis constant
    Yxs = params['Yxs'] # substrate consumption factor
    b0 = params['b0'] # initial conditions 

    # Create figure
    plt.figure()

    # Plot inhibited growth curves (for each item in kis)
    for ki in kis:
        g = odeint(haldane, b0, eval_times, args=(umax, Ks, Yxs, ki))
        cX = g[:,0] # Biomass concentration
        plt.plot(
            eval_times,
            cX,
            '-',
            label='$K_i = $' + str(ki)
            )

    # Plot curve for ZERO inhibition (pure Monod dynamics)
    g = odeint(zero_inhib, b0, eval_times, args=(umax, Ks, Yxs))
    cX = g[:,0] # Biomass concentration
    plt.plot(
        eval_times,
        cX,
        '-',
        label='No Inhibition'
        )

    # Plot vertical line at 48 hours
    plt.axvline(x = 48, linestyle = '--', color = '0.8')

    # Finalise
    plt.xlabel('Time (hours)')
    plt.ylabel('Biomass Concentration (g/L)')
    plt.title('Predicted biomass concentrations over time for various values of inhibition constant and in the absence of inhibition')
    plt.legend()
    plt.show()
    
