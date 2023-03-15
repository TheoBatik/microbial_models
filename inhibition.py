import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import os

def load_csv( csv_name='measured_data' ):
    matrix = []
    with open( os.path.join('data', f'{csv_name}.csv'), 'r', encoding='utf-8-sig') as file:
        for i, line in enumerate(file):
            if i == 0:
                header = list( line.strip().split(',') )
                continue
            row = [float(item) for item in line.strip().split(',')]
            matrix.append(row)
    matrix = np.asarray( matrix )
    matrix.astype( np.float32 )
    return matrix, header


def haldane(f, t, umax, Ks, Yxs, ki):
    X = f[0]
    S = f[1]

    u = umax*(S/(Ks+S+(S**2)/ki))
    ddt0 = u*X           # dXdt
    ddt1 = -ddt0/Yxs    # dSdt     strictly growth associated growth, cell maintenance

    ddt = [ddt0, ddt1]
    return ddt


def haldane_with_products(f,t, Vmax, Km, Yps, Yxs, ki):
    X = f[0]
    S = f[1]
    P = f[2]

    u = Vmax * ( S / (Km + S + (S**2)/ki) )
    ddt0 = u * X  # dXdt
    ddt1 = -ddt0 / Yxs  # dSdt     strictly growth associated growth, cell maintenance
    ddt2 = (-ddt1) * Yps  # dPdt

    ddt = [ddt0, ddt1, ddt2]
    return ddt


def plot_inhibition_curves(
    eval_times, # model evaluation times
    b0, # initial conditions
    inhibs, # list of inhibition constant values
    args, # (limiting rate constant, Michaelis constant, substrate consumption factor,... )
    zero_inhib, # conc. for zero inhibition (e.g. from MONOD)
    inhib_func,
    mic_name, # name of micro-organism
    xvline,
    save_fig=True,
    show_fig=True,
    cells=True, # toggle between mass and cell number concentration (cells = order of magnitude of cell number)
    measured_data=None,
    measured_times=None
    ):


    # Create figure
    plt.figure()

    # Plot inhibited growth curves (for each item in kis)
    for ki in inhibs:
        
        g = odeint(inhib_func, b0, eval_times, args=args + (ki,))
        cX = g[:,0] # Biomass concentration
        label='$K_i = $' + str(ki)

        plt.plot(
            eval_times,
            cX,
            '-',
            label='$K_i = $' + str(round( ki, 5 ) )
            )

    if cells:
        ylabel = 'Biomass Concentration (cells/L)'
    else:
        ylabel = 'Biomass Concentration (g/L)'

            # scale = 10**(cells) 
            # cX /= scale
            # zero_inhib /= scale
            # cells_s = str(cells)
            # if len(cells_s) > 1:
            #     ylabel = f'Biomass Concentration ($10^{cells_s[0]}$' + f'$^{cells_s[1]}$ cells/L)'
            # else:
            #     ylabel = f'Biomass Concentration ($10^{cells_s}$ cells/L)'

            # if measured_data is not None:
            #     measured_data /= scale
    
    # Plot curve for ZERO inhibition (pure Monod dynamics)
    plt.plot(
        eval_times,
        zero_inhib,
        '-',
        label='No Inhibition'
        )

    # Plot measured data
    if measured_data is not None:

        plt.plot(
            measured_times,
            measured_data, 
            'o',
            label='Measured'
        )
        
    # Plot vertical line at 48 hours
    plt.axvline(x = xvline, linestyle = '--', color = '0.8')

    # Finalise
    plt.xlabel('Time (hours)')
    plt.ylabel( ylabel )
    title_start = 'Predicted biomass concentrations over time '
    title_mic_name = 'for ' + mic_name + ' '
    title_end = 'for various values of inhibition constant and in the absence of inhibition'
    title = title_start + title_mic_name + title_end
    plt.title( title, loc='center', wrap=True )
    plt.legend()
    if save_fig:
        save_at = 'plots/Inhibition curve for ' + mic_name.strip('.') + '.jpeg'
        plt.savefig( save_at, dpi=200 )
    if show_fig:
        plt.show()
    
