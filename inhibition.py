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

# Not in use
def contois(f,t, umax, Ks, Yps, Yxs):
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

# In use for not inhibition parameter fitting
def monod(f,t, Vmax, Km, Yps, Yxs):
    X = f[0]
    S = f[1]
    P = f[2]

    u = Vmax * (S / (Km + S))
    ddt0 = u * X  # dXdt
    ddt1 = -ddt0 / Yxs  # dSdt     strictly growth associated growth, cell maintenance
    ddt2 = (-ddt1) * Yps  # dPdt

    ddt = [ddt0, ddt1, ddt2]
    return ddt


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
    inhib_func,
    mic_name, # name of micro-organism
    cX_no_inhib=None, # conc. for biomass - zero inhibition (i.e. using Monod)
    cS_no_inhib=None, # conc. for substrate - zero inhibition
    cP_no_inhib=None, # conc. for product - zero inhibition
    cX_measured=None, # conc. for biomass - measured
    cS_measured=None, # conc. for substrate - measured
    cP_measured=None, # conc. for product - measured
    measurement_times=None,
    save_fig=True,
    show_fig=True,
    xvline=None,
    # cells=False, # toggle between mass and cell number concentration (cells = order of magnitude of cell number)
    scale_cX=None,
    plot_substrate=True,
    cX_label_y='Biomass Concentration (g/L)'
    ):

    ##################################################################################

    # Create figure for biomass concentrations
    plt.figure()
    ylabel=cX_label_y

    # Plot predicted biomass for various inhibition constants
    if cX_no_inhib is not None:

        # Plot biomass inhibited growth curves (for each item in kis)
        for ki in inhibs:
            
            g = odeint(inhib_func, b0, eval_times, args=args + (ki,))
            cX = g[:,0] # Biomass concentration
            
            # if scale_cX is not None:
            #     cX *= scale_cX
                # cX_no_inhib *= 100

            label='$K_i = $' + str(ki)

            plt.plot(
                eval_times,
                cX,
                '-',
                label='$K_i = $' + str(round( ki, 5 ) )
                )

        # if cells:
        #     ylabel = 'Biomass Concentration (cells/L)'
        # else:
        #     ylabel = 'Biomass Concentration (g/L)'

                # if len(cells_s) > 1:
                #     ylabel = f'Biomass Concentration ($10^{cells_s[0]}$' + f'$^{cells_s[1]}$ cells/L)'
                # else:
                #     ylabel = f'Biomass Concentration ($10^{cells_s}$ cells/L)'
                # scale = 10**(cells) 
                # cX /= scale
                # zero_inhib /= scale
                # cells_s = str(cells)

                # if measured_data is not None:
                #     measured_data /= scale
    
        # Plot biomass curve for ZERO inhibition (pure Monod dynamics)
        plt.plot(
            eval_times,
            cX_no_inhib,
            '-',
            label='No Inhibition (predicted)'
            )

    # Plot biomass measured data
    if cX_measured is not None:

        # if scale_cX is not None:
        #     cX_measured *= scale_cX

        plt.plot(
            measurement_times,
            cX_measured, 
            'o',
            ms=3.5,
            label='No inhibition (measured)'
        )
        
    # Plot vertical line at (xvline) hours
    if xvline is not None:
        plt.axvline(x = xvline, linestyle = '--', color = '0.8')

    # Finalise
    plt.xlabel('Time (hours)')
    plt.ylabel( cX_label_y )
    title_start = 'Predicted biomass concentrations over time '
    title_mic_name = 'for ' + mic_name + ' '
    title_end = 'for various values of inhibition constant and in the absence of inhibition'
    title = title_start + title_mic_name + title_end
    plt.title( title, loc='center', wrap=True )
    plt.legend()
    if save_fig:
        if len(inhibs) == 0:
            plot_name_base =  'Biomass curve for '
            plot_name_end = ' (no inhibition)'
        else:
            plot_name_base =  'Biomass inhibition curve for '
            plot_name_end = ''
        save_at = 'plots/' + plot_name_base + mic_name.strip('.') + plot_name_end + '.jpeg'
        plt.savefig( save_at, dpi=200 )
    if show_fig:
        plt.show()


    ##################################################################################

    # Create figure for substrate concentrations
    plt.figure()

    # Plot predicted substrate for various inhibition constants
    if cS_no_inhib is not None:

        # Plot substrate inhibited growth curves (for each item in kis)
        for ki in inhibs:
            
            g = odeint(inhib_func, b0, eval_times, args=args + (ki,))
            cS = g[:,1] # Substrate concentration
            label='$K_i = $' + str(ki)

            plt.plot(
                eval_times,
                cS,
                '-',
                label='$K_i = $' + str(round( ki, 5 ) )
            )

        # Plot substrate curve for ZERO inhibition (pure Monod dynamics)
        plt.plot(
            eval_times,
            cS_no_inhib,
            '-',
            label='No Inhibition (predicted)'
        )

    # Plot substrate measured data
    if cS_measured is not None:

        plt.plot(
            measurement_times,
            cS_measured, 
            'o',
            ms=3,
            label='No Inhibition (measured)'
        )
        
    # Plot vertical line at (xvline) hours
    if xvline is not None:
        plt.axvline(x = xvline, linestyle = '--', color = '0.8')

    # Finalise
    plt.xlabel('Time (hours)')
    plt.ylabel( 'Substrate Concentration (g/L)' )
    title_start = 'Predicted substrate concentrations over time '
    title_mic_name = 'for ' + mic_name + ' '
    title_end = 'for various values of inhibition constant and in the absence of inhibition'
    title = title_start + title_mic_name + title_end
    plt.title( title, loc='center', wrap=True )
    plt.legend()
    if save_fig:
        if len(inhibs) == 0:
            plot_name_base =  'Substrate curve for '
            plot_name_end = ' (no inhibition)'
        else:
            plot_name_base =  'Substrate inhibition curve for '
            plot_name_end = ''
        save_at = 'plots/' + plot_name_base + mic_name.strip('.') + plot_name_end + '.jpeg'
        plt.savefig( save_at, dpi=200 )
    if show_fig:
        plt.show()
    
    ##################################################################################

    # Plot predicted product for various inhibition constants
    if (cP_no_inhib is not None) and inhib_func == haldane_with_products:

        # Create figure for product concentrations
        plt.figure()

        # Plot product inhibited growth curves (for each item in kis)
        for ki in inhibs:
            
            g = odeint(inhib_func, b0, eval_times, args=args + (ki,))
            cP = g[:,2] # Product concentration
            label='$K_i = $' + str(ki)

            plt.plot(
                eval_times,
                cP,
                '-',
                label='$K_i = $' + str(round( ki, 5 ) )
            )

        # Plot product curve for ZERO inhibition (pure Monod dynamics)
        plt.plot(
            eval_times,
            cP_no_inhib,
            '-',
            label='No Inhibition (predicted)'
        )

        # Plot product measured data
        if cP_measured is not None:

            plt.plot(
                measurement_times,
                cP_measured, 
                'o',
                ms=3,
                label='No inhibition (measured)'
            )
        
        # Plot vertical line at (xvline) hours
        if xvline is not None:
            plt.axvline(x = xvline, linestyle = '--', color = '0.8')

        # Finalise
        plt.xlabel('Time (hours)')
        plt.ylabel( 'Product Concentration (g/L)' )
        title_start = 'Predicted product concentrations over time '
        title_mic_name = 'for ' + mic_name + ' '
        title_end = 'for various values of inhibition constant and in the absence of inhibition'
        title = title_start + title_mic_name + title_end
        plt.title( title, loc='center', wrap=True )
        plt.legend()
        if save_fig:
            if len(inhibs) == 0:
                plot_name_base =  'Product curve for '
                plot_name_end = ' (no inhibition)'
            else:
                plot_name_base =  'Product inhibition curve for '
                plot_name_end = ''
            save_at = 'plots/' + plot_name_base + mic_name.strip('.') + plot_name_end + '.jpeg'
            plt.savefig( save_at, dpi=200 )
        if show_fig:
            plt.show()