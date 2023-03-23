'''
Microbial growth model for F. Caldus
including inhibition dynamics based on Haldane's equation.


Experimental data for initial fitting from LNU
    Average of 3 sets taken for fitting value, plotted all 3 sets
    Was: 20220905 Caldus Nathan Data

At. c growth experiment. At. c in MAC media, pH2.0; 1% biologically derived S, 40 degress C, 120 RPM.
Performed in triplicate with negative (cell-free) control. Cell counts are an average of 4 counts.
Assume all Sulfur available for At. caldus

Can fit to Michaelis Menten (MM) or Monod kinetics
Assume 60% of provided sulfur is available for oxidation --> Can also change as S0mass line

Error bars determined and shown in ExtractedData excel file

Molar masses of the acids produced by F. Caldus:
    Acetic acid: 60.05 g/mol
    Citric acid: 192.12 g/mol
    Fumaric acid: 116.07 g/mol
    Isocitric acid: 191.09 g/mol
    Malic acid: 134.09 g/mol
    Oxalic acid: 90.03 g/mol
    Pyruvic acid: 88.06 g/mol
    Succinic acid: 118.09 g/mol
'''

import matplotlib.pyplot as plt
import numpy as np
from numpy.matlib import repmat
from scipy.integrate import odeint, simps
from lmfit import Parameters, minimize, fit_report
import pandas as pd
from control import fit_report_toggle
from scipy.optimize import basinhopping

mic_name = 'F. Caldus'
print( '\n'*2, 'Summary of params used for species ', mic_name)

# acid_molar_masses = np.asarray([
#     60.05,
#     192.12,
#     116.07,
#     # 191.09,
#     # 134.09,
#     # 90.03,
#     # 88.06,
#     # 118.09 
# ])

# ave_acid_mm = np.average(acid_molar_masses)
# print(ave_acid_mm)

df = pd.read_excel(r'data/CaldusReadInData.xlsx', sheet_name='CaldusExperiment')
alldata = np.array(df)

num_elements = len(alldata)-1
Dtdays = alldata[:, 0] # days
Dt = Dtdays * 24    # h
D1biomass = alldata[:, 1]
D2biomass = alldata[:, 2]
D3biomass = alldata[:, 3]
DbiomassAve = alldata[:, 4]

# Convert biomass from cells/L to g/L
DbiomassAve = DbiomassAve / 1e11
DbiomassAveNorm = np.linalg.norm(DbiomassAve)

D1pH = alldata[:, 5]
D2pH = alldata[:, 6]
D3pH = alldata[:, 7]
DpHAve = alldata[:, 8]


# Constants
mwSurlfuricAcid = 98 #g/mol
mwSulfur = 32 # g/mol


def pH_to_concH(pH):
    cH = 10 ** (-pH)
    return cH

def pH_to_AcidConc(pH):
    cH = 10 ** (-pH)
    molP = cH / 2
    #cP = molP * mwSurlfuricAcid
    return molP

def sulfateconc_to_pH(cP):
    pH = -np.log10(cP*2)
    return pH

DP1 = pH_to_AcidConc(D1pH)
DP2 = pH_to_AcidConc(D2pH)
DP3 = pH_to_AcidConc(D3pH)
DPAve = pH_to_AcidConc(DpHAve)
DPAveNorm = np.linalg.norm(DPAve)



# Initial Conditions
Vr = 0.250        # L
S0mass = 10*0.6
S0 = S0mass/mwSulfur
print(S0)
X0 = DbiomassAve[0] # cells/L
P0 = DPAve[0]   # g/L
b0 = [X0, S0, P0]
print(b0)

Km = 15.5 #  g/L
Vmax = 0.01 # mol S/L/h
Yps = 0.6 # mol/mol
Yxs = 1 # 5.5e13 # cells/mol

params = Parameters()
params.add('Km', value=Km, min=0,  max=500, vary=True) 
params.add('Vmax', value=Vmax, min=0, vary=True) # 
params.add('Yps', value=Yps, min=0.01,  vary=True) # max=5,
params.add('Yxs', value=Yxs, min=0, vary=True)

def MM(f,t, Vmax, Km, Yps, Yxs):
    X = f[0]
    S = f[1]
    P = f[2]

    rs = (Vmax*S)/(Km*X+S)
    ddt1 = -1 * rs # dSdt
    ddt0 = Yxs * rs  # dXdt
    ddt2 = Yps * rs # dPdt

    ddt = [ddt0, ddt1, ddt2]
    return ddt


def Monod(f,t, Vmax, Km, Yps, Yxs):
    X = f[0]
    S = f[1]
    P = f[2]

    u = Vmax * (S / (Km + S))
    ddt0 = u * X  # dXdt
    ddt1 = -ddt0 / Yxs  # dSdt     strictly growth associated growth, cell maintenance
    ddt2 = (-ddt1) * Yps  # dPdt

    ddt = [ddt0, ddt1, ddt2]
    return ddt

def regress(params):
    Vmax = params['Vmax'].value
    Km = params['Km'].value
    Yps = params['Yps'].value
    Yxs = params['Yxs'].value

    #c = odeint(MM, b0,Dt, args=(Vmax, Km, Yps, Yxs))
    c = odeint(Monod, b0, Dt, args=(Vmax, Km, Yps, Yxs))
    cX = np.linalg.norm(c[:, 0])
    cP = np.linalg.norm(c[:, 2])

    # weight = [1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 20, 20, 8, 8, 8, 3, 3, 3, 3, 3, 3]
    I = ((DbiomassAveNorm - cX))**2 + ((DPAveNorm - cP))**2 
    #I = (DbiomassAve - cX) **2
    #I = ((DPAve - cP) * weight) ** 2

    return I



# Fit using basinhopping (global + local)

def regress_bh(params):
    Vmax = params[0]
    Km = params[1]
    Yps = params[2]
    Yxs = params[3]

    #c = odeint(MM, b0,Dt, args=(Vmax, Km, Yps, Yxs))
    c = odeint(Monod, b0, Dt, args=(Vmax, Km, Yps, Yxs))
    cX = np.linalg.norm(c[:, 0])
    cP = np.linalg.norm(c[:, 2])

    # weight = [1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 20, 20, 8, 8, 8, 3, 3, 3, 3, 3, 3]
    I = ((DbiomassAveNorm - cX))**2 + ((DPAveNorm - cP))**2 
    #I = (DbiomassAve - cX) **2
    #I = ((DPAve - cP) * weight) ** 2

    return I

params_bh = [
    params['Vmax'].value,
    params['Km'].value,
    params['Yps'].value,
    params['Yxs'].value
]

# print(params_bh)
# result_bh = basinhopping(regress_bh, params_bh, niter=100, T=10.0, stepsize=0.5, disp=True)
# print(result_bh)

# # Unpack params
# params = result_bh.x
# Vmax = params[0]
# Km = params[1]
# Yps = params[2]
# Yxs = params[3]

# print('Km used', Km)
# print('Vmax used', Vmax)
# print('Yps used', Yps)
# print('Yxs used', Yxs)



# Fit using minimize (local)

params = Parameters()
params.add('Km', value=Km, min=0,  max=500, vary=True) 
params.add('Vmax', value=Vmax, min=0, vary=True) # 
params.add('Yps', value=Yps, min=0.01,  vary=True) # max=5,
params.add('Yxs', value=Yxs, min=0, vary=True)


METHOD =  'nelder-mead' # 'least-sq' #
result = minimize(regress, params, method=METHOD)
result.params.pretty_print()
if fit_report_toggle:
    print(fit_report(result))
    print(result.residual)

Vmax = result.params['Vmax'].value
Km = result.params['Km'].value
Yxs = result.params['Yxs'].value
Yps = result.params['Yps'].value

# def pearson(a,b):
#     top = (a - np.average(a))*(b - np.average(b))
#     sumtop = np.sum(top)
#     bottom = np.sum((a - np.average(a))**2) * np.sum((b - np.average(b))**2)
#     sqrtbot = np.sqrt(bottom)
#     r = sumtop/sqrtbot
#     return r **2

# c = odeint(Monod, b0,Dt, args=(Vmax, Km, Yps, Yxs))
# cX = c[:, 0]
# cP = c[:, 2]
# r2X = pearson(DbiomassAve,cX)
# r2P = pearson(DPAve, cP)
# cpH = sulfateconc_to_pH(cP)
# r2pH = pearson(DpHAve, cpH)
# print('r2 pH', r2pH)

# print('r2 Biomass', r2X)
# print('r2 Product', r2P)


# t = np.linspace(1e-6,Dt[num_elements],100)
# t0 = [0,0]
# g = odeint(Monod,b0,t, args=(Vmax, Km, Yps, Yxs))
# #g = odeint(Monod, b0, t, args =(Vmax, Km, Yps, Yxs))
# cX = g[:,0]
# cS = g[:,1]
# cP = g[:,2]



# # plt.figure()
# # plt.subplot(3,1,1)
# # plt.plot(Dt, D1biomass, 'o')
# # plt.plot(Dt, D2biomass, 'o')
# # plt.plot(Dt, D3biomass, 'o')
# # plt.plot(Dt, DbiomassAve, 'o')
# # errorX = np.array([5.00E+09, 1.24E+11, 3.06E+11, 3.08E+12, 2.81E+12, 4.62E+12, 2.99E+12, 3.96E+12, 3.07E+12, 3.15E+12, 2.38E+12, 9.00E+11, 7.09E+11, 1.99E+12, 1.21E+12, 4.04E+11, 2.25E+12, 8.08E+11, 9.64E+11])/1e12
# # plt.errorbar(Dt, DbiomassAve/1e12, yerr=errorX, fmt='.k', ecolor='black', capsize=3, label='Experimental')
# # plt.plot(t, cX/1e12, '--g', label='Model')
# # plt.xlabel('Time (h)')
# # plt.ylabel('Biomass concentration ($x10^{12}$ cells/L)')
# # plt.legend(frameon=False)
# # # plt.savefig(r'C:\Users\ylrae\OneDrive\Pictures\Results\MonodX1_Biomassconc.png', dpi=150, format='png')
# # # plt.legend(['Sample 1', 'Sample 2', 'Sample 3', 'Average', 'Model'])

# # plt.figure()
# # #plt.plot(Dt, DP1, 'o')
# # #plt.plot(Dt, DP2, 'o')
# # #plt.plot(Dt, DP3, 'o')
# # plt.plot(Dt, DPAve, 'ok', label='Experimental')
# # plt.plot(t, cP, '--g', label='Model')
# # plt.legend(frameon=False)
# # plt.xlabel('Time (h)')
# # plt.ylabel('Sulfuric acid concentration (mol/L)')
# # # plt.savefig(r'C:\Users\ylrae\OneDrive\Pictures\Results\MonodX1_sulfuricacidconc.png', dpi=150, format='png')

# # plt.figure()
# # plt.plot(Dt, DpHAve, 'o')
# # plt.gray()
# # errorpH = np.array([0.005773503, 0.028867513, 0.036055513, 0.045825757, 0.035118846, 0.055075705, 0.05033223, 0.011547005, 0.056862407, 0.035118846, 0.037859389, 0.045825757, 0.026457513, 0.030550505, 0.032145503, 0.02081666, 0.02081666, 0.01, 0.015275252])
# # plt.errorbar(Dt, DpHAve, yerr=errorpH, fmt='.k', ecolor='black', capsize=3, label='Experimental') #, #fmt='o')
# # cpH = sulfateconc_to_pH(cP)
# # plt.plot(t, cpH, '--g', label='Model')
# # plt.legend(frameon=False)
# # plt.xlabel('Time (h)')
# # plt.ylabel('pH')
# # # plt.savefig(r'C:\Users\ylrae\OneDrive\Pictures\Results\MonodX1_pHconc.png', dpi=150, format='png')

# # plt.figure()
# # plt.plot(t, cS, '--g', label='Model')
# # plt.xlabel('Time (h)')
# # plt.ylabel('Sulfur concentration (mol/L)')
# # # plt.savefig(r'C:\Users\ylrae\OneDrive\Pictures\Results\MonodX1_sulfurconc.png', dpi=150, format='png')
# # plt.legend(['Sample 1', 'Sample 2', 'Sample 3', 'Average', 'Model'])

# # c = odeint(MM, b0,Dt, args=(Vmax, Km, Yps, Yxs))
# # #c = odeint(Monod, b0, Dt , args=(Vmax, Km, Yps, Yxs))
# # cX = c[:, 0]
# # cS = c[:, 1]
# # cP = c[:, 2]

# # plt.show()


##############################################################################

# Plot inhibition curves

from inhibition import plot_inhibition_curves, haldane_with_products
from control import show_fig

xvline = 216
times = sorted( np.concatenate( ([xvline], np.linspace(1e-5, Dt[-1], 1500)) ) )

Kis = np.asarray( [2, 3, 5, 10] ) / 1000 #np.logspace(-4, -2, 4) #(1, 200, 5) # [1, 1.7, 3]
args = (Vmax, Km, Yps, Yxs)

g = odeint(Monod, b0, times, args=args)
cX_no_inhib = g[:,0] # Biomass concentration
cS_no_inhib = g[:,1] # Substrate concentration
cP_no_inhib = g[:,2] # Product concentration

# Inhibition curves
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
    cX_measured=DbiomassAve,
    cP_measured=DPAve,
    # cells=True,
    # cS_measured=Sy,
    measurement_times=Dt,
    # cX_label_y='Biomass Concentration (cells/L)'
)
# Round the ki's => scientific nt

# Zero inhibition 
# times = sorted( np.concatenate( ([xvline], np.linspace(1e-5, 150, 1500)) ) )
plot_inhibition_curves(
    times,
    b0,
    [],
    args,
    haldane_with_products,
    mic_name,
    cX_no_inhib=cX_no_inhib,
    cS_no_inhib=cS_no_inhib,
    cP_no_inhib=cP_no_inhib,
    # xvline=xvline,
    show_fig=show_fig,
    cX_measured=DbiomassAve,
    cP_measured=DPAve,
    # cells=True,
    # cS_measured=Sy,
    measurement_times=Dt,
    # cX_label_y='Biomass Concentration (cells/L)'
)

print('Initial states (X, S, P)', b0)
print('Km used', Km)
print('Vmax used', Vmax)
print('Yps used', Yps)
print('Yxs used', Yxs)