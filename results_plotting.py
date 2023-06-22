#!/usr/bin/env python
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sys as sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# ML algorithm to be deployed
alg=sys.argv[1]

# gcv value (true or false, if GridSearchValue() was deployed or not)
gcv=sys.argv[2]

# parameter that will be plotted - H0, BVT...
par=sys.argv[3]

# sighz/hz values
sig1 = 0.008
sig2 = 0.01
sig3 = 0.03
sig4 = 0.05
sig5 = 0.08

nz1,h1,errh1,bvt1=np.loadtxt('mean_results_sigh0p00'+str(int(sig1*1000))+'_'+str(alg)+'.dat', unpack='True')
nz2,h2,errh2,bvt2=np.loadtxt('mean_results_sigh0p0'+str(int(sig2*100))+'_'+str(alg)+'.dat', unpack='True')
nz3,h3,errh3,bvt3=np.loadtxt('mean_results_sigh0p0'+str(int(sig3*100))+'_'+str(alg)+'.dat', unpack='True')
nz4,h4,errh4,bvt4=np.loadtxt('mean_results_sigh0p0'+str(int(sig4*100))+'_'+str(alg)+'.dat', unpack='True')
nz5,h5,errh5,bvt5=np.loadtxt('mean_results_sigh0p0'+str(int(sig5*100))+'_'+str(alg)+'.dat', unpack='True')

# LateX rendering text fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Create figure size in inches
fig, ax = plt.subplots(figsize = (13., 10.))

plot_title=alg

# Constructing H0 plot
if par=='H0':

    # Define axes
    ax.set_ylabel(r"$H^{\mathrm{pred}}_0$", fontsize=32)
    ax.set_xlabel(r"$N_z$", fontsize=32)
    ax.set_xlim(17., 90.)
    ax.set_ylim(20., 120.)
    for t in ax.get_xticklabels(): t.set_fontsize(27)
    for t in ax.get_yticklabels(): t.set_fontsize(27)

    # plotting H0 results for each sighz/hz and Nz cases
    plt.errorbar(nz1+0, h1, yerr=1.*errh1, fmt='>', color='#228B22')
    plt.errorbar(nz2+1, h2, yerr=1.*errh2, fmt='o', color='black')
    plt.errorbar(nz3+2, h3, yerr=1.*errh3, fmt='s', color='red')
    plt.errorbar(nz4+3, h4, yerr=1.*errh4, fmt='*', color='blue')
    plt.errorbar(nz5+4, h5, yerr=1.*errh5, fmt='.', color='magenta')
    plt.axhline(67.36, color='cyan')

    # plot title and legend
    plt.title(r"{}".format(plot_title), fontsize='27')
    plt.legend((r"$H^{\mathrm{fid}}_0$", "$\sigma_{H}/H=0.008$", "$\sigma_{H}/H=0.01$","$\sigma_{H}/H=0.03$","$\sigma_{H}/H=0.05$","$\sigma_{H}/H=0.08$"), loc='upper right', fontsize=24)
    #plt.show()

# Constructing BVT plot
if par=='bvt':
    
    # Define axes
    ax.set_ylabel(r"$\mathrm{BVT}$", fontsize=32)
    ax.set_xlabel(r"$N_z$", fontsize=32)
    ax.set_xlim(17., 90.)
    ax.set_ylim(5e-2, 5e4)
    plt.yscale('log')
    for t in ax.get_xticklabels(): t.set_fontsize(27)
    for t in ax.get_yticklabels(): t.set_fontsize(27)

    # plotting BVT results for each sighz/hz and Nz cases
    plt.plot(nz1, bvt1, '->', color='#228B22')
    plt.plot(nz2, bvt2, '-o', color='black')
    plt.plot(nz3, bvt3, '-s', color='red')
    plt.plot(nz4, bvt4, '-*', color='blue')
    plt.plot(nz5, bvt5, '-v', color='magenta')

    plt.title(r"{}".format(plot_title), fontsize='27')
    plt.legend((r"$\sigma_{H}/H=0.008$", "$\sigma_{H}/H=0.01$","$\sigma_{H}/H=0.03$","$\sigma_{H}/H=0.05$","$\sigma_{H}/H=0.08$"), loc='upper right', fontsize=24)

# Saving plot in a png file
fig_name = par+'_estimates_'+alg
fig.savefig(fig_name+".png")
