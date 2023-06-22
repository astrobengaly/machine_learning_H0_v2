#!/usr/bin/env python
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sys as sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

# H(z) as given by flat-LCDM model
def hz_model(z, h, om):
    return h*np.sqrt(om*(1.+z)**3. + (1.-om))

# number of MC simulations to be generated
nsim=int(sys.argv[1])

# sighz/hz value
sig=float(sys.argv[2])

# number of data points Nz
nz=int(sys.argv[3])

# minimum and maximum redshift of the H(z) simulations
zmin=float(sys.argv[4])
zmax=float(sys.argv[5])

# looping through nsim
for n in range(nsim):

    #input fiducial Cosmology (P18 best-fit for TT,TE,EE+lowE+lensing)
    om = 0.3166
    sigom = 0.0084
    h = 67.36
    sigh = 0.54

    z_arr=zmin+(zmax-zmin)*np.arange(nz)/(nz-1.0)

    # H(z) values according to the fiducial Cosmology at each z assuming a Gaussian distribution centred on hz_model
    for i in range(nz):
        hz_arr = np.array([hz_model(z,om,h) + sig*hz_model(z,om,h)*np.random.randn() for z in z_arr])
        sighz_arr = hz_arr*sig
        
        # displaying results (optional)
        #print(sig, n, z_arr[i], hz_arr[i]/hz_model(z_arr[i],om,h))

    # saving the simulated hz results in a text file
    if sig < 0.01:
        filename = 'input/sigh0p00'+str(int(sig*1000))+'/hz_sim_sigh0p00'+str(int(sig*1000))+'_'+str(nz)+'pts_h0p'+str(int(h*100))+'_zmin'+str(int(zmin*100))+'_zmax'+str(int(zmax*100))+'_mc#'+str(n+1)+'.dat'
    else:
        filename = 'input/sigh0p0'+str(int(sig*100))+'/hz_sim_sigh0p0'+str(int(sig*100))+'_'+str(nz)+'pts_h0p'+str(int(h*100))+'_zmin'+str(int(zmin*100))+'_zmax'+str(int(zmax*100))+'_mc#'+str(n+1)+'.dat'
    
    np.savetxt(filename, np.transpose([z_arr, hz_arr, sighz_arr]))
