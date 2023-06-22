#!/usr/bin/env python
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sys as sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

alg=sys.argv[1]
sig=float(sys.argv[2])

# fiducial H0 value
h0 = 67.36

def calc_mean(x):
    return np.mean(x)

def calc_std(x):
    return np.std(x)

#mse definition: (x-mu)**2 + sig**2.
def calc_mse(x,mu,sig):
    return (x-mu)**2 + sig**2.

meanres_arr = []

nz_values = [20]
#nz_values = [20,30,50,80]

# =========== reading reconstruction results for GP and sklearn algorithms
for nz in nz_values:

    # for sighz/hz values in this range (supports sighz/hz = 0.008)
    if sig < 0.01 and sig > 0:

        # reading GP reconstruction, using GaPP
        if  alg=='GAP_SqExp' or alg=='GAP_Mat32' or alg=='GAP_Mat52' or alg=='GAP_Mat92':
            file_name = 'results_GP_sigh0p00'+str(int(sig*1000))+'_'+str(nz)+'pts_'+alg+'_GCV'
            (hz,sighz) = np.loadtxt(file_name+'.dat',unpack='true')
            # calculating stats for the first H(z) reconstruction bin, i.e., H(z=0) - can be adapted for other redshifts
            meanres_arr.append([nz, calc_mean(hz[0]), calc_std(hz[0]), calc_mse(calc_mean(hz[0]),h0,calc_std(hz[0]))])

        # reading sklearn reconstruction, using "alg" (as in alg=ANN, SVM, GBR, EXT)
        else:
            file_name = 'results_sklearn_sigh0p0'+str(int(sig*1000))+'_'+str(nz)+'pts_'+alg+'_GCV'
            (hza,hzb,hzc,hzd,hze,hzf,hzg,hzh,hzi,hzj,hzk,sctrain,sctest) = np.loadtxt(file_name+'.dat',unpack='true')
            # calculating stats for the first H(z) reconstruction bin, i.e., H(z=0) - can be adapted for other redshifts
            meanres_arr.append([nz, calc_mean(hza), calc_std(hza), calc_mse(calc_mean(hza),h0,calc_std(hza))])


    # for sighz/hz values in this range  (supports sighz/hz = 0.01, 0.03, 0.05, 0.08)
    if sig >= 0.01:

        if  alg=='GAP_SqExp' or alg=='GAP_Mat32' or alg=='GAP_Mat52' or alg=='GAP_Mat92':
            file_name = 'results_GP_sigh0p00'+str(int(sig*100))+'_'+str(nz)+'pts_'+alg+'_GCV'
            (hz,sighz) = np.loadtxt(file_name+'.dat',unpack='true')
            # calculating stats for the first H(z) reconstruction bin, i.e., H(z=0) - can be adapted for other redshifts
            meanres_arr.append([nz, calc_mean(hz[0]), calc_std(hz[0]), calc_mse(calc_mean(hz[0]),h0,calc_std(hz[0]))])
        else:
            file_name = 'results_sklearn_sigh0p0'+str(int(sig*100))+'_'+str(nz)+'pts_'+alg+'_GCV'
            (hza,hzb,hzc,hzd,hze,hzf,hzg,hzh,hzi,hzj,hzk,sctrain,sctest) = np.loadtxt(file_name+'.dat',unpack='true')
            # calculating stats for the first H(z) reconstruction bin, i.e., H(z=0) - can be adapted for other redshifts
            meanres_arr.append([nz, calc_mean(hza), calc_std(hza), calc_mse(calc_mean(hza),h0,calc_std(hza))])

    # for sighz/hz values as in the real H(z) data
    if sig == -1:

        if  alg=='GAP_SqExp' or alg=='GAP_Mat32' or alg=='GAP_Mat52' or alg=='GAP_Mat92':
            file_name = 'results_GP_sighz_31pts_'+alg+'_GCV'
            (hz,sighz) = np.loadtxt(file_name+'.dat',unpack='true')
            # calculating stats for the first H(z) reconstruction bin, i.e., H(z=0) - can be adapted for other redshifts
            meanres_arr.append([nz, calc_mean(hz[0]), calc_std(hz[0]), calc_mse(calc_mean(hz[0]),h0,calc_std(hz[0]))])
        else:
            file_name = 'results_sklearn_sighz_31pts_'+alg+'_GCV'
            (hza,hzb,hzc,hzd,hze,hzf,hzg,hzh,hzi,hzj,hzk,sctrain,sctest) = np.loadtxt(file_name+'.dat',unpack='true')
            # calculating stats for the first H(z) reconstruction bin, i.e., H(z=0) - can be adapted for other redshifts
            meanres_arr.append([nz, calc_mean(hza), calc_std(hza), calc_mse(calc_mean(hza),h0,calc_std(hza))])

# appending H0 stats in an array
meanres_arr = np.array(meanres_arr)

# saving results
if sig < 0.01 and sig > 0:
    np.savetxt('mean_results_sigh0p00'+str(int(sig*1000))+'_'+str(alg)+'.dat', meanres_arr)
if sig >= 0.01:
    np.savetxt('mean_results_sigh0p0'+str(int(sig*100))+'_'+str(alg)+'.dat', meanres_arr)
if sig == -1:
    np.savetxt('mean_results_sighz_'+str(alg)+'.dat', meanres_arr)
