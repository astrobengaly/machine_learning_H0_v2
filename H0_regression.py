#!/usr/bin/env python
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import sys as sys

import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

# number of MC simulations to be generated
nsim=int(sys.argv[1])

# sighz/hz value
sig=float(sys.argv[2])

# number of data points Nz
nz=int(sys.argv[3])

# ML algorithm to be deployed
alg=(sys.argv[4])

# Hubble Constant fiducial value for the simulated data - follows the dimensionless notation, i.e., h = H0/(100 km/s/Mpc)
h = 67.36

# creating an array that will receive the reconstruction results
results_arr = []

# looping through nsim
for n in range(nsim):

    # loading simulated H(z) data
    if sig >= 0.01:
        dataset=pd.read_csv('input/sigh0p0'+str(int(sig*100))+'/hz_sim_sigh0p0'+str(int(sig*100))+'_'+str(nz)+'pts_mc#'+str(n+1)+'.dat', delim_whitespace=True)

    if sig < 0.01 and sig > 0:
        dataset=pd.read_csv('input/sigh0p00'+str(int(sig*1000))+'/hz_sim_sigh0p00'+str(int(sig*1000))+'_'+str(nz)+'pts_mc#'+str(n+1)+'.dat',delim_whitespace=True)

    # sig = -1 means the real data
    if sig == -1:
        dataset=pd.read_csv('input/sighz/hz_sim_sighz_31pts_mc#'+str(n+1)+'.dat', delim_whitespace=True)
        
    # associating z and hz with the first and second columns of the dataset, respectively
    z = dataset.iloc[:,0]
    hz = dataset.iloc[:,1]
    errhz = dataset.iloc[:,2]

    # reshaping both arrays to be processed by the regression algorithm later on
    z=z.values.reshape((len(z),1))
    hz=hz.values.reshape((len(hz),1))

    # splitting the hz dataset into training and testing set using sklearn
    test_size_value = 0.25
    z_train, z_test, hz_train, hz_test = train_test_split(z, hz, test_size=test_size_value, random_state=42)

    # regression algorithm - Support Vector Machine
    if alg=='SVM':
        
        gcv = GridSearchCV(SVR(kernel='poly', C=100, gamma='auto', epsilon=.1, coef0=1),
              param_grid={
              #'C': np.arange(1, 150, 10),
              'degree': np.arange(1, 10),
              #'epsilon': np.arange(1e-1, 1e0, 1e-1),
              },
              cv=3, refit=True)

    # regression algorithm - Decision Trees
    if alg=='EXT':
        
        gcv = GridSearchCV(ExtraTreesRegressor(min_samples_split=2, random_state=42),
              param_grid={
              'n_estimators': np.arange(1,100,2),
              'max_depth': np.arange(1,10,2),
              #'min_samples_split': np.arange(2,10,2),
              },
              cv=3, refit=True)
    
    # regression algorithm - Gradient Boosting
    if alg=='GBR':

        gcv = GridSearchCV(GradientBoostingRegressor(random_state=42),
              param_grid={
              'n_estimators': np.arange(1,200,5),
              },
              cv=3, refit=True)
    
    # regression algorithm - Artificial Neural Networks
    if alg=='ANN':
        
        gcv = GridSearchCV(MLPRegressor(activation='relu', solver='lbfgs', learning_rate='adaptive', max_iter=200, random_state=42),
              param_grid={
              'hidden_layer_sizes': np.arange(10,250,10),
              #'activation': ["identity", "logistic", "tanh", "relu"],
              #'max_iter': np.arange(100,1000,100)
              },
              cv=3, refit=True)

    ## regression algorithm - Lasso linear regression
    #if alg=='LAS':

        #a=0.01
        ##reg = linear_model.Lasso(alpha=a)

        #gcv = GridSearchCV(linear_model.Lasso(),
              #param_grid={
              #'alpha': np.logspace(1e-3, 1e0, 10),
              #},
              #cv=3, refit=True)

    ## regression algorithm - Random Forest
    #if alg=='RDF':

        #gcv = GridSearchCV(RandomForestRegressor(random_state=42),
              #param_grid={
              #'n_estimators': np.arange(20,500),
              ##'max_depth': np.arange(1,2),
              ##'min_samples_split': np.arange(2,5),
              #},
              #cv=3, refit=True)
    
    # performing the H(z) reconstruction with the given algorithm
    gcv.fit(z_train, hz_train.reshape((len(hz_train),)))

    # obtaining predictions with the test set
    gcv_hz_pred = gcv.best_estimator_.predict(z_test)

    # calculating cross-validation score values
    gcv_crossval_scores = cross_val_score(gcv, z_train, hz_train, cv=3)

    # defining the redshifts where the reconstruction will be performed (z_i is the initial redshift value, and z_bin is the redshift bin size)
    z_i = 0.
    z_bin = 0.2

    # printing the results of the H(z) reconstruction, along with the score values  (optional)
    print(n, gcv.best_estimator_.predict([[z_i]]), gcv.best_estimator_.predict([[z_i+z_bin]]), gcv.best_estimator_.predict([[z_i+2.*z_bin]]), gcv.best_estimator_.predict([[z_i+3.*z_bin]]), gcv.best_estimator_.predict([[z_i+4.*z_bin]]), gcv.best_estimator_.predict([[z_i+5.*z_bin]]), gcv.best_estimator_.predict([[z_i+6.*z_bin]]), gcv.best_estimator_.predict([[z_i+7.*z_bin]]), gcv.best_estimator_.predict([[z_i+8.*z_bin]]), gcv.best_estimator_.predict([[z_i+9.*z_bin]]), gcv.best_estimator_.predict([[z_i+10.*z_bin]]),
    gcv_crossval_scores, gcv.score(z_train, hz_train), gcv.score(z_test, hz_test))

    # appending the reconstruction results into an array
    results_arr.append([gcv.best_estimator_.predict([[z_i]]), gcv.best_estimator_.predict([[z_i+z_bin]]), gcv.best_estimator_.predict([[z_i+2.*z_bin]]), gcv.best_estimator_.predict([[z_i+3.*z_bin]]), gcv.best_estimator_.predict([[z_i+4.*z_bin]]), gcv.best_estimator_.predict([[z_i+5.*z_bin]]), gcv.best_estimator_.predict([[z_i+6.*z_bin]]), gcv.best_estimator_.predict([[z_i+7.*z_bin]]), gcv.best_estimator_.predict([[z_i+8.*z_bin]]), gcv.best_estimator_.predict([[z_i+9.*z_bin]]), gcv.best_estimator_.predict([[z_i+10.*z_bin]]), gcv.score(z_train, hz_train), gcv.score(z_test, hz_test)])

# creating numpy arrays with appended results
results_arr = np.array(results_arr)

# saving results
if sig < 0.01 and sig > 0:

    np.savetxt('results_sklearn_sigh0p00'+str(int(sig*1000))+'_'+str(nz)+'pts_'+alg+'_GCV.dat', results_arr)

if sig >= 0.01:

    np.savetxt('results_sklearn_sigh0p0'+str(int(sig*100))+'_'+str(nz)+'pts_'+alg+'_GCV.dat', results_arr)

if sig == -1:

    np.savetxt('results_sklearn_sighz_31pts_'+alg+'_GCV.dat', results_arr)
