# Machine learning the Hubble Constant
Scripts I have developed to measure the Hubble Constant (H0) from a given set of H(z) measurements using machine learning (ML) regression algorithms. 
The goal is to create a performance benchmark of these algorithms on measuring H0 given different H(z) data specifications - that is, different number of data points (Nz) and relative measurement uncertainties (sig). 

This repository includes four scripts, which are
- hz_sim_gen.py: Produces H(z) simulated data-sets assuming a fiducial model for different Nz and sig
- H0_regression.py: Performs a regression analysis over the provided H(z) data for different machine learning (ML) algorithms using the scikit-learn package. So we can measure H0 from the reconstructed H(z) values by getting H(z=0). 
- H0_stats.py: Computes the stats of the measured H0 values from a set of nsim simulations. This is necessary in order to estimate the uncertainty and bias-variance tradeoff (BVT) of these measurements
- plotting_results.py: Produces H0 and BVT plots

Examples of simulated H(z) data-sets produced using hz_sim_gen.py can be found in the input.zip file. 

Some considerations to be taken into account:
- Fiducial Cosmology assumes Planck 2018 (TT, TE, EE+lowE+lensing) best-fit
- ML algorithms deployed: Decision Trees, Artificial Neural Networks, Gradient Boosting and Support Vector Machine
- A grid search over the algorithm hyperparameters is carried out to reconstruct the H(z) curve that best fit the data
- 100 Monte Carlo realisations over the fiducial model are produced for each H(z) data-set specification in order to estimate the H0 uncertainty obtained from each algorithm - thus a Monte-Carlo-bootstrap method. 
- The BVT is used as a metric to evaluate the regression performance of each algorithm on each set of simulation specifications. It is defined as BVT = MSE - b^2, where b (bias) is the displacement between the measured and fiducial H0, and MSE corresponds to the H0 mean squared error. 
- Bash scripts examples for each python script are provided to automatise the analysis

For further details, see arxiv:2209.09017 (accepted for publication in European Physical Journal C) and scikit-learn documentation [https://scikit-learn.org/stable/]. 

I hope this code is helpful for students and anyone interested in performing this analysis in the future. Please contact carlosbengaly@on.br or carlosap87@gmail.com for further enquiries. Suggestions are always welcome. 
