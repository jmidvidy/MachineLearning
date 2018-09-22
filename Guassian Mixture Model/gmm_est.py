# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy.stats as sp
import sys
import matplotlib.pyplot as plt
import math

"""
    EECS 349, Fall 2017
    Homework #6
    Jeremy Midvidy, jam658
"""

def main():
    """
    This function runs your code for problem 2.

    You can also use this to test your code for problem 1,
    but make sure that you do not leave anything in here that will interfere
    with problem 2. Especially make sure that gmm_est does not output anything
    extraneous, as problem 2 has a very specific expected output.
    """
    #file_path = "C:\Users\jmidv\Miniconda2\envs\eecs349\gmm_train.csv"
    file_path = sys.argv[1]
    #make sure to change back before submitting sys.argv[1]

    ######################################################
    #read file and construct vectors
    
    x1, x2 = read_gmm_file(file_path)
    
    #use plt to eyeball input paramters
    xs = sorted(x2)
    #plt.hist(x1)
#    
    #print(xs)
    #v = np.var(xs[650:1450])
    #print(v)
    
    ######################################################
    #initialize parameters
    #we know that k = 2 because of two classifications so
    #all input vectors will have length of two
    
    #eyeballed input paramters, 
    #class 1 has 2 gaussians, class 2 has 3 gaussians
    mu1 = [10 , 30]
    mu2 = [-20, -2, 50]
    
    sig1 = [20 , 20]
    sig2 = [20, 20, 20]
    
    w1 = [ 0.5, 0.5 ] #random weights that add up to one
    w2 = [ 1/3 , 1/3, 1/3]
    
    its = 25 #go over why this number, maybe more?
    
    ######################################################
    #get results of EM algorithm
    
    
    mu_results1, sigma2_results1, w_results1, l1 = gmm_est(x1, mu1, sig1, w1, its)
    mu_results2, sigma2_results2, w_results2, l2 = gmm_est(x2, mu2, sig2, w2, its)
    
    #make sure to delete this part before submitting
#    print("\n")
#    print(l1)
#    print("\n")
#    print(l2)
#    print("\n")

    ######################################################
    # mu_results1, sigma2_results1, w_results1 are all numpy arrays
    # with learned parameters from Class 1
    print 'Class 1'
    print 'mu =', mu_results1, '\nsigma^2 =', sigma2_results1, '\nw =', w_results1
    
    

    # mu_results2, sigma2_results2, w_results2 are all numpy arrays
    # with learned parameters from Class 2
    print '\nClass 2'
    print 'mu =', mu_results2, '\nsigma^2 =', sigma2_results2, '\nw =', w_results2

#probability density function needed in later computations
def pdf(x, u, sigma):
    
    nom = math.exp((-(x-u)**2) / (2 * sigma**2))
    denom = ((2*math.pi)**(1/2)) * sigma
    return nom / denom

def gmm_est(X, mu_init, sigmasq_init, wt_init, its):
    """
    Input Parameters:
      - X             : N 1-dimensional data points (a 1-by-N numpy array)
      - mu_init       : initial means of K Gaussian components (a 1-by-K numpy array)
      - sigmasq_init  : initial  variances of K Gaussian components (a 1-by-K numpy array)
      - wt_init       : initial weights of k Gaussian components (a 1-by-K numpy array that sums to 1)
      - its           : number of iterations for the EM algorithm

    Returns:
      - mu            : means of Gaussian components (a 1-by-K numpy array)
      - sigmasq       : variances of Gaussian components (a 1-by-K numpy array)
      - wt            : weights of Gaussian components (a 1-by-K numpy array, sums to 1)
      - L             : log likelihood
    """
    #initialize local variables and outputs
    mu = mu_init
    sigmasq = sigmasq_init
    wt = wt_init
    LogLikliHoods = []
    
    #important variables for E/N
    k = len(wt)
    N = len(X)
    
    #iterate through the list, either number of iteratons
    #or end of data set
    end = its
    if len(X) < its:
        end = len(X)
    
    #do E/M for specified number of iterations    
    for count in range(0, end):
        
        #all responsibilities
        Gs = []
        for row in wt:
            Gs.append([])
            
        #keep track of log likelihood
        lls = []
        
        ######################################################
        #------------- Expectation Step: --------------------#
        #go through each data point and calculate 
        #responsibility for each gausian with current parameters
        
        for i in range(0, N):
            
            xN = X[i]            
            #calculate responsibilities for each gaussian with current parameters
            gs = []
            
            ##first get normal distrobutions of each guassian at current x value 
            for j in range(0, k):
                gs.append(pdf(xN, mu[j], math.sqrt(sigmasq[j])))
        
            ##then responsibility for each gaussian to current datapoint, xN
            for j in range(0, k):
                gs[j] = wt[j] * gs[j]
                
            
            denom = sum(gs)
            lls.append(denom)
            
            ##normalize each 
            for j in range(0, k):
                gs[j] = gs[j] / denom
            
            #aggregate responsibilites for later use
            for j in range(0, k):
                Gs[j].append(gs[j])

        
        #######################################################
        #------------- Maximization Step: --------------------#   
        #adjust paramters to gaussian (best responsibility)
        #within each soft cluster        
        
        GausResponsibility = []
        for row in Gs:
            GausResponsibility.append(sum(row))
        
        #update w vecotr
        for j in range(0, k):
            wt[j] = GausResponsibility[j] / N
        
        #update mu vector
        ds = []
        ##calculate *gij*xi for each gaussian between 0-k
        for j in range(0, k):
            d = []
            for i in range(0, N):
                a = X[i]*Gs[j][i]
                d.append(a)
            ds.append(d)
        
        ##update mu vector with sums of error measures 
        for j in range(0, k):
            mu[j] = sum(ds[j]) / GausResponsibility[j]
        
        #update var vector
        vs = []
        ##calculate each new variance for each gaussian component
        for j in range (0, k):
            v = []
            for i in range(0, N):
                a = Gs[j][i]*((X[i] - mu[j])**2)
                v.append(a)
            vs.append(v)
                
        for j in range(0, k):
            sigmasq[j] = sum(vs[j]) / GausResponsibility[j]
            

        if count < 20:
            LogLikliHoods.append(math.log(sum(lls)))
    

    
    return mu, sigmasq, wt, LogLikliHoods

def read_gmm_file(path_to_file):
    """
    Reads either gmm_test.csv or gmm_train.csv
    :param path_to_file: path to .csv file
    :return: two numpy arrays for data with label 1 (X1) and data with label 2 (X2)
    """
    X1 = []
    X2 = []

    data = open(path_to_file).readlines()[1:] # we don't need the first line
    for d in data:
        d = d.split(',')

        # We know the data is either class 1 or class 2
        if int(d[1]) == 1:
            X1.append(float(d[0]))
        else:
            X2.append(float(d[0]))

    X1 = np.array(X1)
    X2 = np.array(X2)

    return X1, X2

if __name__ == '__main__':
    main()
