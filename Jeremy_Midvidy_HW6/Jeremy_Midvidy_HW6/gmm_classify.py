# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy.stats as sp
import sys
import matplotlib.pyplot as plt
from gmm_est import gmm_est
import math

"""
    EECS 349, Fall 2017
    Homework #6
    Jeremy Midvidy, jam658
"""

def main():
    """
    This function runs your code for problem 3.

    You can use this code for problem 4, but make sure you do not
    interfere with what you need to do for problem 3.
    """
    
    file_path = sys.argv[1]
    #file_path = "C:\Users\jmidv\Miniconda2\envs\eecs349\gmm_test.csv"

    x1, x2 = read_gmm_file(file_path)
    
    #pass both for 
    X = x1 + x2
    

    ###############################################################
    #hardcoded from results of gmm_est.py
    mu1 = [9.7748859235799497, 29.582587182910004] 
    sigmasq1 = [21.92280456278392, 9.7837696131844893]  
    wt1 = [0.5976546303844265, 0.40234536961557466]
        
    mu2 = [-24.822751728500094, -5.0601582830661158, 49.624444719527126]
    sigmasq2 = [7.9473354088778425, 23.322661812346212, 100.02433750443187]  
    wt2 = [0.20364945853348632, 0.49884302378965145, 0.29750751767686234]
    
    p1 = len(x1) / len(X)
    
    ###############################################################
    #get predictions for x1 and x1, then combine into list of total predcitions,
    #predX to be used for outputting the sorted lists
    
    pred1 = gmm_classify(x1, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1)
    pred2 = gmm_classify(x2, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1)
    predX = pred1 + pred2
    
        
    ###############################################################
    #sort predictions and out put sorted lists
    
    class1_data = []
    class2_data = []
            
    for i in range(0, len(predX)):
        if predX[i] == 1:
            class1_data.append(X[i])
        else:
            class2_data.append(X[i])
    

    # class1_data is a numpy array containing
    # all of the data points that your gmm classifier
    # predicted will be in class 1.
    print 'Class 1'
    print class1_data

    # class2_data is a numpy array containing
    # all of the data points that your gmm classifier
    # predicted will be in class 2.
    print '\nClass 2'
    print class2_data
    
#probability density function needed in later computations
def pdf(x, u, sigma):
    
    nom = math.exp((-(x-u)**2) / (2 * sigma**2))
    denom = ((2*math.pi)**(1/2)) * sigma
    return nom / denom


def gmm_classify(X, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1):
    """
    Input Parameters:
        - X           : N 1-dimensional data points (a 1-by-N numpy array)
        - mu1         : means of Gaussian components of the 1st class (a 1-by-K1 numpy array)
        - sigmasq1    : variances of Gaussian components of the 1st class (a 1-by-K1 numpy array)
        - wt1         : weights of Gaussian components of the 1st class (a 1-by-K1 numpy array, sums to 1)
        - mu2         : means of Gaussian components of the 2nd class (a 1-by-K2 numpy array)
        - sigmasq2    : variances of Gaussian components of the 2nd class (a 1-by-K2 numpy array)
        - wt2         : weights of Gaussian components of the 2nd class (a 1-by-K2 numpy array, sums to 1)
        - p1          : the prior probability of class 1.

    Returns:
        - class_pred  : a numpy array containing results from the gmm classifier
                        (the results array should be in the same order as the input data points)
    """

    #pretty simple, for each data point, calculate the pdf and return which ever one
    #is higher over all the gaussian models within it
    
    #initialize important variables
    class_pred = []
    p2 = 1 - p1
    
    
    
    
    for xN in X:
        
        #compute gaussian mixture model probability for each class
        
        #-------------------- class 1 ---------------------------------#
        c1_k = len(wt1)
        c1_gs = []
        
        ###sum up gaussian components for GMM 1
        for r in range(0, c1_k):
            rv = pdf(xN, mu1[r], math.sqrt(sigmasq1[r]))
            b = wt1[r] * rv
            c1_gs.append(b)
            
        ##sum up gaussian probabilities starting at prevPrior
        c1_sum =  p1 * sum(c1_gs)
        
        #-------------------- class 2 ---------------------------------#      
        c2_k = len(wt1)
        c2_gs = []
        
        ###sum up gaussian components for GMM 2
        for r in range(0, c2_k):
            rv = pdf(xN, mu2[r], math.sqrt(sigmasq2[r]))
            b = wt2[r] * rv
            c2_gs.append(b)
            
        ##sum up gaussian probabilities starting at prevPrior
        c2_sum = p2 * sum(c2_gs)
        
        #--------------- compare and assign -----------------------------#
        
        if c1_sum > c2_sum:
            class_pred.append(1)
        else:
            class_pred.append(2) 
  
    #print(class_pred)
    return class_pred


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

    #X1 = np.array(X1)
    #X2 = np.array(X2)

    return X1, X2

if __name__ == '__main__':
    main()
