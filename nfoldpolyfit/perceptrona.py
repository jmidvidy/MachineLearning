# -*- coding: utf-8 -*-
"""
Jeremy Midvidy, jam658
EECS 349, Fall 2017
Assignment 3
"""

import sys
import csv
import numpy as np
import scipy

def perceptrona(w_init, X, Y):
	#figure out (w, k) and return them here. w is the vector of weights, k is how many iterations it took to converge.
	
    w = w_init
    oneWrong  = True
    
    e = 0
    
    while (oneWrong):
        oneWrong = False
        print(w)
        print("\n")
        print(e)
        print("\n")
        for k in range(0, len(X)):
            dar = np.transpose(w)
            v = np.dot(dar, X[k])
            if( v > 0 and Y[k] > 0):
                continue
            if (v < 0 and Y[k] < 0):
                continue
            else:
                r = np.array(X[k])
                update =  w + Y[k]*r
                w = update
                oneWrong = True
        e = e + k
                

    
    return (e, k)


def main(path):
	
	#read in csv file into np.arrays X1, X2, Y1, Y2
	csvfile = open(path, 'r')
	dat = csv.reader(csvfile, delimiter=',')
	X1 = []
	Y1 = []
	X2 = []
	Y2 = []
	for i, row in enumerate(dat):
		if i > 0:
			X1.append(float(row[0]))
			X2.append(float(row[1]))
			Y1.append(float(row[2]))
			Y2.append(float(row[3]))
            
	X1 = np.array(X1)
	X2 = np.array(X2)
	Y1 = np.array(Y1)
	Y2 = np.array(Y2)
    
        xOne = []
        xTwo = []
    
        for row in X1:
            a = [1 , row]
            xOne.append(a)
            
        for row in X2:
            a = [1 , row]
            xTwo.append(a)
    
        w_init = [1, 0]
        perceptrona(w_init, xOne, Y1)



    

    
