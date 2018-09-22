# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:07:29 2017

@author: jmidv
"""

# -*- coding: utf-8 -*-
"""
Jeremy Midvidy, jam658
EECS 349, Fall 2017
Assignment 3
"""

######################################
#need to make sure the inputs are lineraly combinanble
######################################



import sys
import csv
import numpy as np
import scipy

linnerClassFileName = 'C:\Users\jmidv\Miniconda2\envs\eecs349\linearclass.csv'

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
    
       
        xTwo = []
        
    ##make sure list is linearly combinable
    ##find all indicies where there are two inputs
    ##with the same value
    ##remove all elements from x and y according to found indicies 
    
        indicies = []
        for x in range(0, len(X2)):
            for y in range(0, len(X2)):
                if((np.abs(X2[x] - X2[y]) < .01961) and x != y):
                    indicies.append(x)
                    break
                
        cleanX2 = []
        for i in range(0, len(X2)):
            if (i in indicies):
                continue
            else:
                cleanX2.append(X2[i])
                
        cleanY2 = []
        for i in range(0, len(Y2)):
            if (i in indicies):
                continue
            else:
                cleanY2.append(Y2[i])
        for row in cleanX2:
            a = [1 , row]
            xTwo.append(a)
    
        w_init = [1, 0]
        perceptrona(w_init, xTwo, cleanY2)


main(linnerClassFileName)
    

    
