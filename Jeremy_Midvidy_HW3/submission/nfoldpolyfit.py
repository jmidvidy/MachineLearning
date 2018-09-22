# -*- coding: utf-8 -*-
"""
Jeremy Midvidy, jam658
EECS 349 Fall 2017
Assignment 3
"""



#############################################################
#############################################################
#############################################################
#-----------------------------------------------------------#
#------------------------ PROBLEM 1 ------------------------#
#----------------------------------------------------=------#
#############################################################
#############################################################
#############################################################


def main(csvFile, maxK, n, verbose):
    
    import csv
    import numpy as np, scipy as sp, matplotlib
    import copy
    import random
    import time
    import matplotlib.pyplot as plt
   
    #############################################################
    #############################################################
    #############################################################
    #-----------------------------------------------------------#
    #-------------Construct the vector list --------------------#
    #-----------------------------------------------------------#
    #############################################################
    #############################################################
    #############################################################
    
    
    
    #construct list for LinnearReg.  SKIP the first like with values 'X' and 'Y'
    #and convert the string values of the number to float representations using
    #the FLOAT function
    
    with open(LinnearRegFilePath,'r') as filereader:
            
            #skip the first line containing the 'X' and 'Y' values
            next(filereader)
            
            #sort the input file with the csv reader and the , delimiter
            fileListReader = csv.reader(filereader, delimiter=',')
            
            #place holder for dictionary to be created in loop
            fileList = []
            X = []
            Y = []
            for row in fileListReader:
                d = []
                d.append(float(row[0]))
                d.append(float(row[1]))
                X.append(float(row[0]))
                Y.append(float(row[1]))
                fileList.append(d)
    
    
    
    #############################################################
    #############################################################
    #############################################################
    #-----------------------------------------------------------#
    #-------------Run polynomial regression --------------------#
    #-----------------------------------------------------------#
    #############################################################
    #############################################################
    #############################################################
    
    ##############################################################
    
    #constructing groups of n sets
    groups = []
    numGroups = len(fileList) / n
    
    for x in range(0, numGroups):
        groups.append([])
 
    i = -1
    for j in range(0, len(fileList)):
        if (j%numGroups == 0):
            i = i + 1
        groups[i].append(fileList[j])
    
    ##################################################################
    
    #trainingSet and testingSet indicies
    trainingSetIndicies = []
    testingSetIndicies = []
    for x in range(0, n):
        trainSet = []
        for y in range(0,n):
            if (y == x):
                testingSetIndicies.append(y)
            else:
                trainSet.append(y)
        trainingSetIndicies.append(trainSet)
        
    ##################################################################
    
    #construct the indiviual trainingSets/testingSets, and get the 
    #polynomial expressions for each trainingSet and each testingSet
    
    polynomialExpressions = []
    groupsX = []
    groupsY = []
    
    lineCount = 0
    for line in trainingSetIndicies:
        xGroups = []
        yGroups = []
        for row in line:
            temp = groups[row]
            for elem in temp:
                xGroups.append(elem[0])
                yGroups.append(elem[1])
            
        groupsX.append(xGroups)
        groupsY.append(yGroups)   
        lineCount = lineCount + 1
    
    ##################################################################
    
    
    for d in range (0, len(groupsX)):
        expressions = []
        for kVal in range(0, maxK+1):
            reg = np.polyfit(groupsX[d], groupsY[d], kVal)
            expressions.append(reg)
        polynomialExpressions.append(expressions)
         
        
    #################################################################
    
    #now apply the regression equations to whichever subset absent from it's trainingSet
    #and create list of values from evaluationng the appripiate trainingSets with the
    #appropriate polynomial value
    
    #need to construct sets of just the x and y values for each group, 5 each vector
    validationSetGroupsX = []
    validationSetGroupsY = []
    
    for x in range(0, n):
        validationSetGroupsX.append([])
        validationSetGroupsY.append([])
 
    i = -1
    for x in range(0, len(fileList)):
        if (x % numGroups == 0):
            i = i + 1
        else:
           validationSetGroupsX[i].append(fileList[x][0])
           validationSetGroupsY[i].append(fileList[x][1])
            
    vals = []
    for r in range(0, len(testingSetIndicies)):
        kvalues = []
        for row in polynomialExpressions[r]:
            v = np.polyval(row, validationSetGroupsX[r])
            kvalues.append(v)
        vals.append(kvalues)
        
    ##############################################################
    
    #now need to compute the SSE for each k-value per partion
    #get the sum of squared errors for each error in row
    
    SSEs = []
    
    c = 0
    for row in vals:
        errs= []
        for line in row:
            a = 0
            totalError = 0.0
            totalVal = 0.0
            for elem in line:
                totalVal = totalVal + validationSetGroupsY[c][a]
                totalError = totalError + elem
                a = a + 1
            SSE = (totalError - totalVal)**2 / (totalVal)**2
            errs.append(SSE)
        SSEs.append(errs)
        c = c + 1
    
    
    ################################################
    
    #now find the mean value of k value
    
    k = [0,1,2,3,4,5,6,7,8,9]    
    meansMSE = []
    
    #arrange the MSE's into rows whereby each row corresponds to each k-value
    #and the first is k=0 for paritiion 1, k=0 for partion2....k=maxK for partition n
    #and then find the averageMSE for each k value over all of the folds
    for x in range(0,maxK+1):
        meansMSE.append([])

    for x in range(0, n):
        for y in range(0, maxK+1):
            meansMSE[y].append(SSEs[x][y])
    
    avergageMSE = []
    
    for row in meansMSE:
        total = 0.0
        for elem in row:
            total = total + elem
        avg = total / len(row)
        avergageMSE.append(avg)
    
    
    if (verbose == 1):
        
        #plot of k-values vs. mean SSE (MSE) values
        
        plt.figure(1)
        plt.plot(k, avergageMSE)
        plt.title("averageMSE's per k-value over all of the folds")
        plt.xlabel('k values')
        plt.ylabel('mean SSE')
        
        
        #need to finalize graphs
        plt.figure(2)
        plt.scatter(X, Y)
        plt.plot(vals[4][4])
        plt.title("Scatter of X and Y values as well as function for best value of k")
        plt.xlabel('x values')
        plt.ylabel('y values')
        plt.plot(MSEs[3][3])
        
