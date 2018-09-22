# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 21:50:44 2017

@author: Jeremy Midvidy, jam658
"""

import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import classifier_1 as cl1
import classifier_2 as cl2
#%matplotlib inline

def preprocess(images):
    #this function is suggested to help build your classifier. 
    #You might want to do something with the images before 
    #handing them to the classifier. Right now it does nothing.

    return [i.flatten() for i in images]

##############################################
#--------------------------------------------#
#------------- BOOSTING CODE ----------------#
#--------------------------------------------#
##############################################


def boosting_A(training_set, training_labels, testing_set, testing_labels):
    # Build boosting algorithm for question A
    # Return confusion matrix
    
    
    #make boosted classifer and fit it to default model
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(training_set, training_labels)
    
    #train model and compute error
    expected = testing_labels
    predicted = clf.predict(testing_set)
    
    #make confusion matrix for output
    confusion_matrix = metrics.confusion_matrix(expected, predicted)
    
    err = np.count_nonzero(abs(predicted - expected))/float(len(predicted))
    #print(err)
    
    return confusion_matrix

def boosting_B(training_set, training_labels, testing_set, testing_labels):
    # Build boosting algorithm for question B
    # Return confusion matrix
    
    #make boosted classifier with SVC as the assignment says to do
    clf = AdaBoostClassifier(base_estimator=SVC(probability=True), n_estimators=100)
    clf.fit(training_set, training_labels)
    
    #calculate error measure and make confusion matrix
    expected = testing_labels
    predicted = clf.predict(testing_set)
    
    #make confusion matrix for output
    confusion_matrix = metrics.confusion_matrix(expected, predicted)
    
    err = np.count_nonzero(abs(predicted - expected))/float(len(predicted))
    #print(err)
    
    return confusion_matrix
    

if __name__ == "__main__":

    # Code for loading data
    images, labels = load_mnist(digits=range(0,10), path='.')
    
    # preprocessing
    images = preprocess(images)
    
    # pick training and testing set
    # YOU HAVE TO CHANGE THIS TO PICK DIFFERENT SET OF DATA
    training_set = images[0:1000]
    training_labels = labels[0:1000]
    testing_set = images[0:100]
    testing_labels = labels[0:100]

    #build_classifier is a function that takes in training data and outputs an sklearn classifier.
    confusionMatrix_A = boosting_A(training_set, training_labels, testing_set, testing_labels)
    confusionMatrix_B = boosting_B(training_set, training_labels, testing_set, testing_labels)
    
    print confusionMatrix_A
    print confusionMatrix_B
    
    