# -*- coding: utf-8 -*-
"""
EECS 349, HW9 Classifier_1.py 

@author: Jeremy Midvidy, jam658
"""

import matplotlib.pyplot as plt
import pickle
import sklearn
import numpy as np
from sklearn import metrics
from sklearn import svm # this is an example of using SVM
from mnist import load_mnist

def preprocess(images):
    #this function is suggested to help build your classifier. 
    #You might want to do something with the images before 
    #handing them to the classifier. Right now it does nothing.

    return [i.flatten() for i in images]

def build_classifier(images, labels):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.
    
    #---------- my code -----------------#
    #used SVM for this example
    
    classifier = svm.SVC(C=16)
    classifier.fit(images, labels)
    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    
    #changed according to piazza post
    import pickle
    pickle.dump(classifier, open('classifier_1.p', 'w'))


def classify(images, classifier):
    #runs the classifier on a set of images. 
    return classifier.predict(images)

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))

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
    classifier = build_classifier(training_set, training_labels)
    save_classifier(classifier, training_set, training_labels)
    classifier = pickle.load(open('classifier_1.p'))
    predicted = classify(testing_set, classifier)
    
#    cf_matrix = metrics.confusion_matrix(testing_labels, predicted)
#    print(cf_matrix)
    
    print error_measure(predicted, testing_labels)
