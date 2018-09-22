#Starter code for spam filter assignment in EECS349 Machine Learning
#Author: Jeremy Midvidy, jam658.  EECS 349 HW#5

import sys
import numpy as np
import os
import shutil
import math
import csv
import scipy as sp, matplotlib as plt
import copy
import random


def parse(text_file):
	#This function parses the text_file passed into it into a set of words. Right now it just splits up the file by blank spaces, and returns the set of unique strings used in the file. 
	content = text_file.read()
	return np.unique(content.split())

def writedictionary(dictionary, dictionary_filename):
	#Don't edit this function. It writes the dictionary to an output file.
	output = open(dictionary_filename, 'w')
	header = 'word\tP[word|spam]\tP[word|ham]\n'
	output.write(header)
	for k in dictionary:
		line = '{0}\t{1}\t{2}\n'.format(k, str(dictionary[k]['spam']), str(dictionary[k]['ham']))
		output.write(line)
		

def makedictionary(spam_directory, ham_directory, dictionary_filename):
    #Making the dictionary. 
    spam = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))]
    ham = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))]
	
    spam_prior_probability = len(spam)/float((len(spam) + len(ham)))
	
    words = {}

    #These for loops walk through the files and construct the dictionary. The dictionary, words, is constructed so that words[word]['spam'] gives the probability of observing that word, given we have a spam document P(word|spam), and words[word]['ham'] gives the probability of observing that word, given a hamd document P(word|ham). Right now, all it does is initialize both probabilities to 0. TODO: add code that puts in your estimates for P(word|spam) and P(word|ham).
    for s in spam:
        for word in parse(open(spam_directory + s)):
            if word not in words:
                words[word] =  {'spam': 1, 'ham': 1}
                break
            else:
                words[word]['spam'] = words[word]['spam'] + 1 
                break
            
    for h in ham:
        for word in parse(open(ham_directory + h)):
            if word not in words:
                words[word] =  {'spam': 1, 'ham': 1}
                break
            else:
                words[word]['ham'] = words[word]['ham'] + 1
                break

    #now have the number of raw times a word occurs in a document
    #need to convert to probabilities
    
    spam_denom = float(len(spam)) + 1
    ham_denom = float(len(ham)) + 1
    
    for key in words:
        s = words[key]['spam'] / spam_denom
        h = words[key]['ham'] / ham_denom
        words[key] = {'spam' : s, 'ham' : h}
        
    
    #Write it to a dictionary output file.
    writedictionary(words, dictionary_filename)
	
    return words, spam_prior_probability



def is_spam(content, dictionary, spam_prior_probability):
	#TODO: Update this function. Right now, all it does is checks whether the spam_prior_probability is more than half the data. If it is, it says spam for everything. Else, it says ham for everything. You need to update it to make it use the dictionary and the content of the mail. Here is where your naive Bayes classifier goes.
    
    Vj = spam_prior_probability
    
    contentHamDict = {}
    contentSpamDict = {}
    
    #make sure to test for edge case where word is not in the document
    #construc a dictionary of values where each key is the pS or pH of each word
    #in content 
    for row in content:
        if row in dictionary:
            h = dictionary[row]['ham']
            s = dictionary[row]['spam']
            contentHamDict[row] = h
            contentSpamDict[row] = s


    
    
    #edge case where none of the words in an email are in the dictionary
    #then just classify isSpam based on prevProb
    if (len(contentHamDict) == 0):
        if Vj > .5:
            return True
        else:
            return False
        
    
    probsHam = []
    probsSpam = []
    #compute the log10 of probability in spam and ham
    for row in contentHamDict:
        pH = math.log10(contentHamDict[row])
        pS = math.log10(contentSpamDict[row])
        
        probsHam.append(pH)
        probsSpam.append(pS)
       
    #need to find the prob that a message is not in spam/ham.  1-
    notpHam = []
    notpSpam = []
    
    for row in contentHamDict:
        pH = math.log10(1-contentHamDict[row])
        pS = math.log10(1-contentSpamDict[row])
        
        notpHam.append(pH)
        notpSpam.append(pS)
       
    ###########################################
    #these are the probs that a message is IN the spam/ha,
    pHam = probsHam[0] + notpSpam[0]
    pSpam = probsSpam[0] + notpHam[0]
    
 
    
    #summate all the probabilities together
    for x in range(1, len(probsHam)):
        pHam = pHam * probsHam[x] * notpSpam[x]
    for x in range(1, len(probsSpam)):
        pSpam = pSpam * probsSpam[x] * notpHam[x]
            
    pHam = pHam * math.log10(1-Vj)    
    pSpam = pSpam * math.log10(Vj)
        

    #return the larger probabilitiy, T for isSpam
    if pSpam > pHam:
        return True
    else:
        return False



def spamsort(mail_directory, spam_directory, ham_directory, dictionary, spam_prior_probability):
    mail = [f for f in os.listdir(mail_directory) if os.path.isfile(os.path.join(mail_directory, f))]
    for m in mail:
        content = parse(open(mail_directory + m))
        spam = is_spam(content, dictionary, spam_prior_probability)
        if spam:
            shutil.copy(mail_directory + m, spam_directory)
        else:
            shutil.copy(mail_directory + m, ham_directory)



if __name__ == "__main__":
   #Here you can test your functions. Pass it a training_spam_directory, a training_ham_directory, and a mail_directory that is filled with unsorted mail on the command line. It will create two directories in the directory where this file exists: sorted_spam, and sorted_ham. The files will show up  in this directories according to the algorithm you developed.
   training_spam_directory = sys.argv[1]
   training_ham_directory = sys.argv[2]
	
   test_mail_directory = sys.argv[3]
   
   #make sure to change back to above before you turn it in
   
   if not os.path.exists(test_spam_directory):
       os.mkdir(test_spam_directory)
   if not os.path.exists(test_ham_directory):
       os.mkdir(test_ham_directory)

   dictionary_filename = "dictionary.dict"
	
	#create the dictionary to be used
   dictionary, spam_prior_probability = makedictionary(training_spam_directory, training_ham_directory, dictionary_filename)
	#sort the mail
   spamsort(test_mail_directory, test_spam_directory, test_ham_directory, dictionary, spam_prior_probability) 
