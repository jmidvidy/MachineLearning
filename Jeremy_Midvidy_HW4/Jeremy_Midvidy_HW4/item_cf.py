# Starter code for item-based collaborative filtering
# Complete the function item_based_cf below. Do not change its name, arguments and return variables. 
# Do not change main() function, 

# import modules you need here.
import sys
import csv
import numpy as np, scipy as sp, matplotlib
import scipy.stats
import copy
import random

def item_based_cf(datafile, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems):
    '''
    build item-based collaborative filter that predicts the rating 
    of a user for a movie.
    This function returns the predicted rating and its actual rating.
    
    Parameters
    ----------
    <datafile> - a fully specified path to a file formatted like the MovieLens100K data file u.data 
    <userid> - a userId in the MovieLens100K data
    <movieid> - a movieID in the MovieLens 100K data set
    <distance> - a Boolean. If set to 0, use Pearson’s correlation as the distance measure. If 1, use Manhattan distance.
    <k> - The number of nearest neighbors to consider
    <iFlag> - A Boolean value. If set to 0 for user-based collaborative filtering, 
    only users that have actual (ie non-0) ratings for the movie are considered in your top K. 
    For item-based, use only movies that have actual ratings by the user in your top K. 
    If set to 1, simply use the top K regardless of whether the top K contain actual or filled-in ratings.
    <numOfUsers> - the number of users in the dataset 
    <numOfItems> - the number of items in the dataset
    (NOTE: use these variables (<numOfUsers>, <numOfItems>) to build user-rating matrix. 
    DO NOT USE any CONSTANT NUMBERS when building user-rating matrix. We already set these variables in the main function for you.
    The size of user-rating matrix in the test case for grading could be different from the given dataset. )
    
    returns
    -------
    trueRating: <userid>'s actual rating for <movieid>
    predictedRating: <userid>'s rating predicted by collaborative filter for <movieid>


    AUTHOR: Jeremy Midvidy
    '''
    #check to make sure this works
    def refineDataSet(u, dSet, ID):
        keysToDelete = []
        for key in u:
            contains = False
            arr = u[key]
            for row in arr:
                if(row[0] == ID):
                    contains = True
                    break
            if(contains == False):
                keysToDelete.append(key)
                
        for row in keysToDelete:
            del u[row]
         
        cleanDistSet = []
        cleanKeySet = []
        
        for x in range(0, len(dSet)):
            if (x+1 in keysToDelete):
                continue
            else:
                cleanDistSet.append(dSet[x])
                cleanKeySet.append(x+1)
        
        return cleanDistSet, cleanKeySet
        
                
    def dist(u, distanceMeasure):        
        data = []
        mainMovie = movies[movieid]
        mainUsers = []
        mainDict = dict()
        
        #initalize the above
        for row in mainMovie:
            mainDict[row[0]] = row[1]
        for row in mainMovie:
            mainUsers.append(row[0])
            
        for curr in u:
            current = u[curr]
            currentDict = dict()
            currentUsers = []
            
            mainList = []
            currentList = []
            
            #initalize all the above currents
            for row in current:
                currentUsers.append(row[0])
            for row in current:
                currentDict[row[0]] = row[1]
                
            #################################
            #constructing review vectors
            aInds = list(set(mainUsers).union(set(currentUsers)))
            for row in aInds:
                if row in currentDict:
                    currentList.append(currentDict[row])
                else:
                    currentList.append(0)
                if row in mainDict:
                    mainList.append(mainDict[row])
                else:
                    mainList.append(0)
                    
                    
            #manhattan distance or inputted distance == 1, else use pearson
            if(distance == 1):
                d = sp.spatial.distance.cityblock(mainList, currentList)
            else:
                ##do you sum? not sure
                d = sum(sp.stats.pearsonr(mainList, currentList))
                
            data.append(d)
            
        return data
        
  
    #construct dictionary for datafile
    with open(datafile,'r') as filereader:
        #sort the input file with the csv reader and the \t delimiter
        fileListReader = csv.reader(filereader, delimiter='\t')
            
        #place holder for dictionary to be created in loop
        fileList = []
        for row in fileListReader:
            d = row
            for x in range(0, len(d)):
                d[x] = int(d[x])
            fileList.append(d)
                
    #make a dictionary of moveIDs
    movies = dict()
    for x in range(1, numOfItems+1):
        movies[x] = []
        
    #go through the data, every time a movie had been reviewed
    #append the [userID review] to the key for each movieID's value
    for row in fileList:
        mID = row[1]
        uID = row[0]
        r = row[2]
        oldVal = movies[mID]
        newVal = oldVal + [[uID, r]]
        movies[mID] = newVal
        
    #refine the data set so that
    #only movies reviewed by 
    #the current user are contained within
        
    #compute distance of reviews in the datasets
    itemDistanceSet = dist(movies, distance)
    
    if(iFlag == 0):
        itemDistanceSet, keySet  = refineDataSet(movies, itemDistanceSet, userid)
    
     #find trueRating is moveid is watched by userid
    trueRating = 0
    if(movieid in movies):
        for row in movies[movieid]:
            if(row[0] == userid):
                trueRating = row[1]
                if(iFlag == 1):
                    del itemDistanceSet[movieid-1]
                    del movies[movieid]
                else: #iFlag = 0
                    trueIndex = keySet.index(movieid)
                    del itemDistanceSet[trueIndex]
                    del movies[keySet[trueIndex]]
                break
    else:
        trueRating = 0
    
    
    
    finals = []
    for x in range(0, k):
        if (k < len(itemDistanceSet)): #edge case test
            if (distance == 1):
                ind = np.argmin(itemDistanceSet)
                if(iFlag == 1):
                    finals.append(ind+1)
                    del itemDistanceSet[ind]
                else:
                    finals.append(keySet[ind])
                    del itemDistanceSet[ind]
                    del keySet[ind]
            else:
                ind = np.argmax(itemDistanceSet)
                if(iFlag == 1):
                    finals.append(ind+1)
                    del itemDistanceSet[ind]
                else:
                    finals.append(keySet[ind])
                    del itemDistanceSet[ind]
                    del keySet[ind]
                
        
    finalUsers = []
    for row in finals:
        finalUsers.append(movies[row])
        
    #create array of ratings from the finalUsers
    ratings = []
    for row in finalUsers:
        for line in row:
            ratings.append(line[1])
            
    predictedRating = sp.stats.mode(ratings)[0]
    
    return trueRating, predictedRating[0]    

def main():
    datafile = sys.argv[1]
    userid = int(sys.argv[2])
    movieid = int(sys.argv[3])
    distance = int(sys.argv[4])
    k = int(sys.argv[5])
    i = int(sys.argv[6])
    numOfUsers = 943
    numOfItems = 1682

    trueRating, predictedRating = item_based_cf(datafile, userid, movieid, distance, k, i, numOfUsers, numOfItems)
    print 'userID:{} movieID:{} trueRating:{} predictedRating:{} distance:{} K:{} I:{}'\
    .format(userid, movieid, trueRating, predictedRating, distance, k, i)




if __name__ == "__main__":
    main()
    