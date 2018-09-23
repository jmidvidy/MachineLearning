
import sys
import csv
import numpy as np, scipy as sp, matplotlib
import scipy.stats
import copy
import random


def user_based_cf(datafile, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems):
    '''
    build user-based collaborative filter that predicts the rating 
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
    For user-based, use only movies that have actual ratings by the user in your top K. 
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
            
            
                
    def dist(u, distance):
        data = []
        
        mainUser = users[userid]
        mainInds = []
        mainDict = dict()
        
        
        #initialize all the above mains
        for row in mainUser:
            mainDict[row[0]] = row[1]
        for row in mainUser:
            mainInds.append(row[0])
        #################################
        
        for curr in u:
            current = u[curr]
            mainList = []
            currentDict = dict()
            currentInds = []
            currentList = []
            
            #initalize all the above currents
            for row in current:
                currentInds.append(row[0])
            for row in current:
                currentDict[row[0]] = row[1]
            #################################
            
            #constructing review vectors
            aInds = list(set(mainInds).union(set(currentInds)))
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
                
    
                
    #now build a dictionary according to aggregate data about USERS
    #each key represents a user ID
    users = dict()
    for x in range(1, numOfUsers+1):
        users[x] = []
        
    ##now each time a user has seen a film, add that moveID and rating
    ##to it's dictionary
    for row in fileList:
        r = row[0]
        d = row[1]
        m = row[2]
        oldVal = users[r]
        newVal = oldVal + [[d, m]]
        users[r] = newVal
    
        
    #compute according distance with the dataSet
    distanceSet = dist(users, distance)
    
    #refine data set as indicated by the assigment
    if(iFlag == 0):
        distanceSet, keySet  = refineDataSet(users, distanceSet, movieid)
    
    #now aggregate k-cluster of best k number of users
    
    #find trueRating is moveid is watched by userid
    trueRating = 0
    if(userid in users):
        for row in users[userid]:
                if(row[0] == movieid):
                    trueRating = row[1]
                    if(iFlag == 1):
                        del distanceSet[userid-1]
                        del users[userid]
                    else: #iFlag = 0
                        trueIndex = keySet.index(userid)
                        del distanceSet[trueIndex]
                        del users[keySet[trueIndex]]
                    break
    else:
        trueRating = 0
    
    finals = []
    for x in range(0, k):
        if k < len(distanceSet): #edgecase test
            if (distance == 1):
                ind = np.argmin(distanceSet)
                if(iFlag == 1):
                    finals.append(ind+1)
                    del distanceSet[ind]
                else:
                    finals.append(keySet[ind])
                    del distanceSet[ind]
                    del keySet[ind]
            else:
                ind = np.argmax(distanceSet)
                if(iFlag == 1):
                    finals.append(ind+1)
                    del distanceSet[ind]
                else:
                    finals.append(keySet[ind])
                    del distanceSet[ind]
                    del keySet[ind]
            
    
        
    finalUsers = []
    for row in finals:
        finalUsers.append(users[row])
        
    #create array of ratings from the finalUsers
    ratings = []
    for row in finalUsers:
        for line in row:
            ratings.append(line[1])
            
    predictedRating = sp.stats.mode(ratings)[0]
    
    return trueRating, predictedRating[0]



#make sure to change back before submitting
def main():
    datafile = sys.argv[1]
    userid = int(sys.argv[2])
    movieid = int(sys.argv[3])
    distance = int(sys.argv[4])
    k = int(sys.argv[5])
    i = int(sys.argv[6])
    numOfUsers = 943
    numOfItems = 1682
    
    trueRating, predictedRating = user_based_cf(datafile, userid, movieid, distance, k, i, numOfUsers, numOfItems)
    print 'userID:{} movieID:{} trueRating:{} predictedRating:{} distance:{} K:{} I:{}'\
    .format(userid, movieid, trueRating, predictedRating, distance, k, i)




if __name__ == "__main__":
    main()
    
