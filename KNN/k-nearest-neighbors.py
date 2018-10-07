#Example and helloworld introduction

#1    Load the data
#2    Initialise the value of k
#3    For getting the predicted class, iterate from 1 to total number of training data points
#3.1        Calculate the distance between test data and each row of training data. Here we will use Euclidean distance as our distance metric since it’s the most popular method. The other metrics that 
#           can be used are Chebyshev, cosine, etc.
#3.2    Sort the calculated distances in ascending order based on distance values
#3.3    Get top k rows from the sorted array
#3.4    Get the most frequent class of these rows
#3.5    Return the predicted class


#Importing Libraries
import pandas as pd
import numpy as np
import math
import operator
import os

import time

#Dataset directory
path = os.path.join(os.path.expanduser('~'), 'Projects', 'DataCamp', 'DataScience_Python', 'Datasets')
dir = os.path.join(path, 'iris.csv')

#print(dir)
#time.sleep(5.5)


#### STEP 1 
#Importing data
data = pd.read_csv(dir)

# Defining function that will calculate Eclidean distance (square all for avoid negative values and sqrt to crack down measures)
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)

#defining KNN model
def knn(trainingSet, testInstance, k):

    distances = {}
    sort = {}

    length = testInstance.shape[1]

    #### STEP 3

    #Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):

        ####STEP 3.1
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)

        distances[x] = dist[0]
        

    #### STEP 3.2
    #Sorting values on the basis of distance
    sorted_d = sorted(distances.items(), key = operator.itemgetter(1))

    neighbors = []

    ####STEP 3.3
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])

    classVotes = {}

    ####STEP 3.4
    # Calculating the most freq class in neighbors
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]


        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    
    #### STEP 3.5
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedVotes[0][0], neighbors)





#creating dummy testset
testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)


#### STEP 2
# Setting number of neighbors = 1
k = 1

# Running KNN model

result, neigh = knn(data, test, k)


print(result)

print(neigh)


time.sleep(5.5)
