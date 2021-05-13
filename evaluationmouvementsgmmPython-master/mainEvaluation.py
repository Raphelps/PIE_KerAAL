#!/usr/bin/env python
from functions.loadData import loadData
from functions.temporalAlignmentEval import temporalAlignmentEval
from functions.computeLikelihoods import computeLikelihoods
from functions.computeScores import computeScores
import time
import numpy as np
import pickle

## Parameters
nbData = 300  # Number of datapoints
seuil = -200  # Threshold used for computing score in percentage. The more close it is to zero, the more strict is the evaluation
minseuil = -500  # default values
registration = 1  # temporal alignment or not
filt = 1  # filtering of data or not
est = 1  # estimation of orientation from position
rem = 1  # removal of begining of the sequence (no motion) or not
ws = 21  # windows size for segmentation
fastDP = 1  # fast temporal alignment (using windows instead of full sequence) or not
PrintResults = 1  # print the results or not

# load modelExo3
f = open('model.txt', 'rb')
model = pickle.load(f)
f.close()

# data train for temporal alignment
dirTrain = 'data/EtirementsLateraux/'
fnameTrain = 'SkeletonSequence3.txt'
oriMatTrain, posMatTrain, dataTrain = loadData(dirTrain, fnameTrain, filt, est, rem, ws, nbData)

# data test
dataTest = []
oriMatTestLong = []
posMatTestLong = []

dirTest = 'data/'
fnameTest = 'SkeletonSequenceErr1.txt'
[oriMatTest_, posMatTest_, dataTest_] = loadData(dirTest, fnameTest, filt, est, rem, ws, nbData)
dataTest.append(dataTest_)
oriMatTestLong.append(oriMatTest_)
posMatTestLong.append(posMatTest_)

## Evaluate sequence
for rep in range(len(dataTest)):
    if registration == 1:
        dataTestAligned, r, allPoses, poses, motion, distFI = temporalAlignmentEval(model.cuts, dataTrain,
                                                                                    dataTest[rep], fastDP, nbData)
        posMatTest = posMatTestLong[rep][:, r]
    else:
        dataTestAligned = dataTest[rep]

    # compute likelihoods
    Lglobal, Lbodypart, Ljoints = computeLikelihoods(model.dt, model.nbStates, model.nbVar, model.Priors, model.Mu,
                                                     model.Sigma, model.MuMan, dataTestAligned, nbData)
    # It will be clearer to give model as a parameter, but it will be slower

    # get scores
    seuils = np.ones(6) * seuil
    minseuils = np.ones(6) * minseuil
    Sglobal, Sbodypart, Sjoints = computeScores(model.cuts, model.cutsKP, Lglobal, Lbodypart, Ljoints, seuils,
                                                minseuils)

    if PrintResults:
        print('Sglobal')
        for i in Sglobal:
            print(i)
            Sglobal[i].print()
            print()
        print('.................................')
        print('Sbodypart')
        for i in Sbodypart:
            for j in Sbodypart[i]:
                print(j)
                Sbodypart[i][j].print()
                print()
        print('.................................')
        print('Sjoints')
        for i in Sjoints:
            for j in Sjoints[i]:
                print(j)
                Sjoints[i][j].print()
                print()

print('Execution time : ', time.perf_counter(), 's')
