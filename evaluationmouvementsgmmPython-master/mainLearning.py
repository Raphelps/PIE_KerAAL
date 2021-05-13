#!/usr/bin/env python
from Model import Model
from functions.processTrainingData import processTrainingData
from functions.learnGMMmodel import learnGMMmodel
import pickle
import numpy as np
import time


## Parameters
nbData = 300  # Number of datapoints
nbSamples = 2  # Number of demonstrations
trainName = 'data/EtirementsLateraux/'  # folders names from where to load data
nbVar = 46  # Dimension of the tangent space (incl. time)
nbVarMan = 55  # Dimension of the manifold (incl. time)
nbIter = 10  # Number of iteration for the Gauss Newton algorithm
nbIterEM = 10  # Number of iteration for the EM algorithm
nbStates = 15  # Number of states in the GMM
dt = 0.01  # Time step duration
params_diagRegFact = 1E-4  # Regularization of covariance
registration = 1  # temporal alignment or not
filt = 1  # filtering of data or not
est = 1  # estimation of orientation from position or kinect quaternions
rem = 1  # removal of begining of the sequence (no motion) or not
ws = 21  # windows size for segmentation
fastDP = 1  # fast temporal alignment (using windows instead of full sequence) or not

model = Model(nbVar, nbVarMan, nbStates, dt, params_diagRegFact)
xIn, uIn, xOut, uOut = processTrainingData(model, trainName, nbSamples, registration, fastDP, filt, est, rem, ws, nbData) ##1.6s
u = uIn
x = xIn
for i in range(1, 16):
    u = np.vstack((u, uOut[i]))
    x = np.vstack((x, xOut[i]))
model.x = x
learnGMMmodel(model, u, xIn, xOut, nbSamples, nbIterEM, nbIter, nbData)  ## execution time : 2.3s

f = open('model.txt', 'wb')
pickle.dump(model, f)
f.close()

print('Execution time : ',time.perf_counter(),'s')
