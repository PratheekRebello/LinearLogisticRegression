#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 20:10:42 2019

@author: pratheek
"""


import sys
program_name = sys.argv[0]
arguments = sys.argv[1:]

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import random
np.seterr(divide='ignore', invalid='ignore')
from scipy import stats 

def normalised(a,Y):
    a = np.dot(np.transpose(a-Y),(a-Y))
    b = np.dot(np.transpose(Y),Y)
    return a/b

def loss(X,W,Y,lamb):
    samples = X.shape[0]
    weights = X.shape[1]
    X = X.reshape((samples,weights))
    W = W.reshape((weights,1))
    Y = Y.reshape((samples,1))
    error = (Y - np.dot(X,W))/np.sqrt(2*samples)
    loss = np.dot(np.transpose(error),error) + ((0)*np.dot(np.transpose(W),W))
    return loss[0,0]

def MoorePenroseInverse(X,Y,lamb):
    n = X.shape[0]
    m = X.shape[1]
    inverse = np.linalg.pinv(np.dot(np.transpose(X),X)+lamb*np.identity(m))
    #W = np.dot(inverse,np.dot(np.transpose(X),Y))
    W = np.dot(np.dot(inverse,np.transpose(X)), Y)
    return W
#arguments = ['c','a1_lin_data/train.csv','a1_lin_data/test_X.csv','a1_lin_data/sample_regularization.txt','outputfile.txt','weightfile.txt']
#arguments = ['c','a1_lin_data/train.csv','a1_lin_data/test_X.csv','outputfile.txt','weightfile.txt']
#arguments = ['a','a1_lin_data/train.csv','a1_lin_data/test_X.csv','outputfile.txt','weightfile.txt']

def CrossValidationSplit(X,Y):
    tenth = int(X.shape[0]/10)
    l = []
    l.append((X[tenth:tenth*10,:],Y[tenth:tenth*10],X[0:tenth,:], Y[0:tenth]))
    for i in range(8):
        X1 = X[0: tenth*(i+1),:]
        Y1 = Y[0: tenth*(i+1)]
        X2 = X[tenth*(i+2):tenth*10,:]
        Y2 = Y[tenth*(i+2):tenth*10]
        Xtr = np.concatenate((X1,X2),axis=0)
        Ytr = np.concatenate((Y1,Y2),axis=0)
        Xte = X[tenth*(i+1):tenth*(i+2),:]
        Yte = Y[tenth*(i+1):tenth*(i+2)]
        
        l.append((Xtr,Ytr,Xte,Yte))
    l.append((X[:tenth*9,:],Y[:tenth*9],X[tenth*9:tenth*10,:], Y[tenth*9:tenth*10]))
    return l

def predictionError(pred,Y):
    n = Y.shape[0]
    error = (Y - pred)
    loss = np.dot(np.transpose(error),error)/n
    return loss

def CrossValidation(l,lambda_list):
    best_lambda = lambda_list[0]
    best_loss = np.inf
    for lamb in lambda_list:
        trainloss = 0
        testloss = 0
        for i in l:
            Xtr = i[0]
            Ytr = i[1]
            Xte = i[2]
            Yte = i[3]
            W = MoorePenroseInverse(Xtr,Ytr,lamb)
            
            trainloss = trainloss + loss(Xtr,W,Ytr,lamb)
            testloss = testloss + loss(Xte,W,Yte,lamb)
        if(testloss/10 < best_loss):
            best_lambda = lamb
            best_loss = testloss/10
    return best_lambda,best_loss

def CrossValidationLasso(l,lambda_list):
    best_lasso = lambda_list[0]
    overall_best_loss = np.inf
    for lamb in lambda_list:
        testloss = 0
        for i in l:
            Xtr = i[0]
            Ytr = i[1]
            Xte = i[2]
            Yte = i[3]
            
            reg = linear_model.LassoLars(lamb)
            reg.fit(Xtr,Ytr)
            non_zero1 = [i for i in range(m) if reg.coef_[i]!=0]
            
            newX1 = Xtr[:,non_zero1]
            newXtest1 = Xte[:,non_zero1]
            
            
            newX1 = FirstBatch(newX1)
            newXtest1 = FirstBatch(newXtest1)
        
            #reg = linear_model.LassoLars(lasso2)
            reg.fit(newX1,Ytr)
            non_zero2 = [i for i in range(newX1.shape[1]) if reg.coef_[i]!=0]
            newX2 = newX1[:,non_zero2]
            newXtest2 = newXtest1[:,non_zero2]
            
            newX2 = FirstBatch(newX2)
            newXtest2 = FirstBatch(newXtest2)
            
            #reg = linear_model.LassoLars(lasso3)
            reg.fit(newX2,Ytr)
            non_zero3 = [i for i in range(newX2.shape[1]) if reg.coef_[i]!=0]
            newX3 = newX2[:,non_zero3]
            newXtest3 = newXtest2[:,non_zero3]
            
            reg.fit(newX3,Ytr)
                
            prediction = reg.predict(newXtest3)
            best_loss = predictionError(prediction,Yte)
            testloss = testloss + best_loss
            #print(best_loss)
        #print(str(lamb) + " " + str(testloss))
        if(testloss < overall_best_loss):
            overall_best_loss = testloss
            best_lasso = lamb
    return best_lasso


def FirstBatch(newX):
    #expNewX = np.exp(stats.zscore(newX, axis=0))
    #expNewX = np.exp(newX)
    logNewX = np.log(np.abs(newX) + 0.00001*np.ones((newX.shape[0],newX.shape[1])))
    sqNewX = newX*newX
    cubeNewX = sqNewX*newX
    recNewX = np.reciprocal(np.abs(newX) + 0.0001*np.ones((newX.shape[0],newX.shape[1])))
    sqrtNewX = np.sqrt(np.abs(newX))
    newX = np.concatenate((newX,logNewX,sqNewX,cubeNewX,recNewX,sqrtNewX),axis=1)
    return newX


mode = arguments[0]
if((mode == 'a')):
    trainfilename = arguments[1]
    testfilename = arguments[2]
    regularizationfilename = arguments[2]
    outputfilename = arguments[3]
    weightfilename = arguments[4]
elif(mode == 'b'):
    trainfilename = arguments[1]
    testfilename = arguments[2]
    regularizationfilename = arguments[3]
    outputfilename = arguments[4]
    weightfilename = arguments[5]
else:
    trainfilename = arguments[1]
    testfilename = arguments[2]
    regularizationfilename = arguments[2]
    outputfilename = arguments[3]
    weightfilename = arguments[3]
    

X = np.loadtxt(open(trainfilename, "rb"), delimiter=",")
np.random.shuffle(X)
n = X.shape[0]
m = X.shape[1]

Y = X[:,m-1]
ones = np.ones((n,1))
X = X[:,0:m-1]
X = np.concatenate((ones,X),axis=1)

Xtest = np.loadtxt(open(testfilename, "rb"), delimiter=",")
ntest = Xtest.shape[0]
mtest = Xtest.shape[1]+1
ones = np.ones((ntest,1))
Xtest = np.concatenate((ones,Xtest),axis=1)



#X = stats.zscore(X, axis=0)
#X = np.nan_to_num(X)
#Y = stats.zscore(Y)
#Y = np.nan_to_num(Y)

    
if(mode == 'a'):
    W = MoorePenroseInverse(X,Y,0)
    #inverse = np.linalg.inv(np.dot(np.transpose(X),X))
    #W = np.dot(np.dot(inverse,np.transpose(X)),Y)
    outfile = open(outputfilename,'w')
    Ytest = np.dot(Xtest,W)   
    for i in range(ntest):
        outfile.write(str(Ytest[i]) + "\n") 
    outfile.close()
    
    weightfile = open(weightfilename,'w')
    for i in range(m):
        weightfile.write(str(W[i]) + "\n")
    weightfile.close()

elif(mode=='b'):
    l = CrossValidationSplit(X,Y)
    
    lambda_list = []
    with open(regularizationfilename) as f:
        for line in f:
            x = line.split()
            for j in x:
                lambda_list.append(float(j))
    
    #lambda_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000,5000,10000,50000,100000,1000000]
    
    best_lambda,best_loss = CrossValidation(l,lambda_list) 
    print(best_lambda)
    #inverse = np.linalg.inv(np.dot(np.transpose(X),X) + (n*(0.0001)*np.identity(m)))
    W = MoorePenroseInverse(X,Y,best_lambda)
    
    outfile = open(outputfilename,'w')
    Ytest = np.dot(Xtest,W)   
    for i in range(Ytest.shape[0]):
        outfile.write(str(Ytest[i]) + "\n") 
    outfile.close()
    
    weightfile = open(weightfilename,'w')
    for i in range(m):
        weightfile.write(str(W[i]) + "\n")
    weightfile.close()

else:
    from sklearn import linear_model
    lasso_lamb = [0.03,0.01,0.003,0.001,0.0007,0.0003,0.0001,0.00005]
    #lasso_lamb = [0.03]
    lasso = CrossValidationLasso(CrossValidationSplit(X,Y),lasso_lamb)
    print("Lasso is: " +str(lasso))
    reg = linear_model.LassoLars(lasso)
    reg.fit(X,Y)
    non_zero1 = [i for i in range(m) if reg.coef_[i]!=0]
    
    newX1 = X[:,non_zero1]
    newXtest1 = Xtest[:,non_zero1]
    
    
    newX1 = FirstBatch(newX1)
    newXtest1 = FirstBatch(newXtest1)

    #reg = linear_model.LassoLars(lasso)
    reg.fit(newX1,Y)
    non_zero2 = [i for i in range(newX1.shape[1]) if reg.coef_[i]!=0]
    newX2 = newX1[:,non_zero2]
    newXtest2 = newXtest1[:,non_zero2]
    
    newX2 = FirstBatch(newX2)
    newXtest2 = FirstBatch(newXtest2)
    
    #reg = linear_model.LassoLars(lasso)
    reg.fit(newX2,Y)
    non_zero3 = [i for i in range(newX2.shape[1]) if reg.coef_[i]!=0]
    newX3 = newX2[:,non_zero3]
    newXtest3 = newXtest2[:,non_zero3]
    reg.fit(newX3,Y)
    #W = MoorePenroseInverse(newX,Y,0)
    outfile = open(outputfilename,'w')
    #Ytest = np.dot(newXtest,W)   
    prediction = reg.predict(newXtest3)
    predtrain = reg.predict(newX3)
    
    
    for i in range(prediction.shape[0]):
        outfile.write(str(prediction[i]) + "\n") 
    outfile.close()