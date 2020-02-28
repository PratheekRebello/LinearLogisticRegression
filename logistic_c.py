import sys

"""
Created on Sun Aug 18 19:23:45 2019

@author: pratheek
"""

import sys
program_name = sys.argv[0]
arguments = sys.argv[1:]
import numpy as np
from scipy import special
from random import shuffle

#arguments = ['a1_log_data/train.csv','a1_log_data/test_X.csv','a1_log_data/param_b.txt','outputfile.txt','weightfile.txt']

trainfilename = arguments[0]
testfilename = arguments[1]
outputfilename = arguments[2]
#print(outputfilename)
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

def softmax(X, theta = 1.0, axis = None):
    y = np.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1: p = p.flatten()

    return p


def loss(X,W,Y):
    result = np.dot(X,W)
    result = softmax(result,axis=1)
    result = np.log(result)
    loss = result*Y
    summ = np.sum(np.sum(loss))
    return -summ/X.shape[0]

def derivative(X,W,Y,lamb):
    result = np.dot(X,W)
    result = softmax(result,axis=1)
    return np.dot(np.transpose(X),(result-Y))/X.shape[0] - lamb*W

def accuracy(X,W,Y):
    result = np.dot(X,W)
    result = softmax(result,axis=1)
    result = np.argmax(result,axis=1)
    actual = np.argmax(Y,axis = 1) 
    return np.sum(result==actual)/X.shape[0]

import csv

with open(trainfilename, newline='') as csvfile:
    data = list(csv.reader(csvfile))
with open(testfilename, newline='') as csvfile:
    datatest = list(csv.reader(csvfile))

n = len(data)
m = len(data[0])


X = np.zeros((n,32))

onehotlist = [{'usual': 0, 'pretentious': 1, 'great_pret': 2},
 {'critical': 3, 'less_proper': 1, 'improper': 2, 'very_crit': 4, 'proper': 0},
 {'incomplete': 2, 'completed': 1, 'foster': 3, 'complete': 0},
 {'3': 2, '1': 0, '2': 1, 'more': 3},
 {'convenient': 0, 'critical': 2, 'less_conv': 1},
 {'inconv': 1, 'convenient': 0},
 {'nonprob': 0, 'slightly_prob': 1, 'problematic': 2},
 {'recommended': 0, 'not_recom': 2, 'priority': 1},
 {'priority': 3,
  'not_recom': 0,
  'spec_prior': 4,
  'very_recom': 2,
  'recommend': 1}]
counttillnow = 0
for i in range(m):
    types = 0
    for j in range(n):
        try:
            X[j][onehotlist[i][data[j][i]] + counttillnow] = 1
        except:
            continue

    counttillnow = counttillnow + len(onehotlist[i])

k = len(onehotlist[m-1])
Y = X[:,counttillnow - k:] 
X = X[:,0:counttillnow - k] 

Xtest = np.zeros((len(datatest),32-k))
counttillnow = 0
for i in range(m-1):
    types = 0
    for j in range(len(datatest)):
        #print(j)
        try:
            Xtest[j][onehotlist[i][datatest[j][i]] + counttillnow] = 1
        except:
            continue

    counttillnow = counttillnow + len(onehotlist[i])



m = X.shape[0]
n = X.shape[1]  

ones = np.ones((m,1))
X = np.concatenate((ones,X),axis=1)
onestest = np.ones((Xtest.shape[0],1))
Xtest = np.concatenate((onestest,Xtest),axis=1)
n = n+1  

lis = []
for i in range(m):
    if(Y[i][1] == 1):
        lis.append(i)
print(lis)

etan = 0.1
max_iterations = 20000
batchsize = 6000
Xtrain = X[:6000,:]
Ytrain = Y[:6000,:]
Xtestt = X[5400:,:]
Ytestt = Y[5400:,:]

W = np.random.rand(n,k)*np.sqrt(2/(n*k))
best_test_loss = np.inf
bestW = W

for i in range(max_iterations):
    eta = etan
    for j in range(int(X.shape[0]/batchsize)):
        W = W - eta*derivative(Xtrain[j*batchsize:(j+1)*batchsize,:],W,Ytrain[j*batchsize:(j+1)*batchsize,:],0.001)            
    print(str(i) + " " + str(loss(Xtrain,W,Ytrain)) + " " +str(loss(Xtestt,W,Ytestt)))
    if(loss(Xtestt,W,Ytestt) < best_test_loss):
        best_test_loss = loss(Xtrain,W,Ytrain)
        bestW = W
print(best_test_loss)


prediction = np.dot(Xtest,bestW)
actual = np.argmax(prediction,axis = 1) 
prediction = []
inv_map = {v: k for k, v in onehotlist[len(onehotlist)-1].items()}
for i in range(actual.shape[0]):
    prediction.append(str(inv_map[actual[i]]).replace("'",""))
  
import csv

with open(outputfilename, 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(prediction)