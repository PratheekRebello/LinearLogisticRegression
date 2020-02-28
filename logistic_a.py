import sys
program_name = sys.argv[0]
arguments = sys.argv[1:]
import numpy as np
from scipy import special

#arguments = ['a1_log_data/train.csv','a1_log_data/test_X.csv','a1_log_data/param_a.txt','outputfile.txt','weightfile.txt']

trainfilename = arguments[0]
testfilename = arguments[1]
paramfilename = arguments[2]
outputfilename = arguments[3]
weightfilename = arguments[4]


#def softmax(X, theta = 1.0, axis = None):
#    # make X at least 2d
#    y = np.atleast_2d(X)
#
#    # find axis
#    if axis is None:
#        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
#
#    # multiply y against the theta parameter, 
#    y = y * float(theta)
#
#    # subtract the max for numerical stability
#    y = y - np.expand_dims(np.max(y, axis = axis), axis)
#
#    # exponentiate y
#    y = np.exp(y)
#
#    # take the sum along the specified axis
#    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
#
#    # finally: divide elementwise
#    p = y / ax_sum
#
#    # flatten if X was 1D
#    if len(X.shape) == 1: p = p.flatten()
#
#    return p

def softmax(X):
    X = np.exp(X)
    #column = np.sum(X,axis=1)
    X = X/X.sum(axis=1)[:,None]
    return X

def loss(X,W,Y):
    result = np.dot(X,W)
    result = softmax(result)
    result = np.log(result)
    loss = result*Y
    summ = np.sum(np.sum(loss))
    return -summ/X.shape[0]

def derivative(X,W,Y):
    result = np.dot(X,W)
    result = softmax(result)
    return np.dot(np.transpose(X),result-Y)/X.shape[0]
    #der = np.zeros((W.shape))
    #for j in range(X.shape[1]):
    #    der[j,:] = np.sum(((result - Y)*X[:,j,np.newaxis]),axis = 0)
    #return der/X.shape[0]

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

#totaltypes = 0
#onehotlist = [{} for i in range(m)]
#for i in range(m):
#    types = 0
#    for j in range(n):
#        read = data[j][i]
#        if(not(read in onehotlist[i])):
#            onehotlist[i][read] = types
#            types = types+1
#    totaltypes = totaltypes + types
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



#Xtest = Xtest[:,0:counttillnow - k] 

m = X.shape[0]
n = X.shape[1]  

ones = np.ones((m,1))
X = np.concatenate((ones,X),axis=1)
onestest = np.ones((Xtest.shape[0],1))
Xtest = np.concatenate((onestest,Xtest),axis=1)
n = n+1  

#W = np.random.rand(n,k)*np.sqrt(2/(n*k))
W = np.zeros((n,k))

f = open(paramfilename,"r")
f1 = f.readlines()
mode = int(f1[0].strip())
if((mode == 1)|(mode==2)):
    etan = float(f1[1].strip())
else:
    spl = f1[1].split(',')
    etan = float(spl[0].strip())
    alpha = float(spl[1].strip())
    beta = float(spl[2].strip())
max_iterations = int(f1[2].strip())

#alpha = 0.5
#beta = 0.8

#train = X[:5000,:]
#Ytrain = Y[:5000,:]
#Xtest = X[5000:,:]
#Ytest = Y[5000:,:]
#eta = 0.1
#etan = 1
if(mode==1):
    for i in range(max_iterations):
        eta = etan
        #print(eta)
        W = W - eta*derivative(X,W,Y)
        #print(str(i) + " " + str(loss(X,W,Y)))
if(mode==2):
    for i in range(max_iterations):
        eta = etan/np.sqrt(i+1)
        W = W - eta*derivative(X,W,Y)
        #print(loss(X,W,Y))
if(mode==3):
    for i in range(max_iterations):
        eta = etan
        fW = loss(X,W,Y)
        gradfW = derivative(X,W,Y)
        fW_gradfW = loss(X,W-gradfW,Y)
        while(fW_gradfW > fW - (eta*alpha)*np.dot(np.transpose(gradfW),gradfW)[0,0]):
            eta = eta*beta
        W = W - eta*derivative(X,W,Y)
        #print(loss(X,W,Y))

prediction = softmax(np.dot(Xtest,W))
actual = (np.argmax(prediction,axis = 1) )
prediction = []
#print(actual)
inv_map = {v: k for k, v in onehotlist[len(onehotlist)-1].items()}
for i in range(actual.shape[0]):
    prediction.append(str(inv_map[actual[i]]).replace("'",""))
    
import csv

with open(outputfilename, 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(prediction)
#np.savetxt(outputfilename, prediction, delimiter=",")
np.savetxt(weightfilename, W, delimiter=",")