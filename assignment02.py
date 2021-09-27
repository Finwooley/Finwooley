# -*- coding: utf-8 -*-
"""
File:   assignment01.py
Author: James Dooley
Date:   Sept. 19, 2021
Desc:   Assignment 02 python code
    
"""


""" =======================  Import dependencies ========================== """
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.spatial.distance import cdist
from sklearn.linear_model import ElasticNet

plt.close('all') #close any open plots
x=5
""" ======================  Function definitions ========================== """
def assign2(x):
    return 3*(x + np.sin(x)) * np.exp(-x**2.0)

def gaussian(x, s=0.01):
    return np.exp(-x * x /(2* np.power(s, 2.0)))

def linear(r):
    return r

class RBF(object):
    

    def __init__(self, centers, truth, kernel,alpha = .1, l1 = .4, Phi = None, method = 'minimize', descent = 'Powell'):
        '''
        centers - rbf centers, or M values chosen
        truth - expected values for a given X
        kernel - distance evaluation formula as a lambda function
        '''
        
        l2 = 1 - l1
        cenCt = len(centers)
        if Phi is None:
            Phi = np.zeros((cenCt, cenCt))
            for i in range(cenCt):
                for j in range(i, cenCt):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    Phi[i, j] = Phi[j, i] = kernel(dist)
                    if i == j:
                        Phi[i,j]=Phi[i,j] + .001          

        def errorFunc(w,Phi,centers, truth, alpha = .5, l1=.5, l2 = .5 ):
                return (np.linalg.norm(np.asarray(Phi@w-truth)+.000001,2)**2)+alpha*l1*np.sum(np.abs(w))+alpha*l2*np.linalg.norm(w,2)**2

        self.truth=truth
        self.Phi = Phi
        self.centers = centers
        self.kernel = kernel
        w = np.linalg.inv(alpha*l2*np.eye(cenCt)+Phi.T@Phi)@(Phi.T@truth-alpha*l1)
        
        wSum = np.sum(w)
        wSign = wSum/(np.abs(wSum)+.00001)
        w2=truth
        #w2 = np.linalg.inv(alpha*l2*np.eye(cenCt)+Phi.T@Phi)@(Phi.T@truth-wSign*alpha*l1)

        compare = mini = enet = closedForm = False
        
        if method == 'enet':
            enet = True
        
        if method == 'minimize':
            mini = True

        if method == 'closedForm':
            closedForm = True

        if method == 'compare':
            compare = True
            mini = True
        
        bounds = []
        for i in truth:
            bounds.append((-np.abs(i*1.1),np.abs(i*1.1)))

        if enet:

            self.alpha = a+b
            self.l1R = (a/(np.abs(a+b)+.00000001))
            a = l1*alpha
            b = 2*alpha*l2
            if self.l1R >= 1:
                self.l1R = .999999
            elif self.l1R <= 0:
                self.l1R = .000001

            print("l1 ratio is" + str(self.l1R))
            print("alpha is" + str(self.alpha))
            wmin2 = ElasticNet(alpha = self.alpha,l1_ratio=self.l1R, copy_X= True, fit_intercept= True, normalize= True ,warm_start=False, max_iter=10000)
            wmin2.fit(Phi,truth)
            self.w = wmin2.coef_
            print(wmin2.coef_)
            print(np.sum(wmin2.coef_) - np.sum(w2))

        if mini:
            wmin = minimize(errorFunc, w2,(self.Phi,self.centers, self.truth, alpha, l1, l2),method=descent,options={'maxiter':2000}, bounds=bounds)
            self.w = wmin.x
            
            

        if closedForm:
            self.w = w2

        if compare:
            print(np.linalg.norm(wmin.x-w2,2))

    def evaluate(self, centers):
        linearDists = centers[:, np.newaxis] - self.centers
        Phi = self.kernel(linearDists)
        w= self.w
        return np.matmul(w,Phi.T)
        

""" =======================  Load Training Data ======================= """
data_uniform = np.load('data_set.npz')
train, validate, test = data_uniform.values()
x1 = train[:,0]
t1 = train[:,1]
M =  len(x1)
s = .01
rbfDict = {}
rbfDictCl = {}
centers = x1
truth = t1
kernel = lambda x:gaussian(x)
cenCt = len(centers)
Phi = np.zeros((cenCt, cenCt))
for i in range(cenCt):
    for j in range(i, cenCt):
        dist = np.linalg.norm(centers[i] - centers[j])
        Phi[i, j] = Phi[j, i] = kernel(dist)
        if i == j:
            Phi[i,j]=Phi[i,j] + .001 

l1 = np.linspace(0,1,101,endpoint=True)
alpha = np.linspace(0,10,11, endpoint = True)

testOutput = np.zeros((np.shape(l1)[0],np.shape(alpha)[0]))

#myColors = np.linspace(.3,1,20)

""" ========================  Train the Model ============================= """

#for run in range(0,10):
    #rbfdict[run] = RBF(x1[Mb],t1[Mb],lambda x:gaussian(x))

for i in l1:
    for j in alpha:
        rbfDict[str(i)+"XX"+str(j)] = RBF(x1,t1,lambda x:gaussian(x),alpha = j, l1 = i, Phi = Phi, method='minimize', descent='Powell' )



""" ======================== Load Test Data  and Test the Model =========================== """

truth = -(x1 + np.sin(x1)) * np.exp(-x1**2.0)



dt = np.linspace(-4,4,1003)
truth = assign2(dt)



""" ========================  Plot Results ============================== """
#First run where M was varied and distributed evenly
'''j = 0
for i in M:
    plt.plot(data_test,y1A[i],color = (myColors[j],myColors[j]-.2,.1,.5), label = 'M = ' + str(i))
    j = j + 1

p2 = plt.plot(x1,t1,'o', label = 'Training')
'''
#add title, legend and axes labels
plt.ylabel('t') #label x and y axes
plt.xlabel('x')
plt.subplots_adjust(top = .976, bottom = .069, left = .054, right = .987, hspace = .2, wspace = .2)
p3 = plt.plot(dt,truth,color = (.1,.5,.1,.5), label = 'Truth')
plt.ylim(-2.5,5)
plt.legend()
plt.show()

plt.clf()
