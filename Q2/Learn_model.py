import numpy as np
import csv
from cvxopt import matrix
from cvxopt import solvers
import pandas as pd
import time
import json
import os
dirname = os.path.dirname(__file__)
# image size: 28x28, number of features = 784
n = 784 
# noise parameter
C = 1.0

def KernelMat(X, gamma = 0.05):
    norm = np.linalg.norm(X,axis = 1, keepdims = True)
    norm_sq = norm*norm
    norm_sq_exp = np.exp(-1*gamma*norm_sq)
    dot_mat = np.exp(2*gamma*(X.dot(X.T)))
    kernel_mat = norm_sq_exp*dot_mat*(norm_sq_exp.T)
    return kernel_mat

def learn_model(d1,d2,train_file_name, verbose = 0 ):

##### storing the training data ####

    # start timer
    start = time.time()
    Xd1 = [] # training data with label = d1
    Xd2 = [] # training data with label = d2
    file_train = open(train_file_name)
    csvReader = csv.reader(file_train)
    for row in csvReader:
        row_val = list(map(float, row))
        ## down scale data to range [0,1] from [0,255] ##
        if row_val[-1] == d1:
            Xd1.append(list(map( lambda x: x/255.0 ,row_val[:n])))
        elif row_val[-1] == d2:
            Xd2.append(list(map( lambda x: x/255.0 ,row_val[:n])))
    file_train.close()

    nd1 = len(Xd1)
    nd2 = len(Xd2)
    Xd1 = np.array(Xd1).reshape(nd1,-1)
    Xd2 = np.array(Xd2).reshape(nd2,-1)

    X = np.append(Xd1,Xd2,0)       # [X] = mxn
    m = X.shape[0]               # number of training examples
    y = np.concatenate((np.ones((nd1,1))*(-1.0), np.ones((nd2,1))), axis = 0)
    K = KernelMat(X)

    I = np.identity(m)
    I_bar = I * -1

    P = K * np.outer(y,y)       # element wise multiplication
    q = np.ones((m,1))*(-1)
    A = np.concatenate((np.ones((nd1,1))*(-1.0),np.ones((nd2,1))*1.0),axis=0).T
    b = np.array([0]).reshape(1,1)
    G = np.append(I,I_bar,0)                            # [G] = 2m x m
    h = [C for i in range(m)] + [0 for i in range(m)]   # [h] = 2m x 1

    P = matrix(P, tc = 'd')
    q = matrix(q, tc = 'd')
    G = matrix(G, tc = 'd')
    h = matrix(h, tc = 'd')
    A = matrix(A, tc = 'd')
    b = matrix(b, tc = 'd')

    sol = solvers.qp(P,q,G,h,A,b)
    alpha = sol['x']
    alpha = np.array(alpha)
    sv=X[np.ix_(alpha[:,0]>0.0000001,)]  # support vectors 
    sv_y = y[np.ix_(alpha[:,0]>0.0000001,)]       # nsvx1
    alpha_sv = alpha[np.ix_(alpha[:,0]>0.0000001,)]  # nsvx1
    print("No. of support vectors: " + str(sv.shape[0]))

    # end timer
    end = time.time()
    print("Time taken to learn the model: " + str(end-start))

    b1 = float("inf")
    b2 = float("-inf")
    for i in range(m):
        if alpha[i] > 0.00001:
            val = 0
            for j in range(m):
                if alpha[j] > 0.000001:
                    val += alpha[j] * y[j] * K[i][j]
            if i<nd1:  # class label = -1: d = 4
                b1 = min(b1, -1 - val)
            else:
                b2 = max(b2, 1- val)
    b_intercept = 0.5 * (b1+b2)
    print("b = " + str(b_intercept))
    
    model = {"alpha_sv" : alpha_sv, "sv": sv, "sv_y" : sv_y, "b" : b_intercept}
    model_name = "Models/model"+str(d1)+str(d2)+".json"
    model_path = os.path.join(dirname,model_name)
    print(model_path)
    # Serializing json 
    json_object = json.dumps(model, cls=NumpyEncoder)
    
    # Writing to sample.json
    with open(model_path, "w") as outfile:
        outfile.write(json_object)    

##### to store numpy ndarrays as json file #####

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)