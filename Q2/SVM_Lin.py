import numpy as np
import csv
from cvxopt import matrix
from cvxopt import solvers
import pandas as pd
import time

def Linear_SVM(d1,d2,n,C,train_file_name,test_file_name):

    start = time.time()

    ##### storing the training data ####

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

    ##### Optimizing the SVM dual objective using CVXOPT #####

    Xd1 = Xd1 * -1

    X = np.append(Xd1,Xd2,0)       # [X] = mxn
    m = X.shape[0]               # number of training examples

    I = np.identity(m)
    I_bar = I * -1

    P = np.dot(X,X.T)
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
    output_file = "Linear_SV" + str(int(d1)) + str(int(d2)) + ".csv"
    np.savetxt(output_file, sv)
    print("No. of support vectors: " + str(sv.shape[0]))

    # end timer
    end = time.time()
    print("Time taken to learn the model: " + str(end-start))

    ##### Calculating parameters of hyperplane #####

    w = np.dot(np.transpose(X),alpha)    # [w] = nx1
    b1 = float("inf")
    b2 = float("-inf")
    for i in range(m):
        if alpha[i] > 0.00001:
            if i<nd1:  # class label = -1: d = d1
                b1 = min(b1, -1 + np.dot(X[i],w))
            else:
                b2 = max(b2, 1- np.dot(X[i], w))
    b_intercept = 0.5 * (b1+b2)

    print("b = " + str(b_intercept))

    ##### Making predictions on test data #####

    # start timer
    start = time.time()

    X_test = []
    y_test = []
    predictions = []
    conf_mat = [[0,0], [0,0]]
    file_test = open(test_file_name)
    csvReader = csv.reader(file_test)
    accuracy = 0
    for row in csvReader:
        row_val = list(map(float, row))
        ## down scale data to range [0,1] from [0,255] ##
        if row_val[-1] == d1 or row_val[-1] == d2:
            test_x = list(map( lambda x: x/255 ,row_val[:n]))
            X_test.append(test_x)
            pred = np.dot(test_x,w) + b_intercept
            if pred > 0.0:
                pred = 1
            else:
                pred = -1
            label = 0
            if row_val[n] == d1:
                label = -1
            else:
                label = 1   
            y_test.append(label)
            predictions.append(pred)
            col = max(0,pred)
            row = max(0,label)
            conf_mat[row][col] += 1
            if label == pred:
                accuracy +=1
    file_test.close()
    m_test = len(y_test)
    accuracy = accuracy*100/m_test
    # end timer
    end = time.time()
    print("Time taken to make predictions: " + str(end-start))
    return accuracy, conf_mat
