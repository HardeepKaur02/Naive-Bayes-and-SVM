import numpy as np
import math
from libsvm.svmutil import *
import time
import csv

def Gaussian_LIBSVM(d1, d2, gamma, n, C, train_file_name,test_file_name):
    
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

    X = np.append(Xd1,Xd2,0)       # [X] = mxn
    m = X.shape[0]               # number of training examples
    Y=np.concatenate((np.ones(nd1)*(-1),np.ones(nd2)),axis=0)
    prob = svm_problem(Y,X)
    start = time.time()
    param = ' -t 2 -c ' + str(C) + ' -g ' + str(gamma)
    model = svm_train(prob,param)
    end = time.time()
    print("Time taken to learn the model: "+ str(end-start))

    start = time.time()
    X_test = []
    y_test = []

    file_test = open(test_file_name)
    csvReader = csv.reader(file_test)

    for row in csvReader:
        row_val = list(map(float, row))
        ## down scale data to range [0,1] from [0,255] ##
        if row_val[-1] == d1 or row_val[-1] == d2:
            test_x = list(map( lambda x: x/255 ,row_val[:n]))
            X_test.append(test_x)
            label = 0
            if row_val[n] == d1:
                label = -1
            else:
                label = 1   
            y_test.append(label)
            
    p_label, p_acc, p_val = svm_predict(y_test, X_test, model)
    end = time.time()
    print("Time taken to make predictions: "+ str(end-start))
    return p_acc[0]