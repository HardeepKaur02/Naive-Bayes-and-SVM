import numpy as np
import math
from libsvm.svmutil import *
import time
import csv


def CrossValidation(n,train_file_name,test_file_name):
    start = time.time()

    ##### storing the training data ####
    C = [0.00001,0.001,1,5,10]

    X = []
    Y = []
    file_train = open(train_file_name)
    csvReader = csv.reader(file_train)
    for row in csvReader:
        row_val = list(map(float, row))
        ## down scale data to range [0,1] from [0,255] ##
        X.append(list(map( lambda x: x/255.0 ,row_val[:n])))
        Y.append(row_val[-1])
    file_train.close()

    X_test = []
    y_test = []

    file_test = open(test_file_name)
    csvReader = csv.reader(file_test)

    for row in csvReader:
        row_val = list(map(float, row))
        ## down scale data to range [0,1] from [0,255] ##
        test_x = list(map( lambda x: x/255 ,row_val[:n]))
        X_test.append(test_x)
        y_test.append(row_val[-1])

        # first shuffle x_train, y_train
        m = len(Y)
        X = np.array(X).reshape(m,-1)
        Y = np.array(Y)
        p = np.random.permutation(X.shape[0])
        X = X[p]
        Y = Y[p]

        accuracies = [ [0 for i in range(6)] for i in range(5)] 
        

    r = m//5
    for i in range(5):
        X = np.array(X).reshape(m,-1)
        vX = X[i*r : (i+1)*r,:]
        vY = Y[i*r : (i+1)*r]
        
        trainX = np.concatenate((X[0:i*r,:], X[(i+1)*r : , :]), axis = 0)
        trainY = np.concatenate((Y[0:i*r] , Y[(i+1)*r :]), axis = 0)
        
        prob = svm_problem(trainY,trainX)

        m1 = svm_train(prob,'-g 0.05 -t 2 -c 0.00001 -h 0 ')
        accuracies[0][i] = svm_predict(vY,vX,m1)[1]

        m2 = svm_train(prob,'-g 0.05 -t 2 -c 0.001 -h 0')
        accuracies[1][i] = svm_predict(vY,vX,m2)[1]

        m3 = svm_train(prob,'-g 0.05 -t 2 -c 1 -h 0')
        accuracies[2][i] = svm_predict(vY,vX,m3)[1]

        m4 = svm_train(prob,'-g 0.05 -t 2 -c 5 -h 0')
        accuracies[3][i] = svm_predict(vY,vX,m4)[1]

        m5 = svm_train(prob,'-g 0.05 -t 2 -c 10 -h 0')
        accuracies[4][i] = svm_predict(vY,vX,m5)[1]

    for i in range(5):
        prob = svm_problem(Y,X)
        param = '-g 0.05 -t 2 -h 0 -c ' + str(C[i])
        model = svm_train(prob, param)
        accuracies[i][-1] = svm_predict(y_test,X_test,model)[1]



    for i in range(5):
        print("Validation set accuracies for C = " + str(C[i]))
        temp = [accuracies[i][j] for j in range(5)]
        print(*temp)
        print("Test set accuracy for C = " + str(C[i]))
        print(accuracies[i][-1])


#C = 0.00001   batch_1, batch_2, batch_3, batch_4, batch_5, test
#C = 0.001     batch_1, batch_2, batch_3, batch_4, batch_5, test
#C = 1         batch_1, batch_2, batch_3, batch_4, batch_5, test
#C = 5         batch_1, batch_2, batch_3, batch_4, batch_5, test
#C = 10        batch_1, batch_2, batch_3, batch_4, batch_5, test

