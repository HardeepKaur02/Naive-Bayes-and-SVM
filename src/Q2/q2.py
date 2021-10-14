import numpy as np
import csv
from cvxopt import matrix
from cvxopt import solvers
import pandas as pd
import time
import json
import math
from libsvm.svmutil import *
import matplotlib.pyplot as plt
import sys
import os
from SVM_Lin import Linear_SVM
from SVM_Gaussian import Gaussian_SVM
from Libsvm_Lin import Linear_LIBSVM
from Libsvm_Gaussian import Gaussian_LIBSVM
from Learn_models import Learn_Models
from Predictions import MultiClassifier
from Multi_Libsvm import MultiClassifier_LIBSVM
from Cross_Validation import CrossValidation
n = 784
C = 1.0

##### function to obtain confusion matrix from test labels and model predictions #####

def Confusion_Matrix(y_test,pred,r):
    m_test = len(y_test)
    conf_mat = [ [0 for i in range(r)] for i in range(r) ]
    for i in range(m_test):
        conf_mat[int(y_test[i]) - 4][int(pred[i]) - 4] += 1
    return conf_mat

# plotting the confusion matrix
def draw_confusion(confatrix,title_str,d = 1):
    r = len(confatrix)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, r, 1))
    ax.set_yticks(np.arange(0, r, 1))
    ax.set_xticklabels(np.arange(d, r+d, 1))
    ax.set_yticklabels(np.arange(d, r+d, 1))
    plt.imshow(confatrix)
    plt.title(title_str)
    plt.colorbar()
    plt.set_cmap("Purples")
    plt.ylabel("True labels")
    plt.xlabel("Predicted label")
    plt.show()        


##### driver code #####

def main():
    print(len(sys.argv))
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    classification = sys.argv[3]
    part = sys.argv[4]
    binary = (classification=='0')
    start = time.time()
    # entry number : 2019CS10354    
    d = 4.0
    # image size: 28x28, number of features = 784
    n = 784 
    # noise parameter
    C = 1.0

    
    if binary == True:
    
        if part == 'a':
            st =  "Confusion Matrix - Binary Classifier for digits " + str(d) + " and " + str(d+1) 
            print(st)
            print("TRAINING DATA SET")
            accuracy, conf_mat = Linear_SVM(d,d+1,n,C,train_file_name,train_file_name)
            print("Accuracy: " + str(accuracy))
            print("TEST DATA SET")
            accuracy, conf_mat = Linear_SVM(d,d+1,n,C,train_file_name,test_file_name)
            print("Accuracy: " + str(accuracy))
            print(conf_mat)
            draw_confusion(conf_mat,st,d)
        if part == 'b':
            gamma = 0.05
            st =  "GAUSSIAN SVM FOR BINARY CLASSIFICATION"
            print(st)
            print("TRAINING DATA SET")
            accuracy, conf_mat = Gaussian_SVM(d,d+1,gamma,n,C,train_file_name,train_file_name)
            print("Accuracy: " + str(accuracy))
            print("TEST DATA SET")
            accuracy,conf_mat = Gaussian_SVM(d,d+1,gamma,n,C,train_file_name,test_file_name)
            print("Accuracy: " + str(accuracy))
            print(conf_mat)
            draw_confusion(conf_mat,st,d)
        if part == 'c':
            st =  "LINEAR SVM FOR BINARY CLASSIFICATION"
            print(st)
            print("TRAINING DATA SET")
            accuracy = Linear_LIBSVM(d,d+1,n,C,train_file_name,train_file_name)
            print("Accuracy: " + str(accuracy))
            print("TEST DATA SET")
            accuracy = Linear_LIBSVM(d,d+1,n,C,train_file_name,test_file_name)
            print("Accuracy: " + str(accuracy))

            gamma = 0.05
            st =  "GAUSSIAN SVM FOR BINARY CLASSIFICATION"
            print(st)
            print("TRAINING DATA SET")
            accuracy = Gaussian_LIBSVM(d,d+1,gamma,n,C,train_file_name,train_file_name)
            print("Accuracy: " + str(accuracy))
            print("TEST DATA SET")
            accuracy = Gaussian_LIBSVM(d,d+1,gamma,n,C,train_file_name,test_file_name)
            print("Accuracy: " + str(accuracy))
    else:
        if part == 'a':
            st = "ONE vs ONE GAUSSIAN MULTI CLASSIFIER"
            print(st)
            ##### UNCOMMENT TO LEARN MODELS AGAIN #####
            #Learn_Models(train_file_name)
            print("TRAINING DATA SET")
            accuracy, conf_mat = MultiClassifier(train_file_name)
            print(conf_mat)
            print("Accuracy: " + str(accuracy))
            print("TEST DATA SET")
            accuracy,conf_mat = MultiClassifier(test_file_name,1)
            print("Accuracy: " + str(accuracy))
            print(conf_mat)
            draw_confusion(conf_mat,st,d = 0)
        
        if part == 'b':
            st = "LIBSVM MULTI CLASSIFIER"
            print(st)
            MultiClassifier_LIBSVM(train_file_name,train_file_name)
        if part == 'd':
            st = "5- FOLD CROSS VALIDATION"
            print(st)
            CrossValidation(n,train_file_name,test_file_name)
            x = [0.00001,0.001,1,5,10]
            x = list(map(np.log, x))
            print(x)
            y1 = [9.4,9.4,97.75,97.8,97.8]
            y2 = [72.08,72.08,97.23,97.3,97.3]
            plt.plot(x, y1, label = "Max. Validation Set Accuracy", color = 'purple')
            plt.plot(x, y2, label = "Test Set Accuracy", color = 'xkcd:sky blue')            
            plt.xlabel('log C')
            plt.ylabel('Accuracy (%)')
            plt.title( 'Choosing the value of C')
            plt.legend()
            plt.savefig('Accuracies_vs_C')
            plt.show()


        
if __name__ == "__main__":
	main()            









