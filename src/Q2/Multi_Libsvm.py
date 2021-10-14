import numpy as np
import math
from libsvm.svmutil import *
import time
import csv
import matplotlib.pyplot as plt
# image size: 28x28, number of features = 784
n = 784 
# noise parameter
C = 1.0

##### Multiclass classification for 10 classes 0-9 #####
def multiclass_svm(X,Y,X_test,y_test):
    prob = svm_problem(Y,X)
    start = time.time()
    model = svm_train(prob,'-g 0.05 -t 2 -c 1')
    end = time.time()
    print("Time taken to learn the model: "+ str(end-start))

    p_label, p_acc, p_val = svm_predict(y_test, X_test, model)
    print("Accuracy: " + str(p_acc))

    num_classes = 10
    m_test = len(y_test)
    conf_mat = [[0 for i in range(num_classes)] for i in range(num_classes)]
    for i in range(m_test):
        conf_mat[int(y_test[i])][int(p_label[i])] += 1
    return conf_mat, p_label


##### Multiclass classification using 45 individual libsvms i.e. one vs one classification #####
def multiclass_svm_45(X,Y,X_test,y_test):
    m_test = len(y_test)
    start = time.time()
    prediction_count = dict()
    num_classes = 10
    for i in range(len(y_test)):
        prediction_count[i] = [0 for j in range(10)]
    prediction = [0 for i in range(m_test)]

	# learning parameters phase (45 individual svms)
    for i in range(num_classes):
        for j in range(i):
            X_subset = []
            Y_subset = []
            for k in range(len(Y)):
                if Y[k] == i or Y[k] == j:
                    X_subset.append(X[k])
                    Y_subset.append(Y[k])
            
            idx = str(i)+str(j)
			
            prob = svm_problem(Y_subset,X_subset)
            model = svm_train(prob,'-g 0.05 -t 2 -c 1')
            p_label, p_acc, p_val = svm_predict(y_test,X_test,model)
			
            for k in range(len(p_label)):
                if p_label[k] == 1:
                    prediction_count[k][i]+=1
                else:
                    prediction_count[k][j]+=1

    end = time.time()
    print("Time taken to learn the models: "+ str(end-start))

    conf_mat = [[0 for i in range(num_classes)] for i in range(num_classes) ]
    accuracy = 0
    for i in range(len(y_test)):
        prediction[i] = np.argmax(prediction_count[i])
        y_label = int(y_test[i])
        if y_label == prediction[i]:
            accuracy+=1
           
    accuracy = accuracy*100 /len(y_test)
    print("Accuracy: " + str(accuracy))

    for i in range(m_test):
        y_label = int(y_test[i])
        conf_mat[y_label][prediction[i]] += 1 

    return(conf_mat, prediction)

# plotting the confusion matrix
def draw_confusion(confatrix,title_str):
    plt.imshow(confatrix)
    plt.title(title_str)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_yticks(np.arange(0, 10, 1))
    ax.set_xticklabels(np.arange(0, 10, 1))
    ax.set_yticklabels(np.arange(0, 10, 1))
    plt.colorbar()
    plt.set_cmap("Purples")
    plt.ylabel("True labels")
    plt.xlabel("Predicted label")
    plt.show()        

def MultiClassifier_LIBSVM(train_file_name,test_file_name):

    ##### storing the training data ####

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

    ##### storing the test data ####

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
    file_test.close()

    ##### driver code #####
    st = "Confusion Matrix Multi Classifier LIBSVM"
    print("TRAINING DATA SET")
    conf_mat, predictions = multiclass_svm(X,Y,X,Y)
    print(conf_mat)
    #print("Waheguru")
    print("TEST DATA SET")
    conf_mat, predictions = multiclass_svm(X,Y,X_test,y_test)
    print(conf_mat)
    #print("Waheguru")
    draw_confusion(conf_mat,st)

