import numpy as np
import csv 
import json
import matplotlib.pyplot as plt
from PIL import Image
import os
dirname = os.path.dirname(__file__)

n = 784
C = 1.0

def predictor(X_test,m_test, d1,d2,gamma = 0.05):
    model_name = "Models/model"+str(d1)+str(d2)+".json"
    model_path = os.path.join(dirname,model_name)
    with open(model_path, 'r') as openfile:
        model = json.load(openfile)
    
    alpha_sv = np.asarray(model["alpha_sv"])
    sv = np.asarray(model["sv"])
    sv_y = np.asarray(model["sv_y"])
    b = np.asarray(model["b"])
    b = b[0]    
    norm_sq = np.linalg.norm(X_test,axis = 1, keepdims = True)**2
    norm_sq_exp = np.exp(-1*gamma*norm_sq)
     
    norm_sq_sv = np.linalg.norm(sv, axis = 1, keepdims = True)**2
    norm_sq_exp_sv = np.exp(-1*gamma*norm_sq_sv)

    dot_mat = X_test.dot(sv.T)
    exp_dot_mat = np.exp(2*gamma*dot_mat)

    predictions = np.zeros(m_test)
    predictions = norm_sq_exp*(exp_dot_mat.dot(sv_y*alpha_sv*norm_sq_exp_sv)) + b
    
    predictions[predictions >= 0.0 ] = d2
    predictions[predictions < 0.0] = d1
    predictions = np.ravel(predictions)
    return predictions

def MultiClassifier(test_file_name, visualise = 0):
    X_test = []
    y_test = []
    X = []
    file_test = open(test_file_name)
    csvReader = csv.reader(file_test)
    for row in csvReader:
        row_val = list(map(float, row))
        test_x = list(map( lambda x: x/255 ,row_val[:n]))
        X_test.append(test_x)
        y_test.append(row_val[-1])
        X.append(row_val[:n])
    m_test = len(X_test)
    X_test = np.array(X_test).reshape(m_test,n)
    
    model_predictions = np.zeros((m_test,45))
    k=0
    for i in range(10):
        for j in range(i+1,10):
            model_predictions[:,k] = list(map(int, predictor(X_test,m_test, i,j)))
            k=k+1

#    np.savetxt("predictions.csv", model_predictions)
    predictions = []
    conf_mat = np.zeros((10,10),dtype=int)
    correct = 0
    for i in range(m_test):
        actual = (int)(y_test[i])
        b = np.bincount(list(map(int, model_predictions[i])), minlength = 10)
        maxm = -1
        prev_max = -1
        for j in range(10):
            if b[j] > maxm:
                maxm = b[j]
                prev_max = j
            elif b[j] == maxm:
                model_num = 0
                p1,p2 = int(min(prev_max,j)), int(max(prev_max,j))
                for q1 in range(p1):
                    for q2 in range(q1+1, 10):
                        model_num+=1        
                model_num += (p2-p1) - 1 # model of prev_max, j
                prev_max = int(model_predictions[i][model_num])
        prediction = prev_max
        predictions.append(prediction)
        if prediction == actual:
            correct += 1
        conf_mat[actual][prediction] += 1

    accuracy = correct*100/m_test
    draw_confusion(conf_mat,"Confusion Matrix for One vs One Multi Classifier")

    if visualise:
        ###### plot misclassified digits #####
        digits = [0 for i in range(10)]
        for i in range(10):
            dig = 0
            count = 0
            for j in range(10):
                if i != j and conf_mat[i][j] > count:
                    count = conf_mat[i][j]
                    dig = j
            digits[i] = dig
        #### get corresponding imgPixels for label =i, pred = digits[i] ####    
        y_test = np.array(y_test)
        predictions = np.array(predictions)
        X = np.array(X).reshape(m_test,-1)
        images = []
        for i in range(10):
            j = digits[i]
            imgPixel = X[ (y_test == i) & (predictions == j) ][1]
            img = imgPixel.reshape((28,28))
            img = np.array(img, dtype = np.uint8)
            img = Image.fromarray(img)
            img_name = "Img" + str(i) + str(j) + ".png"
            img.save(img_name)
            
    print("Image files generated for mis classiifed digits.")
    return accuracy,conf_mat

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

