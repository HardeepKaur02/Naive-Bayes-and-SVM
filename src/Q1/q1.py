import pandas as pd
import numpy as np
import random
import re
import sys
import os
import time
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from collections import Counter
import json
basePath = os.path.dirname(os.path.abspath(__file__))

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def cleanPunctuation(s):
    return re.sub(r'[^\w\s]', ' ', s.lower())

def learnModel(r,m,data,stemming_stopwords,feature = 0):    
    # feature = 0: words 
    #           1: bigrams 
    #           2: trigrams  
    class_sizes = {}                           ### number of reviews of one rating, for P(c = i)
    for i in range(1,r+1):
        class_sizes[i] = 0
    class_doc_sizes = {}                      ### number of words in mega doc of each class
    for i in range(1,r+1):
        class_doc_sizes[i] = 0
    class_counters = {}                       ### word counts for each mega doc
    vocab = set()
    for i in range(1,r+1):
        class_mega_doc = []
        df = data.loc[data["overall"] == i, ["reviewText"]]
        reviews = df["reviewText"].tolist()
        reviews2 = []
        if feature == 3:
            df = data.loc[data["overall"] == i, ["summary"]]
            reviews2 = df["summary"].tolist()
        leng = len(reviews)
        for q in range(leng):
            review = reviews[q]
            if feature == 3:
                review += reviews2[q]         ### using summary
            #review = cleanPunctuation(review)
            words = nltk.word_tokenize(review)
            if stemming_stopwords:
                words =  [ps.stem(w) for w in words]
                words = [w for w in words if not w.lower() in stop_words]
            entities = words    
            if feature >= 1:
                bigrams = []
                for k in range(len(words) -1):
                    bigram = words[k] + words[k+1]
                    bigrams.append(bigram)
                entities += bigrams    
            if feature == 2: 
                trigrams = []
                for k in range(len(words) - 2):
                    trigram = words[k] + words[k+1] + words[k+2]
                    trigrams.append(trigram)
                entities += trigrams
            class_mega_doc.extend(entities)
            class_sizes[i]+=1
        class_counters[i] = Counter(class_mega_doc)
        class_doc_sizes[i] = len(class_mega_doc)
        vocab.update(class_mega_doc)
    V = len(vocab)
    vocab = list(vocab)  # [entity1, entity2,..., entityV]
    # phi = {P(c = 1), P(c = 2),..., P(c = 5) }
    phi = {}                          
    for i in range(1,6):
        phi[i] = (class_sizes[i])/m
    # theta = {1: {entity1: p11, entity2: p12, ..., entityV: p1V },...,5: {entity1: p51, entity2: p52, ..., entityV: p5V }}
    theta = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
    for entity in vocab:
        for i in range(1,6):
            count = 0
            if entity in class_counters[i]:
                count = class_counters[i][entity]
            theta[i][entity] = (count + 1) / (class_doc_sizes[i] + V)
            
    default_probs = {}
    for i in range(1,6):
        default_probs[i] = 1/ (V+ class_doc_sizes[i])        
    return (phi,theta,default_probs)

def predict(model,data,r,m,stemming_stopwords,feature = 0):
    phi,theta,default_probs = model
    reviews = data["reviewText"].tolist()
    ratings = data["overall"].tolist()
    predictions = []
    reviews2 = []
    if feature == 3:
        reviews2 = data["reviewText"].tolist()
    for q in range(m):
        review = reviews[q]
        if feature == 3:
            review += reviews2[q]         ### using summary
        #review = cleanPunctuation(review)        
        words = nltk.word_tokenize(review)
        if stemming_stopwords:
            words =  [ps.stem(w) for w in words]
            words = [w for w in words if not w.lower() in stop_words]
        entities = words
        if feature == 1:
            bigrams = []
            for k in range(len(words) -1):
                bigram = words[k] + words[k+1]
                bigrams.append(bigram)
            entities += bigrams 
        if feature == 2:
            trigrams = []
            for k in range(len(words) - 2):
                trigram = words[k] + words[k+1] + words[k+2]
                trigrams.append(trigram)
            entities += trigrams
        entities_count = Counter(entities)
        # {entity1: n1, entity2: n2,..., entityk: nk }
        pred_vals = [np.log(phi[i]) for i in range(1,6)]
        for entity in entities_count:
            for i in range(1,6):
                update = default_probs[i]
                if entity in theta[i]:
                    update = theta[i][entity]
                pred_vals[i-1] += entities_count[entity] * np.log(update)    

        prediction = 1+ pred_vals.index(max(pred_vals))
        predictions.append(prediction)
    
    accuracy = 0
    conf_mat = [[0 for i in range(r+1)] for i in range(r+1) ]
    for i in range(m):
        if(ratings[i] == predictions[i]):
            accuracy+=1
        conf_mat[ratings[i]][predictions[i]] += 1    
    accuracy/=m
    accuracy*= 100
    print(Counter(predictions))
    return accuracy,conf_mat       

def learnModel2(r,m,data,stemming_stopwords,feature = 0):    
    # feature = 0: words 
    #           1: bigrams 
    #           2: trigrams  
    class_sizes = {}                           ### number of reviews of one rating, for P(c = i)
    for i in range(1,r+1):
        class_sizes[i] = 0
    class_doc_sizes = {}                      ### number of words in mega docs (unigrams,bigrams,trigrams) of each class
    for i in range(1,r+1):
        class_doc_sizes[i] = [0,0,0]
    class_counters = {}                       ### entity counts for mega docs of each class
    vocab1 = set()                            ### vocabulary of unigrams
    vocab2 = set()                            ### vocabulary of bigrams
    vocab3 = set()                            ### vocabulary of trigrams
    for i in range(1,r+1):
        class_mega_doc = [[],[],[]]           ### words, bigrams , trigrams occuring in that class (repeated entries included)
        df = data.loc[data["overall"] == i, ["reviewText"]]
        reviews = df["reviewText"].tolist()
        for review in reviews:
            #review = cleanPunctuation(review)
            words = nltk.word_tokenize(review)
            if stemming_stopwords:
                words =  [ps.stem(w) for w in words]
                words = [w for w in words if not w.lower() in stop_words]    
            bigrams = []
            for k in range(len(words) -1):
                bigram = words[k] + words[k+1]
                bigrams.append(bigram)
            trigrams = []
            for k in range(len(words) - 2):
                trigram = words[k] + words[k+1] + words[k+2]
                trigrams.append(trigram)
            
            class_mega_doc[0].extend(words)
            class_mega_doc[1].extend(bigrams)  
            # entities = words + bigrams
            # print(entities)
            class_mega_doc[2].extend(trigrams)                      
            class_sizes[i]+=1

        class_counters[i] = [Counter(class_mega_doc[0]), Counter(class_mega_doc[1]), Counter(class_mega_doc[2])] 
        class_doc_sizes[i]= [len(class_mega_doc[0]), len(class_mega_doc[1]), len(class_mega_doc[2])]
        vocab1.update(class_mega_doc[0])
        vocab2.update(class_mega_doc[1])
        vocab3.update(class_mega_doc[2])
    V1 = len(vocab1)
    V2 = len(vocab2)
    V3 = len(vocab3)
    
    vocab1 = list(vocab1)  # [entity1, entity2,..., entityV]
    vocab2 = list(vocab2)
    vocab3 = list(vocab3)
    # phi = {P(c = 1), P(c = 2),..., P(c = 5) }
    phi = {}                          
    for i in range(1,6):
        phi[i] = (class_sizes[i])/m
    # theta = {1: {entity1: p11, entity2: p12, ..., entityV: p1V },...,5: {entity1: p51, entity2: p52, ..., entityV: p5V }}
    theta1 = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
    theta2 = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
    theta3 = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
    V = V1+V2+V3
    
    for entity in vocab1:
        for i in range(1,6):
            count = 0
            if entity in class_counters[i][0]:
                count = class_counters[i][0][entity]
            theta1[i][entity] = (count + 1) / (class_doc_sizes[i][0] + V1)
    for entity in vocab2:
        for i in range(1,6):
            count = 0
            if entity in class_counters[i][1]:
                count = class_counters[i][1][entity]
            theta2[i][entity] = (count + 1) / (class_doc_sizes[i][1] + V2)
    for entity in vocab3:
        for i in range(1,6):
            count = 0
            if entity in class_counters[i][2]:
                count = class_counters[i][2][entity]
            theta3[i][entity] = (count + 1) / (class_doc_sizes[i][2] + V3)
            
    default_probs = {}
    for i in range(1,6):
        #default_probs[i] = 1/ (V+ sum(class_doc_sizes[i]))
        default_probs[i] = {1: (1/ (V1+ class_doc_sizes[i][0])), 2: (1/ (V2+ class_doc_sizes[i][1])), 3:  (1/ (V3+ class_doc_sizes[i][2])) }     
    thetas = {1: theta1, 2: theta2, 3:theta3}
    return (phi,thetas,default_probs)

def predict2(model,data,r,m,stemming_stopwords,feature = 0):
    w = 0.01
    phi,thetas,default_probs = model
    theta1 = thetas[1]
    theta2 = thetas[2]
    theta3 = thetas[3]
    reviews = data["reviewText"].tolist()
    ratings = data["overall"].tolist()
    predictions = []
    for review in reviews:  
        #review = cleanPunctuation(review)        
        words = nltk.word_tokenize(review)
        if stemming_stopwords:
            words =  [ps.stem(w) for w in words]
            words = [w for w in words if not w.lower() in stop_words]
        bigrams = []
        for k in range(len(words) -1):
            bigram = words[k] + words[k+1]
            bigrams.append(bigram)
        trigrams = []
        for k in range(len(words) - 2):
            trigram = words[k] + words[k+1] + words[k+2]
            trigrams.append(trigram)
        words_count = Counter(words)
        bigrams_count = Counter(bigrams)
        trigrams_count = Counter(trigrams)
        # {entity1: n1, entity2: n2,..., entityk: nk }
        pred_vals = [np.log(phi[i]) for i in range(1,6)]
        for entity in words_count:
            for i in range(1,6):
                update = default_probs[i][1]
                if entity in theta1[i]:
                    update = theta1[i][entity]
                pred_vals[i-1] += words_count[entity] * np.log(update)    
        for entity in bigrams_count:
            for i in range(1,6):
                update = default_probs[i][2]
                if entity in theta2[i]:
                    update = theta2[i][entity]
                pred_vals[i-1] +=  w * bigrams_count[entity] * np.log(update)    
        for entity in trigrams_count:
            for i in range(1,6):
                if entity in theta3[i]:
                    update = theta3[i][entity]
                    pred_vals[i-1] +=  trigrams_count[entity] * np.log(update)    

        prediction = 1+ pred_vals.index(max(pred_vals))
        predictions.append(prediction)
    
    accuracy = 0
    conf_mat = [[0 for i in range(r+1)] for i in range(r+1) ]
    for i in range(m):
        if(ratings[i] == predictions[i]):
            accuracy+=1
        conf_mat[ratings[i]][predictions[i]] += 1    
    accuracy/=m
    accuracy*= 100
    print(Counter(predictions))
    return accuracy,conf_mat       


# random prediction
def random_predict(model,data,r,m,stemming_stopwords):
    phi,theta,default_probs,vocab = model
    reviews = data["reviewText"].tolist()
    ratings = data["overall"].tolist()
    predictions = []
    for review in reviews:
        predictions.append(random.randrange(1,r+1))
    accuracy = 0
    conf_mat = [[0 for i in range(r+1)] for i in range(r+1) ]
    for i in range(m):
        if(ratings[i] == predictions[i]):
            accuracy+=1
        conf_mat[ratings[i]][predictions[i]] += 1    
    accuracy/=m
    accuracy*= 100
    print(Counter(predictions))
    return accuracy,conf_mat       

# majority prediction
def majority_predict(model,data,r,m,stemming_stopwords):
    ratings = data["overall"].tolist()
    rating_count = Counter(ratings)
    prediction = 0
    count = 0
    for i in range(r):
        if rating_count[i] > count:
            prediction = i+1
            count = rating_count[i]
    accuracy = 0
    conf_mat = [[0 for i in range(r+1)] for i in range(r+1) ]
    for i in range(m):
        if(ratings[i] == prediction):
            accuracy+=1
        conf_mat[ratings[i]][prediction] += 1    
    accuracy/=m
    accuracy*= 100
    return accuracy,conf_mat       
            
# plotting the confusion matrix
def draw_confusion(confatrix,title_str):
    r = len(confatrix)
    confatrix = confatrix[1:]
    for i in range(r-1):
        confatrix[i] = confatrix[i][1:]
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 5, 1))
    ax.set_yticks(np.arange(0, 5, 1))
    ax.set_xticklabels(np.arange(1, 6, 1))
    ax.set_yticklabels(np.arange(1, 6, 1))
    plt.imshow(confatrix)
    plt.title(title_str)
    plt.colorbar()
    plt.set_cmap("Purples")
    plt.ylabel("True labels")
    plt.xlabel("Predicted label")
    plt.show()        

# calculate F1 score from confusion matrix
def F1_score(conf_mat):
    r = len(conf_mat)
    recall = [0 for i in range(r)]
    col_sum = [0 for i in range(r)]
    precision = [0 for i in range(r)]
    for i in range(r):
        for j in range(r):
            col_sum[j] += conf_mat[i][j]
    for i in range(r):
        if col_sum[i] > 0:
            precision[i] = conf_mat[i][i] / col_sum[i]  
        row_sum = sum(conf_mat[i])
        if row_sum > 0:
            recall[i] = conf_mat[i][i] / row_sum     
    f1_scores = [ (2*precision[i] * recall[i]) / (precision[i] + recall[i]) for i in range(r) if (precision[i] + recall[i]) > 0]
    return f1_scores

######### driver code #########

# main function
def main():
	# Taking parameters from command line
	
    train_data_path = sys.argv[1]
    
    test_data_path = sys.argv[2]
    
    part = sys.argv[3]
    
    r = 5
    m = 50000
    m_test = 14000
    
    start = time.time()
    stemming_stopwords = 0
    feature = 0
    raw_data = pd.read_json(train_data_path, lines=True)
    data = raw_data[["reviewText","overall"]]
    data2 = raw_data[["reviewText", "overall","summary"]]
    test_data_raw = pd.read_json(test_data_path, lines=True)
    test_data = test_data_raw[["reviewText","overall"]]
    test_data2 = test_data_raw[["reviewText", "overall","summary"]]

    if part=='a':
        model = learnModel(r,m,data,stemming_stopwords)
        st = "Simple Naive Bayes implementation"
        print(st)
        accuracy,conf_mat = predict(model, data,r,m,stemming_stopwords)
        print("Accuracy on training data set: " + str(accuracy))
        accuracy,conf_mat = predict(model, test_data,r,m_test,stemming_stopwords)
        print("Accuracy on test data set: " + str(accuracy))
        print("Confusion Matrix: ")
        print(conf_mat)
        draw_confusion(conf_mat,st)
        print("F1 score: ")
        f = F1_score(conf_mat)
        print(f)
        print("Macro score: ")
        ms = sum(f)/r
        print(ms)
        end = time.time()
        print("Total time taken: " + str(end - start))
    elif part=='b':
        model = learnModel(r,m,data,stemming_stopwords)
        st = "Random prediction Model"
        print(st)
        accuracy,conf_mat = random_predict(model, data,r,m,stemming_stopwords)
        print("Accuracy on training data set: " + str(accuracy))
        accuracy,conf_mat = random_predict(model, test_data,r,m_test,stemming_stopwords)
        print("Accuracy: " + str(accuracy))
        print("Confusion Matrix: ")
        print(conf_mat)
        draw_confusion(conf_mat,st)
        print("F1 score: ")
        f = F1_score(conf_mat)
        print(f)
        print("Macro score: ")
        ms = sum(f)/r
        print(ms)
        end = time.time()
        print("Total time taken: " + str(end - start))


        st = "Majority prediction Model"
        print(st)
        accuracy,conf_mat = majority_predict(model, data,r,m,stemming_stopwords)
        print("Accuracy on training data set: " + str(accuracy))
        accuracy,conf_mat = majority_predict(model, test_data,r,m_test,stemming_stopwords)
        print("Accuracy: " + str(accuracy))
        print("Confusion Matrix: ")
        print(conf_mat)
        draw_confusion(conf_mat,st)
        print("F1 score: ")
        f = F1_score(conf_mat)
        ms = sum(f)/r
        print(f)
        print("Macro score: ")
        print(ms)
        end = time.time()
        print("Total time taken: " + str(end - start))


    elif part=='d':
        stemming_stopwords = 1
        model = learnModel(r,m,data,stemming_stopwords)
        st = "Naive Bayes + Stemming + Stop word removal"
        print(st)
        accuracy,conf_mat = predict(model, data,r,m,stemming_stopwords)
        print("Accuracy on training data set: " + str(accuracy))
        accuracy,conf_mat = predict(model, test_data,r,m_test,stemming_stopwords)
        print("Accuracy: " + str(accuracy))
        print("Confusion Matrix: ")
        print(conf_mat)
        draw_confusion(conf_mat,st)
        print("F1 score: ")
        f = F1_score(conf_mat)
        ms = sum(f)/r
        print(f)
        print("Macro score: ")
        print(ms)
        end = time.time()
        print("Total time taken: " + str(end - start))


    elif part=='e':
        
        stemming_stopwords = 1
        st = "Naive Bayes implementation using bigrams"
        print(st)       
        model = learnModel(r,m,data,stemming_stopwords,feature = 1)
        accuracy,conf_mat = predict(model, data,r,m,stemming_stopwords, feature = 1)
        print("Accuracy on training data set: " + str(accuracy))
        accuracy,conf_mat = predict(model, test_data,r,m_test,stemming_stopwords,feature = 1)
        print("Accuracy: " + str(accuracy))
        print("Confusion Matrix: ")
        print(conf_mat)
        draw_confusion(conf_mat,st)
        print("F1 score: ")
        f = F1_score(conf_mat)
        ms = sum(f)/r
        print(f)
        print("Macro score: ")
        print(ms)
        end = time.time()
        print("Total time taken: " + str(end - start))
        
        start = time.time()
        st = "Naive Bayes using weighted bi-grams and tri-grams"
        model = learnModel2(r,m,data,stemming_stopwords,feature = 1)
        accuracy,conf_mat = predict2(model, data,r,m,stemming_stopwords, feature = 1)
        print("Accuracy on training data set: " + str(accuracy))
        accuracy,conf_mat = predict2(model, test_data,r,m_test,stemming_stopwords)
        print("Accuracy: " + str(accuracy))
        print("Confusion Matrix: ")
        print(conf_mat)
        draw_confusion(conf_mat,st)
        print("F1 score: ")
        f = F1_score(conf_mat)
        ms = sum(f)/r
        print(f)
        print("Macro score: ")
        print(ms)
        end = time.time()
        print("Total time taken: " + str(end - start))
        
    elif part == 'f':
        model = learnModel(r,m,data2,stemming_stopwords,feature=3)
        st = "Naive Bayes using review text and summary "
        print(st)
        accuracy,conf_mat = predict(model, data2,r,m,stemming_stopwords,feature=3)
        print("Accuracy on training data set: " + str(accuracy))
        accuracy,conf_mat = predict(model, test_data2,r,m_test,stemming_stopwords,feature=3)
        print("Accuracy on test data set: " + str(accuracy))
        print("Confusion Matrix: ")
        print(conf_mat)
        draw_confusion(conf_mat,st)
        print("F1 score: ")
        f = F1_score(conf_mat)
        print(f)
        print("Macro score: ")
        ms = sum(f)/r
        print(ms)
        end = time.time()
        print("Total time taken: " + str(end - start))

    else:
        print("No such part, valid parts: a,b,d,e")

if __name__ == "__main__":
	main()

