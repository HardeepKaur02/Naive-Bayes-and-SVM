from Learn_model import learn_model

def Learn_Models(train_file_name):
    for d1 in range(10):
        for d2 in range(d1+1,10):
            learn_model(d1,d2,train_file_name)