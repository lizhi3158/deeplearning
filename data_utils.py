# -*- coding: utf-8 -*-
def unpickle(file):      
    import pickle
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        
    return dict

def load_CIFAR10(file):   
    #get the training data  因为是bytes编码的，需要在标签前面加b，提取数据
    dataTrain = []
    labelTrain = []
    
    for i in range(1,6):
        dic = unpickle(file+"\\data_batch_"+str(i))
        for item in dic[b"data"]:    
            dataTrain.append(item)
        for item in dic[b"labels"]:
            labelTrain.append(item)

    #get test data
    dataTest = []
    labelTest = []
    dic = unpickle(file+"\\test_batch")
    
    for item in dic[b"data"]:
       dataTest.append(item)
    for item in dic[b"labels"]:
       labelTest.append(item)
       
    return dataTrain, labelTrain, dataTest, labelTest