# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append('E:/deeplearning')
from data_utils import load_CIFAR10

Xtr, Ytr, Xte, Yte = load_CIFAR10('E:/assets/cifar-10-batches-py')
Xtr = np.asarray(Xtr)
Xte = np.asarray(Xte)
Ytr = np.asarray(Ytr)
Yte = np.asarray(Yte)

class NearestNeighbor(object):
    
    
    def __init__(self):
        pass
    
    def train(self, x, y):
        self.xtr = x
        self.ytr = y

    def predict(self, x):
        num_test = x.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
          #using the L2 distance
          distances = np.sqrt(np.sum(np.square(self.xtr - x[i,:]), axis=1))
          min_index = np.argmin(distances) # get the index with smallest distance
          Ypred[i] = self.ytr[min_index]   # predict the label of the nearest example
       
        return Ypred

nn = NearestNeighbor() 
nn.train(Xtr, Ytr)            # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte) # predict labels on the test images
                              # and now print the classification accuracy, which is the average number
                              # of examples that are correctly predicted (i.e. label matches)
print ('accuracy: %f' % (np.mean(Yte_predict == Yte)))
