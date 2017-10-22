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

class KNearestNeighbor(object):
    
    
    def __init__(self):
        pass
    
    def train(self, x, y):
        self.xtr = x
        self.ytr = y
        
    def predict(self, x, k=1):
        dists = self.compute_distances(x)
        
        return self.predict_labels(dists, k=k)

    def compute_distances(self, x):
        num_test = x.shape[0]
        num_train = self.xtr.shape[0]
        dists = np.zeros((num_test, num_train))

        # loop over all test rows
        for i in range(num_test):
          #using the L2 distance
          dists[i, :] = np.sqrt(np.sum(np.square(self.xtr - x[i,:]), axis=1))
         
        return dists
    
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        
        for i in range(num_test):
            closest_y = []
            kids = np.argsort(dists[i])
            closest_y = self.ytr[kids[:k]]
            count = 0
            label = 0
            
            for j in closest_y:
                tmp = 0
                for kk in closest_y:
                    tmp += (kk == j)
                    if tmp > count:
                         count = tmp
                         label = j
            y_pred[i] = label
        
        return y_pred
    
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []

for k in [1, 3, 5, 10, 20, 50, 100]:

  # use a particular value of k and evaluation on validation data
  knn = KNearestNeighbor()
  knn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = knn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print('k = %d => accuracy: %f' % (k, acc))

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    