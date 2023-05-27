#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 18:59:12 2018

@author: miguelrocha
"""

import numpy as np

class Dataset:
    
    # constructor
    def __init__(self, filename = None, X = None, Y = None):
        if filename is not None:
            self.readDataset(filename)
        elif X is not None and Y is not None:
            self.X = X
            self.Y = Y
        else:
            self.X = None
            self.Y = None
        
    def readDataset(self, filename, sep = ","):
        data = np.genfromtxt(filename, delimiter=sep)
        self.X = data[:,0:-1]
        self.Y = data[:,-1]
        
    def writeDataset(self, filename, sep = ","):
        fullds = np.hstack( (self.X, self.Y.reshape(len(self.Y),1)))
        np.savetxt(filename, fullds, delimiter = sep)
        
    def getXy (self):
        return self.X, self.Y
    
    def train_test_split(self, p = 0.7):
        from random import shuffle
        ninst = self.X.shape[0]
        inst_indexes = np.array(range(ninst))
        ntr = (int)(p*ninst)
        shuffle(inst_indexes)
        tr_indexes = inst_indexes[1:ntr]
        tst_indexes = inst_indexes[ntr+1:]
        Xtr = self.X[tr_indexes,]
        ytr = self.Y[tr_indexes]
        Xts = self.X[tst_indexes,]
        yts = self.Y[tst_indexes]
        return (Xtr, ytr, Xts, yts) 
    
    def process_binary_y(self):
        y_values = np.unique(self.Y)
        if len(y_values) == 2:
            self.Y = np.where(self.Y == y_values[0], 0, 1)
        else:
            print("Non binary")

def test():
    d = Dataset("lr-example1.data")
    print(d.getXy())
    
    Xtr, ytr, Xts, yts = d.train_test_split()
    print(Xts.shape)
    print(yts.shape)

def convert_bin():
    d = Dataset("datasets/hearts.data")
    d.process_binary_y()
    d.writeDataset("datasets/hearts-bin.data")
    
    
#test()
#convert_bin()
        