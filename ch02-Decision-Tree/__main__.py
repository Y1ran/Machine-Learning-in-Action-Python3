# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:23:05 2018

@author: Administrator
"""

from numpy import *
from Cal_Entropy import *
from Split_by_entropy import *
from Decision_Tree import *
from Plot_tree import *
from Classify_tree import *

def create_data():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

if __name__ == '__main__':
    myData, labels = create_data()
    print(myData)
    print(cal_entropy(myData))
    
    print(Split_Data(myData,0,1))
    print(Split_by_entropy(myData))
    
    mytree = Create_Tree(myData, labels)
    print(mytree)
    
    myTree = retrieveTree(0)
    print(Num_of_leaf(myTree), Depth_of_tree(myTree))
    myTree['no surfacing'][3] = 'maybe'
    createPlot(myTree)

    with open('lenses.txt') as fp:
        lenses = [line.strip().split('\t') for line in fp.readlines()]
        lensesLabels=['age','prescript','astigmatic','tearRate']
    
    lense_Tree = Create_Tree(lenses, lensesLabels)
    #createPlot(lense_Tree)
    print(classify(lense_Tree, lensesLabels, ['young','hyper','yes','reducedno']))
