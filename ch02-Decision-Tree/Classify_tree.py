# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 21:44:42 2018

@author: Administrator
"""

def classify(inp_tree, labels, test_vec):
    first_node = list(inp_tree.keys())[0]
    second_dict = inp_tree[first_node]
    index = labels.index(first_node)
    
    for key in second_dict.keys():
        if test_vec[index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], labels, test_vec)
            else:   class_label = second_dict[key]
    return class_label

def store_tree(inp_tree, filename):
    import pickle
    with open(filename,'w') as fp:
        pickle.dump(inp_tree, fp)
    
def grab_tree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
