# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:36:34 2020

@author: User
"""

import pickle as p
from os.path import isfile
import tensorflow as tf


def get_embeddings_initializer(embeddings_matrix_file):
    
    assert isfile(embeddings_matrix_file), \
        'wrong file path {}'.format(embeddings_matrix_file)
        
    matrix = p.load(open(embeddings_matrix_file, 'rb'))
    
    def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
        
        assert dtype is tf.float32
        
        return matrix
    
    return my_initializer, matrix.shape
    