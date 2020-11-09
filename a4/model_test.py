# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:21:03 2020

@author: User
"""


import tensorflow as tf
from model import SimpleModel
from utils import iterator_utils
from utils.misc_utils import HParams, Files
from os.path import join
import numpy as np


TRAIN_FILES = Files(src_vcb_file = join('test_data', 'vcbs', 'src_vcb.txt'),
                            trg_vcb_file = join('test_data', 'vcbs', 'trg_vcb.txt'),
                            src_train = join('test_data', 'train', 'src.txt'),
                            trg_train = join('test_data', 'train', 'trg.txt'),
                            src_dev = join('test_data', 'dev', 'src.txt'),
                            trg_dev = join('test_data', 'dev', 'trg.txt'),
                            src_test = join('test_data', 'test', 'src.txt'),
                            trg_test = join('test_data', 'test', 'trg.txt'))

TRAIN_HPARAMS = HParams('TRAIN',
                          filesobj = TRAIN_FILES,
                          buffer_size = None,
                          num_epochs = 1,
                          batch_size = 2,
                          model_type = 'simple_model',
                          logdir = './logs/train_logs',
                          src_embeddings_matrix_file = join('test_data', 'pretrained_embeddings',
                                                            'src_matrix.p'),
                          trg_embeddings_matrix_file = join('test_data', 'pretrained_embeddings',
                                                            'trg_matrix.p'),
                          num_units = 32,
                          learning_rate = 3e-04,
                          translation_file_path = None,
                          num_steps_to_eval = None, 
                          chkpts_dir = './chkpts'
                          )



class ModelTest(tf.test.TestCase):
    
    def setUp(self):
        
        super(ModelTest, self).setUp()
        
        self.graph = tf.Graph()
        
        self.sess = tf.Session(graph = self.graph)
        
        with self.graph.as_default():
            
            self.iterator, _ = iterator_utils.get_iterator('TRAIN',
                                                   filesobj = TRAIN_FILES,
                                                   buffer_size = TRAIN_HPARAMS.buffer_size,
                                                   num_epochs = TRAIN_HPARAMS.num_epochs,
                                                   batch_size = TRAIN_HPARAMS.batch_size,
                                                   debug_mode = True)
            
            self.model = SimpleModel(TRAIN_HPARAMS, 
                                self.iterator, 
                                'TRAIN')
            
            self.table_init_op = tf.tables_initializer()
            
            self.vars_init_op = tf.global_variables_initializer()
            
    
    def testGraphVariables(self):
        
        expected_variables = {'decoder/decoder_outter_scope/dense/kernel:0': (2 * self.model._num_units,
                                                                  13),
                              
                              'encoder/bidirectional_rnn/fw/lstm_cell/kernel:0': (3 + self.model._num_units,
                                                                          4 * self.model._num_units),
                              
                              'encoder/bidirectional_rnn/fw/lstm_cell/bias:0': (4 * self.model._num_units, ),
                              
                              'encoder/bidirectional_rnn/bw/lstm_cell/kernel:0': (3 + self.model._num_units,
                                                                          4 * self.model._num_units),
                              
                              'encoder/bidirectional_rnn/bw/lstm_cell/bias:0': (4 * self.model._num_units, ),
                              
                              'decoder/decoder_outter_scope/basic_lstm_cell/kernel:0': (3 + 2 * self.model._num_units,
                                                                 4 * 2 * self.model._num_units),
                              
                              'decoder/decoder_outter_scope/basic_lstm_cell/bias:0': (4 * 2 * self.model._num_units, )
                              }
        
        with self.graph.as_default():
            
            var_names = [var.name for var in tf.trainable_variables()]
            
        self.assertAllEqual(sorted(var_names), sorted(list(expected_variables.keys())),
                            'variables are not compatible')
        
        with self.graph.as_default():
            
            for var in tf.trainable_variables():
                
                self.assertAllEqual(tuple(var.get_shape().as_list()),
                                    expected_variables[var.name],
                                    'missed shapes at {}'.format(var.name))
    
        
    
    def testEmbeddingsMatrixCorrectness(self):
                        
        self.sess.run(self.vars_init_op)
        
        
        src_matrix = self.model.source_embeddings_matrix.eval(session = self.sess)
        
        trg_matrix = self.model.target_embeddings_matrix.eval(session = self.sess)
        
        
        expected_src = np.ones(shape = (12, 3))
        
        expected_trg = np.ones(shape = (13, 3))
        
        
        self.assertAllEqual(expected_src, src_matrix, 
                            'src matrix is wrong')
        
        self.assertAllEqual(expected_trg, trg_matrix, 
                            'trg matrix is wrong')
        
        
        
    def testLossMaskFunction(self):
        
        res1 = np.array([[1, 1, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1]])
        
        res2 = np.array([[1, 1],
                         [1, 1],
                         [1, 1],
                         [1, 1],
                         [1, 1]])
        
        res3 = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        
        
        with self.graph.as_default():
        
            with tf.name_scope('first_try'):
            
                try1 = self.model._return_loss_mask(tf.constant((2, 3, 7)),
                                                    tf.constant(7))
                
            with tf.name_scope('second_try'):
                
                try2 = self.model._return_loss_mask(tf.constant((2, 2, 2, 2, 2)),
                                                    tf.constant(2))
                
            with tf.name_scope('third_try'):
                
                try3 = self.model._return_loss_mask(tf.constant((3, 2, 1, 4, 6, 4, 3, 10)),
                                                    tf.constant(10))
                
        exp1 = self.sess.run(try1)
        
        exp2 = self.sess.run(try2)
        
        exp3 = self.sess.run(try3)
        
        
        self.assertAllEqual(res1, exp1,
                            'first case is down')
        
        self.assertAllEqual(res2, exp2,
                            'second case is down')
        
        self.assertAllEqual(res3, exp3,
                            'third case is down')
        
        
    def testModelInitialLoss(self):
        
        loss_tensor = self.graph.get_tensor_by_name('loss/add:0')
        
        self.sess.run(self.table_init_op)
        
        self.sess.run(self.iterator.initializer)
        
        self.sess.run(self.vars_init_op)
        
        real_loss = self.sess.run(loss_tensor)
        
        ### Based on suggestion that on the very first step model 
        ### does not know anything hence it should output equal
        ### probabilities for all the words in the target vcb
        
        expected_loss = -6 * np.log(1 / 13) #15.389696144769221
        
        self.assertAllClose(real_loss, expected_loss, 
                            atol = 0.3,
                            msg = '{} vs {}'.format(real_loss, expected_loss))
        
        
if __name__ == '__main__':
    
    tf.test.main()
        
        
    
    