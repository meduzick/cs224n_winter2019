# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 20:38:45 2020

@author: User
"""


import tensorflow as tf
from model import AttentionModel
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
                          model_type = 'attention_model',
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
        
        self.session = tf.Session(graph = self.graph)
        
        with self.graph.as_default():
            
            self.iterator, _ = iterator_utils.get_iterator('TRAIN',
                                                   filesobj = TRAIN_FILES,
                                                   buffer_size = TRAIN_HPARAMS.buffer_size,
                                                   num_epochs = TRAIN_HPARAMS.num_epochs,
                                                   batch_size = TRAIN_HPARAMS.batch_size,
                                                   debug_mode = True)
            
            self.model = AttentionModel(TRAIN_HPARAMS, 
                                self.iterator, 
                                'TRAIN')
            
            self.table_init_op = tf.tables_initializer()
            
            self.vars_init_op = tf.global_variables_initializer()
            
    
    def testGraphVariables(self):
                
        expected_variables = {
                              
            'encoder/bidirectional_rnn/fw/lstm_cell/kernel:0': (3 + self.model._num_units,
                                                        4 * self.model._num_units),
            
            'encoder/bidirectional_rnn/fw/lstm_cell/bias:0': (4 * self.model._num_units, ),
            
            'encoder/bidirectional_rnn/bw/lstm_cell/kernel:0': (3 + self.model._num_units,
                                                        4 * self.model._num_units),
            
            'encoder/bidirectional_rnn/bw/lstm_cell/bias:0': (4 * self.model._num_units, ),
            
            
            
            'decoder/decoder_outter_scope/attention_wrapper/basic_lstm_cell/kernel:0': \
                (3 + 2 * self.model._num_units + 2 * self.model._num_units,
                4 * 2 * self.model._num_units),
            
            'decoder/decoder_outter_scope/attention_wrapper/basic_lstm_cell/bias:0': (4 * 2 * self.model._num_units, ),
            
            'decoder/decoder_outter_scope/attention_wrapper/attention_layer/kernel:0':\
                (2 * self.model._num_units + 2 * self.model._num_units, 2 * self.model._num_units),
            
            'decoder/memory_layer/kernel:0': (2 * self.model._num_units, 
                                                                   2 * self.model._num_units),
            
            'decoder/decoder_outter_scope/dense/kernel:0': (2 * self.model._num_units,
                                                                  13)
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
        
        
    def testModelInitialLoss(self):
        
        loss_tensor = self.graph.get_tensor_by_name('loss/add:0')
        
        self.session.run(self.table_init_op)
        
        self.session.run(self.iterator.initializer)
        
        self.session.run(self.vars_init_op)
        
        real_loss = self.session.run(loss_tensor)
        
        ### Based on suggestion that on the very first step model 
        ### does not know anything hence it should output equal
        ### probabilities for all the words in the target vcb
        
        expected_loss = -6 * np.log(1 / 13) #15.389696144769221
        
        self.assertAllClose(real_loss, expected_loss, 
                            atol = 0.3,
                            msg = '{} vs {}'.format(real_loss, expected_loss))
                


if __name__ == '__main__':
    
    tf.test.main()