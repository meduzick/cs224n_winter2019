# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:20:04 2020

@author: User
"""


import tensorflow as tf
import numpy as np
from os.path import join
from model import AttentionHybridModel
from model_helper import BuildTrainModel, BuildEvalModel, BuildInferModel
from utils.misc_utils import HParams, Files
from utils.iterator_utils import get_iterator

files = Files(src_char_vcb_file = join('sanity_checks_data', 'vcbs', 
                                                    'src_char_vcb.txt'),
            trg_char_vcb_file = join('sanity_checks_data', 'vcbs', 
                                     'trg_char_vcb.txt'),
             trg_vcb_file = join('sanity_checks_data', 'vcbs',
                                 'trg_vcb.txt'),
             src_train = join('sanity_checks_data', 'train', 'src.txt'),
             trg_train = join('sanity_checks_data', 'train', 'trg.txt'),
             src_dev = join('sanity_checks_data', 'dev', 'src.txt'),
             trg_dev = join('sanity_checks_data', 'dev', 'trg.txt'),
             src_test = join('sanity_checks_data', 'dev', 'src.txt'),
             trg_test = join('sanity_checks_data', 'dev', 'trg.txt'))

hparams = HParams(regime = 'TRAIN',
                  filesobj = files,
                  buffer_size = None,
                  num_epochs = 1, 
                  batch_size = 2,
                  model_type = 'AttentionHybrid',
                  logdir = None,
                  trg_embeddings_matrix_file = './sanity_checks_data/vcbs/trg_embeddings.p',
                  num_units = 32, 
                  learning_rate = 3e-04, 
                  translation_file_path = None,
                  char_translation_file_path = None,
                  num_steps_to_eval = None, 
                  chkpts_dir = None,
                  trg_char_vcb_file = './sanity_checks_data/vcbs/trg_char_vcb.txt',
                  src_char_vcb_file = './sanity_checks_data/vcbs/src_char_vcb.txt',
                  kernel_size = 2, 
                  word_emb_dim = 5,
                  stride = 1,
                  char_emb_dim = 3)


expected_variables = {
            'embeddings_matricies/trg_char_embeddings:0': (63, 3),
            'embeddings_matricies/src_char_embeddings:0': (68, 3),
            
            'encoder/cnn_encoder/cnn_filters:0': (2, 3, 5),
            
            'encoder/highway_layer/transform_weights:0': (5, 5),
            'encoder/highway_layer/transition_weights:0': (5, 5),
            'encoder/highway_layer/transform_bias:0': (5,),
            'encoder/highway_layer/transition_bias:0': (5,),
            
            'encoder/bidirectional_rnn/fw/lstm_cell/kernel:0': (32 + 5, 32 * 4),
            'encoder/bidirectional_rnn/fw/lstm_cell/bias:0': (32 * 4, ),
            'encoder/bidirectional_rnn/bw/lstm_cell/kernel:0': (32 + 5, 32 * 4),
            'encoder/bidirectional_rnn/bw/lstm_cell/bias:0': (32 * 4, ),
            
            'decoder/word_level_decoder/decoder_outter_scope/attention_wrapper/basic_lstm_cell/kernel:0': \
                                                        (2 * 32 + 3 + 2 * 32,
                                                         2 * 32 * 4),
            'decoder/word_level_decoder/decoder_outter_scope/attention_wrapper/basic_lstm_cell/bias:0':\
                                                            (2 * 32 * 4, ),
            'decoder/word_level_decoder/decoder_outter_scope/attention_wrapper/dense/kernel:0':\
                                                                (2 * 32 + 2 * 32,
                                                                 2 * 32),
            'decoder/word_level_decoder/memory_layer/kernel:0': (2 * 32, 2 * 32),
            'decoder/word_level_decoder/decoder_outter_scope/dense/kernel:0': (2 * 32, 59465),
            
            
            'decoder/word_level_decoder/dense/kernel:0': (59465, 2 * 32),
            
            
            'decoder/char_level_decoder/char_decoder_outter_scope/lstm_cell/kernel:0':\
                                                                    (2 * 32 + 3,
                                                                     2 * 32 * 4),
            'decoder/char_level_decoder/char_decoder_outter_scope/lstm_cell/bias:0': (2 * 32 * 4, ),
            'decoder/char_level_decoder/char_decoder_outter_scope/dense/kernel:0': (2 * 32, 63)
            }
    

class ModelTest(tf.test.TestCase):
    
    def setUp(self):
        
        super(ModelTest, self).setUp()
        
        self.graph = tf.Graph()
        
        self.sess = tf.Session(graph = self.graph)
        
    
    def testLossFunction(self):
        
        word_logits = np.array(
                                [
                                    [
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 1],
                                        [0, 0, 1, 0]
                                    ],
                                    [
                                        [0, 1, 0, 0],
                                        [1, 0, 0, 0],
                                        [1, 0, 0, 1]
                                    ]
                                ]
                            )

        trg_labels = np.array(
                                [
                                    [0, 1, 2],
                                    [3, 0, 1]
                                ]
                            )
        
        char_logits = np.array(
                                [
                                    [
                                        [0, 0, 0],
                                        [0, 0, 1]
                                    ],
                                    [
                                        [0, 1, 0],
                                        [1, 0, 0]
                                    ],
                                    [
                                        [1, 0, 1],
                                        [1, 1, 0]
                                    ],
                                    [
                                        [1, 1, 1],
                                        [0, 0, 0]
                                    ],
                                ]
                            )
        
        trg_chars_out = np.array(
                                [
                                    [
                                        [0, 1],
                                        [0, 2],
                                        [1, 0]
                                    ],
                                    [
                                        [1, 2],
                                        [2, 1],
                                        [2, 0]
                                    ]
                                ]
                            )
                            
        trg_size = np.array([3, 1])
        
        trg_char_lens = np.array(
                                [
                                    [1, 2, 1],
                                    [2, 0, 0]
                                ]
                            )
        
        expected_loss = 11.069370849429646
        
        
        with self.graph.as_default():
            
            word_logits = tf.constant(word_logits, dtype = tf.float32)

            trg_labels = tf.constant(trg_labels, dtype = tf.int32)
            
            char_logits = tf.constant(char_logits, dtype = tf.float32)
            
            trg_chars_out = tf.constant(trg_chars_out, dtype = tf.int32)
            
            trg_size = tf.constant(trg_size, dtype = tf.int32)
            
            trg_char_lens = tf.constant(trg_char_lens, dtype = tf.int32)
            
            res = AttentionHybridModel._calculate_loss(word_logits, 
                                                        trg_labels, 
                                                        char_logits,
                                                        trg_chars_out, 
                                                        trg_size,
                                                        trg_char_lens)
            
        loss = self.sess.run(res)
        
        self.assertAllClose(loss, expected_loss, 
                            msg = 'wrong loss')
        
        
    def testGraphVariables(self):
        
        with self.graph.as_default():
            
            iterator, total_num = get_iterator('TRAIN',
                                                filesobj = files,
                                                buffer_size = hparams.buffer_size,
                                                num_epochs = hparams.num_epochs,
                                                batch_size = hparams.batch_size, 
                                                debug_mode = True)
            
            _ = BuildTrainModel(hparams,
                                iterator,
                                tf.get_default_graph())
            
            var_names = [var.name for var in tf.trainable_variables()]
            
            
        self.assertAllEqual(sorted(var_names), sorted(list(expected_variables.keys())),
                            'variables are not compatible')
            
        with self.graph.as_default():
            
            for var in tf.trainable_variables():
                
                self.assertAllEqual(tuple(var.get_shape().as_list()),
                                    expected_variables[var.name],
                                    'missed shapes at {}'.format(var.name))
                
    def testGraphVariablesInferMode(self):
        
        with self.graph.as_default():
            
            iterator, total_num = get_iterator('TEST',
                                                filesobj = files,
                                                buffer_size = hparams.buffer_size,
                                                num_epochs = hparams.num_epochs,
                                                batch_size = hparams.batch_size, 
                                                debug_mode = True)
            
            _ = BuildInferModel(hparams,
                                iterator,
                                tf.get_default_graph(),
                                None,
                                None)
            
            var_names = [var.name for var in tf.trainable_variables()]
            
            
        self.assertAllEqual(sorted(var_names), sorted(list(expected_variables.keys())),
                            'variables are not compatible infer mode')
            
        with self.graph.as_default():
            
            for var in tf.trainable_variables():
                
                self.assertAllEqual(tuple(var.get_shape().as_list()),
                                    expected_variables[var.name],
                                    'missed shapes at {} infer mode'.format(var.name))
                
    
    def testGraphVariablesDevMode(self):
        
        with self.graph.as_default():
            
            iterator, total_num = get_iterator('DEV',
                                                filesobj = files,
                                                buffer_size = hparams.buffer_size,
                                                num_epochs = hparams.num_epochs,
                                                batch_size = hparams.batch_size, 
                                                debug_mode = True)
            
            _ = BuildEvalModel(hparams,
                               iterator,
                               tf.get_default_graph())
            
            var_names = [var.name for var in tf.trainable_variables()]
            
            
        self.assertAllEqual(sorted(var_names), sorted(list(expected_variables.keys())),
                            'variables are not compatible dev mode')
            
        with self.graph.as_default():
            
            for var in tf.trainable_variables():
                
                self.assertAllEqual(tuple(var.get_shape().as_list()),
                                    expected_variables[var.name],
                                    'missed shapes at {} dev mode'.format(var.name))
        
        
        
if __name__ == '__main__':
    
    tf.test.main()