# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:24:59 2020

@author: User
"""


import tensorflow as tf
from os.path import join
import numpy as np
from utils import iterator_utils
from utils.misc_utils import Files
from utils.iterator_utils import BatchedInput


class InputTest(tf.test.TestCase):
    
    def setUp(self):
        
        super(InputTest, self).setUp()
        
        self.graph = tf.Graph()
        
        self.sess = tf.Session(graph = self.graph)
        
        self.files = Files(src_char_vcb_file = join('sanity_checks_data', 'vcbs', 
                                                    'src_char_vcb.txt'),
                           trg_char_vcb_file = join('sanity_checks_data', 'vcbs', 
                                                    'trg_char_vcb.txt'),
                            trg_vcb_file = join('sanity_checks_data', 'vcbs',
                                                'trg_vcb.txt'),
                            src_train = join('sanity_checks_data', 'train', 'src.txt'),
                            trg_train = join('sanity_checks_data', 'train', 'trg.txt'),
                            src_dev = join('sanity_checks_data', 'dev', 'src.txt'),
                            trg_dev = join('sanity_checks_data', 'dev', 'trg.txt'),
                            src_test = None,
                            trg_test = None)
        
        
    def _check_all_equal(self,
                         expected,
                         gt,
                         num_batch):
        
        self.assertAllEqual(gt.src_chars, expected[0], 
                    'mismatch in source sequence chars {} batch'.format(num_batch))
        
        self.assertAllEqual(gt.trg_chars_in, expected[1],
                    'mismatch in target in sequence chars {} batch'.format(num_batch))
        
        self.assertAllEqual(gt.trg_chars_out, expected[2],
                    'mismatch in target out sequence chars {} batch'.format(num_batch))
        
        self.assertAllEqual(gt.trg_chars_lens, expected[3],
                    'mismatch in target chars sequence sizes {} batch'.format(num_batch))
        
        self.assertAllEqual(gt.trg_words_in, expected[4],
                    'mismatch in target in sequence words {} batch'.format(num_batch))
        
        self.assertAllEqual(gt.trg_words_out, expected[5],
                            'mismatch in target out sequence words {} batch'.format(num_batch))
        
        self.assertAllEqual(gt.src_size, expected[6],
                            'mismatch in source sequence sizes {} batch'.format(num_batch))
        
        self.assertAllEqual(gt.trg_size, expected[7],
                            'mismatch in target sequence sizes {} batch'.format(num_batch))
        
        self.assertAllEqual(gt.sos_token_id, expected[8],
                            'mismatch in sos_token'.format(num_batch))
        
        self.assertAllEqual(gt.eos_token_id, expected[9],
                            'mismatch in eos_token'.format(num_batch))
        
        
        
    def testTrainInput(self):
        
        try1 = BatchedInput(src_chars = np.array([[
                                      [41, 53, 52, 42, 59, 41, 47, 43, 52, 42, 53],
                                      [52, 53, 57, 53, 58, 56, 53, 57, 2, 2, 2],
                                      [51, 47, 57, 51, 53, 57, 2, 2, 2, 2, 2],
                                      [15, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
                                     
                                     [[52, 53, 57, 58, 56, 53, 57, 2, 2, 2, 2],
                                      [57, 53, 51, 53, 57, 2, 2, 2, 2, 2, 2],
                                      [59, 52, 53, 2, 2, 2, 2, 2, 2, 2, 2],
                                      [15, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                                     ]]),
                    trg_chars_in = np.array([
                                    [
                                        [1, 40, 54, 45, 58, 45, 50, 43, 2, 2],
                                        [1, 51, 57, 54, 55, 41, 48, 58, 41, 55],
                                        [1, 15, 2, 2, 2, 2, 2, 2, 2, 2],
                                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                                    ],
                                    [
                                        [1, 59, 41, 2, 2, 2, 2, 2, 2, 2],
                                        [1, 37, 54, 41, 2, 2, 2, 2, 2, 2],
                                        [1, 51, 50, 41, 2, 2, 2, 2, 2, 2],
                                        [1, 15, 2, 2, 2, 2, 2, 2, 2, 2]
                                    ]
                                ]),
                    trg_chars_out = np.array([
                                    [
                                        [40, 54, 45, 58, 45, 50, 43, 2, 2, 2],
                                        [51, 57, 54, 55, 41, 48, 58, 41, 55, 2],
                                        [15, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                                    ],
                                    [
                                        [59, 41, 2, 2, 2, 2, 2, 2, 2, 2],
                                        [37, 54, 41, 2, 2, 2, 2, 2, 2, 2],
                                        [51, 50, 41, 2, 2, 2, 2, 2, 2, 2],
                                        [15, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                                    ]
                                ]),
                    trg_chars_lens =  np.array(
                                    [
                                        [8, 10, 2, 0],
                                        [3, 4, 4, 2]
                                    ]
                                ),
                    trg_words_in = np.array(
                                    [
                                        [1, 100, 110, 9, 2],
                                        [1, 98, 348, 81, 9]
                                    ]
                                ),
                    trg_words_out = np.array(
                                    [
                                        [100, 110, 9, 2, 2],
                                        [98, 348, 81, 9, 2]
                                    ]
                                ),
                    src_size = np.array(
                                    [4, 4]
                        ),
                    trg_size = np.array(
                                    [4, 5]
                        ),
                    initializer = None,
                    sos_token_id = 1,
                    eos_token_id = 2
                            )
        
        try2 = BatchedInput(src_chars = np.array(
                                    [
                                        [
                                            [46, 59, 40, 53, 2, 2],
                                            [51, 59, 41, 46, 53, 2],
                                            [42, 47, 50, 53, 45, 53],
                                            [15, 2, 2, 2, 2, 2]
                                        ]
                                    ]
                            ),
                            trg_chars_in = np.array(
                                    [
                                        [
                                            [1, 56, 44, 41, 54, 41, 2],
                                            [1, 59, 37, 55, 2, 2, 2],
                                            [1, 37, 2, 2, 2, 2, 2],
                                            [1, 59, 44, 51, 48, 41, 2],
                                            [1, 48, 51, 56, 2, 2, 2],
                                            [1, 51, 42, 2, 2, 2, 2],
                                            [1, 40, 45, 37, 48, 51, 43],
                                            [1, 15, 2, 2, 2, 2, 2]
                                        ]
                                    ]
                                ),
                            trg_chars_out = np.array(
                                    [
                                        [
                                            [56, 44, 41, 54, 41, 2, 2],
                                            [59, 37, 55, 2, 2, 2, 2],
                                            [37, 2, 2, 2, 2, 2, 2],
                                            [59, 44, 51, 48, 41, 2, 2],
                                            [48, 51, 56, 2, 2, 2, 2],
                                            [51, 42, 2, 2, 2, 2, 2],
                                            [40, 45, 37, 48, 51, 43, 2],
                                            [15, 2, 2, 2, 2, 2, 2]
                                        ]
                                    ]
                                ),
                            trg_chars_lens = np.array(
                                    [
                                        [6, 4, 2, 6, 4, 3, 7, 2]
                                    ]
                                ),
                            trg_words_in = np.array(
                                    [
                                        [1, 121, 122, 14, 836, 416, 37, 1471, 9]
                                    ]
                                ),
                            trg_words_out = np.array(
                                [
                                    [121, 122, 14, 836, 416, 37, 1471, 9, 2]
                                ]
                                ),
                            src_size = np.array([4]),
                            trg_size = np.array([9]),
                            initializer = None,
                            sos_token_id = 1,
                            eos_token_id = 2
                            )
        
        with self.graph.as_default():
        
            iterator, total_num = iterator_utils.get_iterator('TRAIN',
                                                   filesobj = self.files,
                                                   buffer_size = None,
                                                   num_epochs = 1,
                                                   batch_size = 2, 
                                                   debug_mode = True)
            
            table_init_op = tf.tables_initializer()
            
        
        self.sess.run(table_init_op)
        
        self.sess.run(iterator.initializer)
        
        
        res1 = self.sess.run([iterator.src_chars,
                              iterator.trg_chars_in,
                              iterator.trg_chars_out,
                              iterator.trg_chars_lens,
                              iterator.trg_words_in,
                              iterator.trg_words_out,
                              iterator.src_size,
                              iterator.trg_size,
                              iterator.sos_token_id,
                              iterator.eos_token_id])
        
        self._check_all_equal(res1, try1, 1)
        
        
        res2 = self.sess.run([iterator.src_chars,
                              iterator.trg_chars_in,
                              iterator.trg_chars_out,
                              iterator.trg_chars_lens,
                              iterator.trg_words_in,
                              iterator.trg_words_out,
                              iterator.src_size,
                              iterator.trg_size,
                              iterator.sos_token_id,
                              iterator.eos_token_id])
        
        
        self._check_all_equal(res2, try2, 2)
        
        
        self.assertAllEqual(total_num, 3,
                            'mismatch in total num somehow')
        
        
    def testDevInput(self):
        
        try1 = BatchedInput(src_chars = np.array([[
                                      [41, 53, 52, 42, 59, 41, 47, 43, 52, 42, 53],
                                      [52, 53, 57, 53, 58, 56, 53, 57, 2, 2, 2],
                                      [51, 47, 57, 51, 53, 57, 2, 2, 2, 2, 2],
                                      [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
                                     
                                     [[52, 53, 57, 58, 56, 53, 57, 2, 2, 2, 2],
                                      [57, 53, 51, 53, 57, 2, 2, 2, 2, 2, 2],
                                      [59, 52, 53, 2, 2, 2, 2, 2, 2, 2, 2],
                                      [15, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                                     ]]),
                    trg_chars_in = np.array([
                                    [
                                        [1, 40, 54, 45, 58, 45, 50, 43, 2, 2],
                                        [1, 51, 57, 54, 55, 41, 48, 58, 41, 55],
                                        [1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
                                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                                    ],
                                    [
                                        [1, 59, 41, 2, 2, 2, 2, 2, 2, 2],
                                        [1, 37, 54, 41, 2, 2, 2, 2, 2, 2],
                                        [1, 51, 50, 41, 2, 2, 2, 2, 2, 2],
                                        [1, 15, 2, 2, 2, 2, 2, 2, 2, 2]
                                    ]
                                ]),
                    trg_chars_out = np.array([
                                    [
                                        [40, 54, 45, 58, 45, 50, 43, 2, 2, 2],
                                        [51, 57, 54, 55, 41, 48, 58, 41, 55, 2],
                                        [0, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                                    ],
                                    [
                                        [59, 41, 2, 2, 2, 2, 2, 2, 2, 2],
                                        [37, 54, 41, 2, 2, 2, 2, 2, 2, 2],
                                        [51, 50, 41, 2, 2, 2, 2, 2, 2, 2],
                                        [15, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                                    ]
                                ]),
                    trg_chars_lens =  np.array(
                                    [
                                        [8, 10, 2, 0],
                                        [3, 4, 4, 2]
                                    ]
                                ),
                    trg_words_in = np.array(
                                    [
                                        [1, 100, 110, 0, 2],
                                        [1, 98, 348, 81, 9]
                                    ]
                                ),
                    trg_words_out = np.array(
                                    [
                                        [100, 110, 0, 2, 2],
                                        [98, 348, 81, 9, 2]
                                    ]
                                ),
                    src_size = np.array(
                                    [4, 4]
                        ),
                    trg_size = np.array(
                                    [4, 5]
                        ),
                    initializer = None,
                    sos_token_id = 1,
                    eos_token_id = 2
                            )
        
        try2 = BatchedInput(src_chars = np.array(
                                    [
                                        [
                                            [46, 59, 40, 53, 2, 2],
                                            [51, 59, 41, 46, 53, 2],
                                            [42, 47, 50, 53, 45, 53],
                                            [15, 2, 2, 2, 2, 2]
                                        ]
                                    ]
                            ),
                            trg_chars_in = np.array(
                                    [
                                        [
                                            [1, 56, 44, 41, 54, 41, 2],
                                            [1, 59, 37, 55, 2, 2, 2],
                                            [1, 37, 2, 2, 2, 2, 2],
                                            [1, 59, 44, 51, 48, 41, 2],
                                            [1, 48, 51, 56, 2, 2, 2],
                                            [1, 51, 42, 2, 2, 2, 2],
                                            [1, 40, 45, 37, 48, 51, 43],
                                            [1, 15, 2, 2, 2, 2, 2]
                                        ]
                                    ]
                                ),
                            trg_chars_out = np.array(
                                    [
                                        [
                                            [56, 44, 41, 54, 41, 2, 2],
                                            [59, 37, 55, 2, 2, 2, 2],
                                            [37, 2, 2, 2, 2, 2, 2],
                                            [59, 44, 51, 48, 41, 2, 2],
                                            [48, 51, 56, 2, 2, 2, 2],
                                            [51, 42, 2, 2, 2, 2, 2],
                                            [40, 45, 37, 48, 51, 43, 2],
                                            [15, 2, 2, 2, 2, 2, 2]
                                        ]
                                    ]
                                ),
                            trg_chars_lens = np.array(
                                    [
                                        [6, 4, 2, 6, 4, 3, 7, 2]
                                    ]
                                ),
                            trg_words_in = np.array(
                                    [
                                        [1, 121, 122, 14, 836, 416, 37, 1471, 9]
                                    ]
                                ),
                            trg_words_out = np.array(
                                [
                                    [121, 122, 14, 836, 416, 37, 1471, 9, 2]
                                ]
                                ),
                            src_size = np.array([4]),
                            trg_size = np.array([9]),
                            initializer = None,
                            sos_token_id = 1,
                            eos_token_id = 2
                            )
        
        with self.graph.as_default():
        
            iterator, total_num = iterator_utils.get_iterator('DEV',
                                                   filesobj = self.files,
                                                   buffer_size = None,
                                                   num_epochs = 1,
                                                   batch_size = 2, 
                                                   debug_mode = True)
            
            table_init_op = tf.tables_initializer()
            
        
        self.sess.run(table_init_op)
        
        self.sess.run(iterator.initializer)
        
        
        res1 = self.sess.run([iterator.src_chars,
                              iterator.trg_chars_in,
                              iterator.trg_chars_out,
                              iterator.trg_chars_lens,
                              iterator.trg_words_in,
                              iterator.trg_words_out,
                              iterator.src_size,
                              iterator.trg_size,
                              iterator.sos_token_id,
                              iterator.eos_token_id])
        
        self._check_all_equal(res1, try1, 1)
        
        
        res2 = self.sess.run([iterator.src_chars,
                              iterator.trg_chars_in,
                              iterator.trg_chars_out,
                              iterator.trg_chars_lens,
                              iterator.trg_words_in,
                              iterator.trg_words_out,
                              iterator.src_size,
                              iterator.trg_size,
                              iterator.sos_token_id,
                              iterator.eos_token_id])
        
        
        self._check_all_equal(res2, try2, 2)
        
        
        self.assertAllEqual(total_num, 3,
                            'mismatch in total num somehow')
        
    
   
        
if __name__ == '__main__':
    
    tf.test.main()
            
        