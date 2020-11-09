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
        
        self.files = Files(src_vcb_file = join('test_data', 'vcbs', 'src_vcb.txt'),
                            trg_vcb_file = join('test_data', 'vcbs', 'trg_vcb.txt'),
                            src_train = join('test_data', 'train', 'src.txt'),
                            trg_train = join('test_data', 'train', 'trg.txt'),
                            src_dev = join('test_data', 'dev', 'src.txt'),
                            trg_dev = join('test_data', 'dev', 'trg.txt'),
                            src_test = join('test_data', 'test', 'src.txt'),
                            trg_test = join('test_data', 'test', 'trg.txt'))
        
        
    def _check_all_equal(self,
                         expected,
                         gt,
                         num_batch):
        
        self.assertAllEqual(gt.source, expected[0], 
                    'mismatch in source sequence {} batch'.format(num_batch))
        
        self.assertAllEqual(gt.target_in, expected[1],
                    'mismatch in target in sequence {} batch'.format(num_batch))
        
        self.assertAllEqual(gt.target_out, expected[2],
                    'mismatch in target out sequence {} batch'.format(num_batch))
        
        self.assertAllEqual(gt.source_size, expected[3],
                    'mismatch in source sequence sizes {} batch'.format(num_batch))
        
        self.assertAllEqual(gt.target_size, expected[4],
                    'mismatch in target sequence sizes {} batch'.format(num_batch))
        
        self.assertAllEqual(gt.sos_token_id, expected[5],
                            'sos tokens mismatch')
        
        self.assertAllEqual(gt.eos_token_id, expected[6],
                            'eos tokens mismatch')
        
        
        
    def testTrainInput(self):
        
        try1 = BatchedInput(source = np.array([[3, 4, 5, 10, 5, 5, 5, 5, 5, 5],
                                               [6, 7, 8, 6, 9, 11, 2, 2, 2, 2]]),
                            target_in = np.array([[1, 3, 4, 5, 10, 2, 2],
                                                  [1, 6, 11, 7, 8, 9, 12]]),
                            target_out = np.array([[3, 4, 5, 10, 2, 2, 2],
                                                   [6, 11, 7, 8, 9, 12, 2]]),
                            source_size = np.array([10, 6]),
                            target_size = np.array([5, 7]),
                            initializer = None,
                            sos_token_id = 1,
                            eos_token_id = 2
                            )
        
        try2 = BatchedInput(source = np.array([[11, 10, 4, 8]]),
                            target_in = np.array([[1, 10, 11, 12, 7, 8]]),
                            target_out = np.array([[10, 11, 12, 7, 8, 2]]),
                            source_size = np.array([4]),
                            target_size = np.array([6]),
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
        
        
        res1 = self.sess.run([iterator.source,
                              iterator.target_in,
                              iterator.target_out,
                              iterator.source_size,
                              iterator.target_size,
                              iterator.sos_token_id,
                              iterator.eos_token_id])
        
        self._check_all_equal(res1, try1, 1)
        
        
        res2 = self.sess.run([iterator.source,
                              iterator.target_in,
                              iterator.target_out,
                              iterator.source_size,
                              iterator.target_size,
                              iterator.sos_token_id,
                              iterator.eos_token_id])
        
        
        self._check_all_equal(res2, try2, 2)
        
        
        self.assertAllEqual(total_num, 3,
                            'mismatch in total num somehow')
        
        
    def testDevInput(self):
        
        try1 = BatchedInput(source = np.array([[6, 7, 8],
                                               [6, 9, 2]]),
                            target_in = np.array([[1, 7, 8, 9, 5, 3, 3, 3, 3, 3, 3],
                                                  [1, 3, 4, 0, 2, 2, 2, 2, 2, 2, 2]]),
                            target_out = np.array([[7, 8, 9, 5, 3, 3, 3, 3, 3, 3, 2],
                                                   [3, 4, 0, 2, 2, 2, 2, 2, 2, 2, 2]]),
                            source_size = np.array([3, 2]),
                            target_size = np.array([11, 4]),
                            initializer = None,
                            sos_token_id = 1,
                            eos_token_id = 2
                            )
        
        try2 = BatchedInput(source = np.array([[5, 0]]),
                            target_in = np.array([[1, 10, 10, 0]]),
                            target_out = np.array([[10, 10, 0, 2]]),
                            source_size = np.array([2]),
                            target_size = np.array([4]), 
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
        
        
        res1 = self.sess.run([iterator.source,
                              iterator.target_in,
                              iterator.target_out,
                              iterator.source_size,
                              iterator.target_size,
                              iterator.sos_token_id,
                              iterator.eos_token_id])
        
        self._check_all_equal(res1, try1, 1)
        
        
        res2 = self.sess.run([iterator.source,
                              iterator.target_in,
                              iterator.target_out,
                              iterator.source_size,
                              iterator.target_size,
                              iterator.sos_token_id,
                              iterator.eos_token_id])
        
        
        self._check_all_equal(res2, try2, 2)
        
        
        self.assertAllEqual(total_num, 3,
                            'mismatch in total num somehow')
        
    
   
        
if __name__ == '__main__':
    
    tf.test.main()
            
        