# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:32:23 2020

@author: User
"""


import tensorflow as tf
import numpy as np
from layers import (cnn_encoder, highway_layer, decoder_unroll_layer,
                    distil_layer, char_decoder_unroll_layer)

class LayersFunctionsTest(tf.test.TestCase):
    
    def setUp(self):
        
        super(LayersFunctionsTest, self).setUp()
        
        self.graph = tf.Graph()
        
        self.sess = tf.Session(graph = self.graph)
        
    
    def testCnnEncoderLayer(self):
        
        inputs = np.array(
                            [
                                [
                                    [1, 2, 3],
                                    [4, 5, 1]
                                ],
                                [
                                    [2, 4, 5],
                                    [0, 1, 1]
                                ]
                            ]
                        )
        
        embeddings = np.array(
                                [
                                    [0, 0, 0],
                                    [0, 0, 1],
                                    [0, 1, 1],
                                    [1, 1, 1],
                                    [0, 1, 0],
                                    [1, 0, 0]
                                ]
                            )
        
        filters = np.array(
                            [[
                                [0, 1],
                                [1, 0],
                                [1, 1]
                            ],
                            [
                                [0, 2],
                                [2, 0],
                                [2, 2]
                            ]]
                        )
        
        expected_res = np.array(
                                [
                                    [
                                        [6, 5],
                                        [2, 3]
                                    ],
                                    [
                                        [4, 2],
                                        [3, 3]
                                    ]
                                ]
                            )
        
        with self.graph.as_default():
            
            input_chars = tf.constant(inputs)
    
            emb_matrix = tf.get_variable('embs',
                                        shape = tf.TensorShape(embeddings.shape),
                                        initializer = tf.constant_initializer(embeddings),
                                        trainable = False)
            
            filters_var = tf.get_variable('filters',
                                        shape = tf.TensorShape(filters.shape),
                                        initializer = tf.constant_initializer(filters),
                                        trainable = False)
            
            conv_output = cnn_encoder(input_chars, 
                                     emb_matrix, 
                                     filters_var, 
                                     1)
            
            init_op = tf.global_variables_initializer()
            
        
        self.sess.run(init_op)
        
        output = self.sess.run(conv_output)
        
        self.assertAllEqual(output, expected_res, 
                    'failed with conv encoder layer')
        
    
    def testHighwayLayer(self):
        
        inputs = np.array(
                            [
                                [
                                    [1, 2, 3],
                                    [2, 3, 4]
                                ],
                                [
                                    [0, 1, 2],
                                    [0, 0, 1]
                                ]
                            ]
                        )
        
        w_h = np.array(
                            [
                                [1, 1, 0],
                                [1, 0, 1],
                                [0, 0, 1]
                            ]
                        )
        
        w_t = np.array(
                            [
                                [1, 0, 1],
                                [1, 1, 0],
                                [1, 1, 1]
                            ]
                        )
        
        b_h = np.array([0, 0, 0])
        
        b_t = np.array([-1, -1, -1])
        
        
        expected_res = np.array([[[2.9866143 , 1.01798621, 4.90514825],
                                 [4.99899395, 2.00247262, 6.97992145]],
                         
                                [[0.88079708, 0.11920292, 2.73105858],
                                 [0.        , 0.        , 1.        ]]]
                                )
        
        
        with self.graph.as_default():
            
            hw_input = tf.constant(inputs,
                        dtype = tf.float32)
    
            wh = tf.get_variable('wh',
                                shape = w_h.shape,
                                initializer = tf.constant_initializer(w_h),
                                dtype = tf.float32)
            
            wt = tf.get_variable('wt',
                                shape = w_t.shape,
                                initializer = tf.constant_initializer(w_t),
                                dtype = tf.float32)
            
            bh = tf.get_variable('bh',
                                shape = b_h.shape,
                                initializer = tf.constant_initializer(b_h),
                                dtype = tf.float32)
            
            bt = tf.get_variable('bt',
                                shape = b_t.shape,
                                initializer = tf.constant_initializer(b_t),
                                dtype = tf.float32)
            
            out = highway_layer(hw_input, wh, wt, bh, bt)
            
            init_op = tf.global_variables_initializer()
            
        
        self.sess.run(init_op)
        
        output = self.sess.run(out)
        
        self.assertAllClose(output, expected_res, 
                    msg = 'failed with highway layer')
        
    def testDecoderUnrollLayer(self):
        
        hiddens = np.array(
                            [
                                [
                                    [1, 0],
                                    [0, 1],
                                    [0, 0]
                                ],
                                [
                                    [1, 1],
                                    [-1, 1],
                                    [0, 0] ### manually set to zeros, because
                                           ### it tries to mimic behaviour of _prepare_memory
                                           ### function in attention mechanism
                                ]
                            ]
                        )

        src_lens = np.array([3, 2])


        inputs = np.array(
                            [
                                [0, 1, 2],
                                [3, 4, 5]
                            ]
                        )

        embeddings = np.array(
                            [
                                [1, 0, 1],
                                [1, 1, 0],
                                [1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 1],
                                [1, 1, 1]
                            ]
                        )

        lens = np.array([2, 3])
        
        i_w = np.array(
                            [
                                [1, 0],
                                [0, 1],
                                [1, 1],
                                [0, 0],
                                [-1, 0],
                                [-1, -1],
                                [2, 3]
                            ]
                        )

        j_w = np.array(
                            [
                                [2, 1],
                                [1, 2],
                                [0, 2],
                                [2, 0],
                                [-2, 0],
                                [1, 1],
                                [0, 0]
                            ]
                        )
        
        f_w = np.array(
                            [
                                [0, 0],
                                [-2, 1],
                                [0, 1],
                                [1, 0],
                                [1, 1],
                                [-1, -1],
                                [0, -1]
                            ]
                        )

        o_w = np.array(
                            [
                                [1, 0],
                                [0, 1],
                                [-1, 0],
                                [0, -1],
                                [0, 0],
                                [1, 1],
                                [-1, -1]
                            ]
                        )

        weights = np.concatenate([i_w, j_w, f_w, o_w], axis = 1)

        scoring_attention_matrix = np.array(
                            [
                                [1, 0],
                                [1, 1],
                            ]
                        )
        
        attention_layer_matrix = np.array(
                            [
                                [1, 0], 
                                [0, 1],
                                [1, 1],
                                [0, 0]
                            ]
                        )
        
        projection_matrix = np.array(
                            [
                                [1, 2, 3],
                                [4, 5, 6]
                            ]
                        )
        
        expected_res = np.array(
                                [[[3.21566436, 4.52266218, 5.82966],
                                 [4.23862632, 5.96526894, 7.69191156],
                                 [3.57543327, 5.24724654, 6.91905981]],
                         
                                [[1.21476168, 1.5184521 , 1.82214252],
                                 [2.83353715, 3.63336593, 4.43319471],
                                 [3.93314365, 5.33087906, 6.72861447]]]
                            )
        
        with self.graph.as_default():
            
            inputs = tf.constant(inputs, dtype = tf.int32)

            embeddings = tf.constant(embeddings, dtype = tf.float32)
            
            lens = tf.constant(lens, dtype = tf.int32)
            
            weights = tf.constant_initializer(weights,
                                 dtype = tf.float32)
            
            hiddens = tf.constant(hiddens, dtype = tf.float32)
            
            src_lens = tf.constant(src_lens, dtype = tf.int32)
            
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                                            num_units = 2,
                                            memory = hiddens,
                                            memory_sequence_length = src_lens
                                            )
                    
            cell = tf.nn.rnn_cell.LSTMCell(2,
                                           initializer = weights)


            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell = cell,
                attention_mechanism = attention_mechanism,
                attention_layer_size = None,### this allows for customization of 
                                            ### attention layer
                cell_input_fn = lambda inputs, attention: \
                    tf.concat([inputs, attention],
                             axis = -1),
                output_attention = True,### this allows for outputing
                                        ### attention vector on each time step
                attention_layer = tf.layers.Dense(2,
                     activation = None,
                     use_bias = False,
                     kernel_initializer = \
                         tf.constant_initializer(attention_layer_matrix))
                                )
                
            init_state = attention_cell.zero_state(2, tf.float32)
            
            projection_layer = tf.layers.Dense(3,
                                  use_bias = False,
                                  kernel_initializer = \
                                      tf.constant_initializer(projection_matrix))
            
            res = decoder_unroll_layer(embeddings,
                                       inputs, 
                                       lens,
                                       attention_cell, 
                                       init_state,
                                       projection_layer)
            
            kernel_ref = (tf
                          .get_default_graph()
                          .get_tensor_by_name('memory_layer/kernel:0'))

            kernel_init_op = tf.assign(kernel_ref,
                                       tf.constant(scoring_attention_matrix,
                                                   dtype = tf.float32))
            
            
            init_op = tf.initialize_variables(tf.trainable_variables())
            
        self.sess.run(init_op)
        
        self.sess.run(kernel_init_op)
    
        output = self.sess.run(res)
        
        self.assertAllClose(output.rnn_output, expected_res, 
                    msg = 'failed with decoder unroll layer')
        
        
    def testDistilLayer(self):
        
        inputs = np.array(
                            [
                                [
                                    [
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 1]
                                    ],
                                    [
                                        [0, 0, 1, 0],
                                        [0, 1, 0, 0]
                                    ],
                                    [
                                        [1, 0, 0, 0],
                                        [1, 0, 0, 1]
                                    ]
                                ],
                                [
                                    [
                                        [1, 0, 1, 0],
                                        [1, 1, 0, 0]
                                    ],
                                    [
                                        [1, 1, 0, 1],
                                        [1, 1, 1, 0]
                                    ],
                                    [
                                        [1, 1, 1, 1],
                                        [1, -1, -1, 1]
                                    ]
                                ]
                            ]
                        )
    
        lens = np.array(
                            [
                                [2, 0, 2],
                                [0, 0, 2]
                            ]
                        )

        combs = np.array(
                            [
                                [
                                    [0, 0],
                                    [0, 1],
                                    [1, 0]
                                ],
                                [
                                    [1, 1],
                                    [1, 0],
                                    [0, 1]
                                ],
                            ]
                        )

        expected_inputs = np.array(
                            [
                                [
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 1]
                                ],
                                [
                                    [1, 0, 0, 0],
                                    [1, 0, 0, 1]
                                ],
                                [
                                    [1, 1, 1, 1],
                                    [1, -1, -1, 1]
                                ]
                            ]
                        )

        expected_lens = np.array(
                            [2, 2, 2]
                        )

        expected_combs = np.array(
                            [
                                [0, 0],
                                [1, 0],
                                [0, 1]
                            ]
                        )
        
        with self.graph.as_default():
            
            inputs = tf.constant(inputs)

            lens = tf.constant(lens)
            
            combs = tf.constant(combs)
            
            res = distil_layer(inputs,
                               lens, 
                               combs,
                               2,
                               4)
            
        output = self.sess.run(res)
            
        self.assertAllEqual(output[0], expected_inputs,
                            'wrong inputs')
        
        self.assertAllEqual(output[1], expected_lens,
                            'wrong lens')
        
        self.assertAllEqual(output[2], expected_combs,
                            'wrong inits')
        
    
    def testCharDecoderUnrollLayer(self):
        
        combined_outputs = np.array(
                                    [
                                        [
                                            [0, 0],
                                            [0, 1],
                                            [1, 0]
                                        ],
                                        [
                                            [1, 0],
                                            [1, 1],
                                            [0, 1]
                                        ],
                                    ]
                                )

        sample_id = np.array(
                                [
                                    [0, 1, 2],
                                    [3, 0, 0],
                                ]
                            )
        
        words_lens = np.array([1, 3])
        
        embeddings = np.array(
                                [
                                    [1, 0],
                                    [3, -2],
                                    [0, 1],
                                    [1, 1],
                                    [1, 2]
                                ]
                            )

        i_w = np.array(
                        [
                            [1, 0],
                            [0, 1],
                            [1, 1],
                            [0, 0],
                        ]
                    )
        
        j_w = np.array(
                        [
                            [2, 1],
                            [1, 2],
                            [0, 2],
                            [2, 0],
                        ]
                    )
        
        f_w = np.array(
                        [
                            [0, 0],
                            [-2, 1],
                            [0, 1],
                            [1, 0],
                        ]
                    )
        
        o_w = np.array(
                        [
                            [1, 0],
                            [0, 1],
                            [-1, 0],
                            [0, -1],
                        ]
                    )

        weights = np.concatenate([i_w, j_w, f_w, o_w], axis = 1)
        
        projection_weights = np.array(
                                        [
                                            [1, 0, -1, 1, 0],
                                            [0, 1, -1, 1, 1]
                                        ]
                                    )
        
        sos_token_id = 1
        
        eos_token_id = 2
        
        unk_id = 0
        
        max_char_iterations = 3
        
        
        expected_ids = np.array(
                                [
                                    [0, 3, 3],
                                    [3, 3, 3],
                                    [3, 3, 3]
                                ]
                            )

        expected_word_len = 3
        
        expected_unk_ids = np.array(
                                    [
                                        [0],
                                        [4],
                                        [5]
                                    ]
                                )
                                        
        
        with self.graph.as_default():
            
            combined_outputs = tf.constant(combined_outputs, dtype = tf.float32)

            sample_id = tf.constant(sample_id, dtype = tf.int32)
            
            words_lens = tf.constant(words_lens, dtype = tf.int32)
            
            embeddings = tf.get_variable('E',
                                        shape = embeddings.shape,
                                        initializer = tf.constant_initializer(embeddings),
                                        dtype = tf.float32)
            
            
            cell = tf.nn.rnn_cell.LSTMCell(2,
                    initializer = tf.constant_initializer(weights,
                                                          dtype = tf.float32))
            
            projection_layer = tf.layers.Dense(5,
                                              use_bias = False,
                            kernel_initializer = \
                                tf.constant_initializer(projection_weights,
                                                        dtype = tf.float32))
                
            res = char_decoder_unroll_layer(embeddings, 
                                            cell,
                                            projection_layer,
                                            combined_outputs,
                                            sample_id,
                                            words_lens,
                                            1,
                                            unk_id,
                                            sos_token_id,
                                            eos_token_id,
                                            max_char_iterations)
            
            self.sess.run(tf.variables_initializer(tf.trainable_variables()))
            
            output = self.sess.run(res)
            
            self.assertAllEqual(output[0], expected_ids,
                                'wrong ids')
            
            self.assertAllEqual(output[1], expected_word_len,
                                'wrong word len')
            
            self.assertAllEqual(output[2], expected_unk_ids,
                                'wrong unk ids')



if __name__ == '__main__':
    
    tf.test.main()