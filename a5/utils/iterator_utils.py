# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 20:53:48 2020

@author: User
"""


import tensorflow as tf
from collections import namedtuple
from utils import misc_utils

UNK_ID = 0


class BatchedInput(namedtuple('BatchedInput', 
                              ['src_chars',
                               'trg_chars_in',
                               'trg_chars_out',
                               'trg_chars_lens',
                               'trg_words_in',
                               'trg_words_out',
                               'src_size',
                               'trg_size',
                               'initializer',
                               'sos_token_id',
                               'eos_token_id'])):
    
    pass


class DataPack(namedtuple('data_pack',
                          ['src_dataset',
                           'trg_dataset',
                           'total_num'])):
    
    pass


def _get_data(regime,
              filesobj):
    
    assert regime in ['TRAIN', 'DEV', 'TEST'], 'wrong regime {}'.format(regime)
    
    if regime == 'TRAIN':
        
        total_num = misc_utils.count_num_lines(filesobj.src_train)
        
        src_name = tf.gfile.Glob(filesobj.src_train)
        
        trg_name = tf.gfile.Glob(filesobj.trg_train)
        
        src_dataset = tf.data.TextLineDataset(src_name)
        
        trg_dataset = tf.data.TextLineDataset(trg_name)
        
    
    if regime == 'DEV':
        
        total_num = misc_utils.count_num_lines(filesobj.src_dev)
        
        src_name = tf.gfile.Glob(filesobj.src_dev)
        
        trg_name = tf.gfile.Glob(filesobj.trg_dev)
        
        src_dataset = tf.data.TextLineDataset(src_name)
        
        trg_dataset = tf.data.TextLineDataset(trg_name)
        
    
    if regime == 'TEST':
        
        total_num = misc_utils.count_num_lines(filesobj.src_test)
        
        src_name = tf.gfile.Glob(filesobj.src_test)
        
        trg_name = tf.gfile.Glob(filesobj.trg_test)
        
        src_dataset = tf.data.TextLineDataset(src_name)
        
        trg_dataset = tf.data.TextLineDataset(trg_name)
        
        
    return DataPack(src_dataset = src_dataset,
                    trg_dataset = trg_dataset,
                    total_num = total_num) 


def _get_lookup_tables(src_char_vcb_file,
                       trg_char_vcb_file,
                       trg_vcb_file):
    
    src_char_lookup_table = tf.contrib.lookup.index_table_from_file(src_char_vcb_file,
                                                        default_value = UNK_ID)
    
    trg_char_lookup_table = tf.contrib.lookup.index_table_from_file(trg_char_vcb_file,
                                                        default_value = UNK_ID)

    trg_lookup_table = tf.contrib.lookup.index_table_from_file(trg_vcb_file,
                                                        default_value = UNK_ID)    
    
    return src_char_lookup_table, trg_char_lookup_table, trg_lookup_table
    
    
    


def _get_batched_input(regime,
                       src_dataset,
                       trg_dataset,
                       src_char_lookup_table,
                       trg_char_lookup_table,
                       trg_lookup_table,
                       num_epochs,
                       batch_size,
                       shuffle_buffer_size,
                       src_maxlen = 30,
                       trg_maxlen = 30,
                       sos_token = '<s>',
                       eos_token = '</s>',
                       debug_mode = False):
    
    trg_sos_id = tf.cast(trg_lookup_table.lookup(tf.constant(sos_token)),
                         dtype = tf.int32)
    
    trg_eos_id = tf.cast(trg_lookup_table.lookup(tf.constant(eos_token)),
                         dtype = tf.int32)
    
    
    dataset = tf.data.Dataset.zip((src_dataset, trg_dataset))
    
    dataset = dataset.map(lambda src, trg: (tf.string_split([src]).values,
                                       tf.string_split([trg]).values))

    dataset = dataset.filter(lambda src, trg: tf.logical_and(tf.size(src) > 0,
                                                        tf.size(trg) > 0))

    if src_maxlen is not None:
        
        dataset = dataset.map(lambda src, trg: (src[:src_maxlen],
                                               trg))
        
    if trg_maxlen is not None:
        
        dataset = dataset.map(lambda src, trg: (src,
                                               trg[:trg_maxlen]))
        
    dataset = dataset.map(lambda src, trg: (src,
                                            trg,
                                           tf.cast(trg_lookup_table.lookup(trg),
                                                  dtype = tf.int32)))
    
    dataset = dataset.map(lambda src, raw_trg, trg: (src,
                                                    raw_trg,
                                                    tf.concat(([trg_sos_id], trg), axis = 0),
                                                    tf.concat((trg, [trg_eos_id]), axis = 0)))
    
    dataset = dataset.map(lambda src, raw_trg, trg_in, trg_out: (src,
                                                                raw_trg,
                                                                trg_in,
                                                                trg_out,
                                                                tf.size(src),
                                                                tf.size(trg_in)))
    
    dataset = dataset.map(lambda src, raw_trg, trg_in, trg_out, src_size, trg_size: \
                      ((tf
                        .string_split(src, sep = '', result_type = 'RaggedTensor')
                        .to_tensor(default_value = eos_token)),
                       (tf
                        .string_split(raw_trg, sep = '', result_type = 'RaggedTensor')),
                      trg_in,
                      trg_out,
                      src_size,
                       trg_size
                      ))
        
    dataset = dataset.map(lambda src_char_tensor, char_trg, trg_in, trg_out,
                                  src_size, trg_size:\
                      (tf.cast(src_char_lookup_table.lookup(src_char_tensor),
                               dtype = tf.int32),
                       tf.ragged.map_flat_values(trg_char_lookup_table.lookup,
                                                 char_trg),
                       tf.add(tf.cast(char_trg.row_lengths(axis = 1),
                                      dtype = tf.int32),
                                    tf.constant(1)),
                      trg_in,
                      trg_out,
                      src_size,
                      trg_size)
                     )
        
    dataset = dataset.map(lambda src_char_tensor, char_trg, char_trg_lens, trg_in,
                          trg_out, src_size, trg_size:\
                      (src_char_tensor,
                       tf.cast(char_trg.to_tensor(tf.cast(trg_eos_id,
                                                            dtype = tf.int64)),
                                                      dtype = tf.int32),
                      char_trg_lens,
                      trg_in,
                      trg_out,
                      src_size,
                      trg_size)
                     )
        
    dataset = dataset.map(lambda src_char_tensor, char_trg_tensor, char_trg_lens,
                          trg_in, trg_out, src_size, trg_size:\
                      (src_char_tensor,
                        tf.map_fn(lambda word: tf.concat(([trg_sos_id], word), axis = 0),
                                                       char_trg_tensor),
                        tf.map_fn(lambda word: tf.concat((word, [trg_eos_id]), axis = 0),
                                                       char_trg_tensor),
                      char_trg_lens,
                      trg_in,
                      trg_out,
                      src_size,
                      trg_size)
                     )
        
    if regime == 'TRAIN' and not debug_mode:
        
        dataset = (dataset
                   .shuffle(shuffle_buffer_size,
                            reshuffle_each_iteration = True)
                   )
        
    dataset = (dataset
               .repeat(num_epochs)
               .padded_batch(batch_size,
                             padded_shapes = (
                                 tf.TensorShape([None, None]),
                                 tf.TensorShape([None, None]),
                                 tf.TensorShape([None, None]),
                                 tf.TensorShape([None]),
                                 tf.TensorShape([None]),
                                 tf.TensorShape([None]),
                                 tf.TensorShape([]),
                                 tf.TensorShape([])
                                 ),
                             padding_values = (
                                 trg_eos_id,
                                 trg_eos_id,
                                 trg_eos_id,
                                 0,
                                 trg_eos_id,
                                 trg_eos_id,
                                 0,
                                 0
                                 ),
                             drop_remainder = False)
               )
    
    iterator = dataset.make_initializable_iterator()
        
    src_char_tensor, trg_char_tensor_in, trg_char_tensor_out, trg_char_lens, \
        trg_words_in, trg_words_out, src_size, trg_size = iterator.get_next()
        
    return BatchedInput(src_chars = src_char_tensor,
                        trg_chars_in = trg_char_tensor_in, 
                        trg_chars_out = trg_char_tensor_out, 
                        trg_chars_lens = trg_char_lens, 
                        trg_words_in = trg_words_in,
                        trg_words_out = trg_words_out, 
                        src_size = src_size,
                        trg_size = trg_size,
                        initializer = iterator.initializer,
                        sos_token_id = trg_sos_id,
                        eos_token_id = trg_eos_id)
        
    



def get_iterator(regime,
                 filesobj,
                 buffer_size,
                 num_epochs,
                 batch_size,
                 debug_mode = False):
    
    with tf.name_scope('inputs'):
    
        data = _get_data(regime,
                        filesobj)
        
        src_char_lookup_table, trg_char_lookup_table, trg_lookup_table = \
                    _get_lookup_tables((filesobj
                                     .src_char_vcb_file),
                                    (filesobj
                                     .trg_char_vcb_file),
                                    (filesobj
                                     .trg_vcb_file))
        
        iterator  = _get_batched_input(regime,
                                       data.src_dataset,
                                       data.trg_dataset,
                                       src_char_lookup_table,
                                       trg_char_lookup_table,
                                       trg_lookup_table,
                                       num_epochs,
                                       batch_size,
                                       buffer_size,
                                       debug_mode = debug_mode
                                      )
    
    return iterator, data.total_num
    