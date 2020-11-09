# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 20:53:48 2020

@author: User
"""


import tensorflow as tf
from collections import namedtuple
from utils import misc_utils

UNK_ID = 0

class BatchedInput(namedtuple('batched_input', 
                              ['source',
                               'target_in',
                               'target_out',
                               'source_size',
                               'target_size',
                               'initializer',
                               'sos_token_id',
                               'eos_token_id'
                               ])):
    
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


def _get_lookup_tables(src_vcb_file,
                       trg_vcb_file):
    
    src_lookup_table = tf.contrib.lookup.index_table_from_file(src_vcb_file,
                                                default_value = UNK_ID)
    
    trg_lookup_table = tf.contrib.lookup.index_table_from_file(trg_vcb_file,
                                                default_value = UNK_ID)
    
    return src_lookup_table, trg_lookup_table
    
    


def _get_batched_input(src_dataset,
                 trg_dataset,
                 src_lookup_table,
                 trg_lookup_table,
                 regime,
                 buffer_size,
                 num_epochs,
                 batch_size,
                 debug_mode,
                 max_src_len = 30,
                 max_trg_len = 30,
                 sos_token = '<s>',
                 eos_token = '</s>'):
    
    trg_sos_id = tf.cast(trg_lookup_table.lookup(tf.constant(sos_token)),
                     tf.int32)
    
    trg_eos_id = tf.cast(trg_lookup_table.lookup(tf.constant(eos_token)),
                     tf.int32)
    
    src_eos_id = tf.cast(src_lookup_table.lookup(tf.constant(eos_token)),
                     tf.int32)
    
    dataset = tf.data.Dataset.zip((src_dataset, trg_dataset))
    
    dataset = dataset.map(lambda src, trg: (tf.string_split([src]).values,
                                            tf.string_split([trg]).values))
    
    dataset = dataset.filter(lambda src, trg: tf.logical_and(tf.size(src) > 0,
                                                             tf.size(trg) > 0))
    
    if max_src_len is not None:
        
        dataset = dataset.map(lambda src, trg: (src[:max_src_len],
                                                trg))
        
    if max_trg_len is not None:
        
        dataset = dataset.map(lambda src, trg: (src,
                                                trg[:max_trg_len]))
        
    dataset = dataset.map(lambda src, trg: (tf.cast(src_lookup_table.lookup(src),
                                                    tf.int32),
                                            tf.cast(trg_lookup_table.lookup(trg),
                                                    tf.int32)))
                          
    dataset = dataset.map(lambda src, trg: (src,
                                            tf.concat(([trg_sos_id], trg), axis = 0),
                                            tf.concat((trg, [trg_eos_id]), axis = 0))
                          )
    
    dataset = dataset.map(lambda src, in_trg, out_trg: (src,
                                                        in_trg,
                                                        out_trg,
                                                        tf.size(src),
                                                        tf.size(out_trg)))
    
    if regime == 'TRAIN' and not debug_mode:
        
        dataset = (dataset
                   .shuffle(buffer_size,
                            reshuffle_each_iteration = True)
                   .repeat(num_epochs)
                   )
        
    if regime == 'TRAIN' and debug_mode:
        
        dataset = (dataset
                   .repeat(num_epochs)
                   )
        
        
    dataset = (dataset
               .padded_batch(batch_size,
                             padded_shapes = (
                                 tf.TensorShape([None]),
                                 tf.TensorShape([None]),
                                 tf.TensorShape([None]),
                                 tf.TensorShape([]),
                                 tf.TensorShape([])),
                             padding_values = (src_eos_id,
                                               trg_eos_id,
                                               trg_eos_id,
                                               0,
                                               0),
                             drop_remainder = False))
    
    iterator = dataset.make_initializable_iterator()
    
    src_seq, trg_in_seq, trg_out_seq, src_size, trg_size  = iterator.get_next()
    
    return BatchedInput(source = src_seq,
                        target_in = trg_in_seq,
                        target_out = trg_out_seq,
                        source_size = src_size,
                        target_size = trg_size,
                        initializer = iterator.initializer,
                        sos_token_id = trg_sos_id,
                        eos_token_id = trg_eos_id
                        )


def get_iterator(regime,
                 filesobj,
                 buffer_size,
                 num_epochs,
                 batch_size,
                 debug_mode = False):
    
    with tf.name_scope('inputs'):
    
        data = _get_data(regime,
                        filesobj)
        
        src_lookup_table, trg_lookup_table = _get_lookup_tables((filesobj
                                                                 .src_vcb_file),
                                                                (filesobj
                                                                 .trg_vcb_file))
        
        iterator  = _get_batched_input(data.src_dataset,
                                      data.trg_dataset,
                                      src_lookup_table,
                                      trg_lookup_table,
                                      regime,
                                      buffer_size,
                                      num_epochs,
                                      batch_size,
                                      debug_mode
                                      )
    
    return iterator, data.total_num
    