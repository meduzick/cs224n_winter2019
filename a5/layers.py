# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:19:14 2020

@author: User
"""


import tensorflow as tf


def cnn_encoder(source_chars,
                 source_chars_embeddings_matrix,
                 filters,
                 stride):
        
        source_chars_embedded = tf.nn.embedding_lookup(source_chars_embeddings_matrix,
                                                       source_chars)
        
        batch_size = tf.shape(source_chars_embedded)[0]
        
        max_sent_len = tf.shape(source_chars_embedded)[1]
        
        max_word_len = tf.shape(source_chars_embedded)[2]
        
        char_emb_dim = tf.shape(source_chars_embedded)[3]
        
        filters_dim = tf.shape(filters)[2]
        
        source_chars_embedded = tf.reshape(source_chars_embedded,
                                           shape = [-1, 
                                                    max_word_len,
                                                    char_emb_dim])
        
        conv_output = tf.nn.conv1d(source_chars_embedded,
                                   filters = filters,
                                   stride = stride,
                                   padding = 'VALID',
                                   data_format = 'NWC')
        
        conv_output = tf.reduce_max(tf.nn.relu(conv_output),
                                    axis = 1)
        
        conv_output = tf.reshape(conv_output,
                                 shape = [batch_size,
                                          max_sent_len,
                                          filters_dim])
        
        return conv_output
    

def highway_layer(conv_output,
                   w_h,
                   w_t,
                   b_h,
                   b_t):
    
    batch_size = tf.shape(conv_output)[0]
    
    max_sent_len = tf.shape(conv_output)[1]
    
    word_emb_dim = tf.shape(conv_output)[2]
    
    fixed_word_emb_dim = tf.shape(w_h)[1]
    
    
    conv_output = tf.reshape(conv_output,
                             shape = [-1, word_emb_dim])
    
    transform = tf.nn.relu(tf.matmul(conv_output, w_h) + b_h)
    
    transition = tf.math.sigmoid(tf.matmul(conv_output, w_t) + b_t)
    
    highway_output = tf.multiply(transform, transition) + \
                        tf.multiply(conv_output, (1 - transition))
                        
                        
    highway_output = tf.reshape(highway_output,
                                shape = [batch_size,
                                         max_sent_len,
                                         fixed_word_emb_dim])
    
    return highway_output
        
        

def distil_layer(char_level_input,
                 char_level_lens,
                 combined_outputs,
                 cell_units,
                 fixed_char_emb):
        
    max_word_len = tf.shape(char_level_input)[2]
    
    char_level_input = tf.reshape(char_level_input,
                                  shape = [-1, 
                                           max_word_len,
                                           fixed_char_emb])
    
    char_level_lens = tf.reshape(char_level_lens,
                                 shape = [-1])
    
    combined_outputs = tf.reshape(combined_outputs,
                                  shape = [-1,
                                           cell_units])
    
    
    mask_ids = tf.where(tf.greater(char_level_lens, tf.constant(0)))
    
    non_zero_lens = tf.reshape(tf.gather(char_level_lens, mask_ids),
                               shape = [-1])
    
    non_zero_lens_char_input = tf.reshape(tf.gather(char_level_input,
                                                    mask_ids),
                                          shape = [-1,
                                                   max_word_len,
                                                   fixed_char_emb])
    
    non_zero_comb_outputs = tf.reshape(tf.gather(combined_outputs,
                                                 mask_ids),
                                       shape = [-1,
                                                cell_units])
        
    return non_zero_lens_char_input, non_zero_lens, non_zero_comb_outputs


def decoder_unroll_layer(target_embeddings_matrix,
                        target_in,
                        target_size,
                        cell,
                        init_state,
                        projection_layer):
    
    decoder_input = tf.nn.embedding_lookup(target_embeddings_matrix,
                                          target_in)
                        
    helper = tf.contrib.seq2seq.TrainingHelper(
        inputs = decoder_input,
        sequence_length = target_size,
        )
    
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell = cell,
        helper = helper,
        initial_state = init_state,
        output_layer = projection_layer 
        )
    
    outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder = decoder,
        scope = 'decoder_outter_scope'
        )
    
    return outputs

def char_decoder_unroll_layer(target_chars_embeddings_matrix,
                              char_decoder_cell,
                              char_projection_layer,
                              combined_outputs,
                              word_sample_id,
                              words_seq_lenghts,
                              num_units,
                              unk_id,
                              sos_token_id,
                              eos_token_id,
                              max_char_iterations
                              ):
    
    combined_outputs = tf.reshape(combined_outputs,
                                shape = [-1,
                                         2 * num_units])
                
                
    max_word_len = tf.shape(word_sample_id)[1]
    
    mask = tf.sequence_mask(words_seq_lenghts, 
                            max_word_len)
    
    sample_cp = tf.where(mask,
                         word_sample_id,
                         tf.fill(tf.shape(word_sample_id), -1))

    sample_cp = tf.reshape(sample_cp, shape = [-1])
    
    unk_ids = tf.where(tf.math.equal(sample_cp, unk_id))
    
    
    char_decoder_inits = tf.reshape(tf.gather(combined_outputs,
                                     unk_ids),
                                    shape = [-1,
                                             2 * num_units])
    
    char_decoder_inits = tf.nn.rnn_cell.LSTMStateTuple(c = char_decoder_inits,
                                                       h = char_decoder_inits)
    
    
    start_char_tokens = tf.fill(tf.expand_dims(tf.size(unk_ids),
                                      axis = 0),
                                sos_token_id)
    
    char_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embedding = target_chars_embeddings_matrix,
        start_tokens = start_char_tokens,
        end_token = eos_token_id)
    
    char_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell = char_decoder_cell,
        helper = char_helper,
        initial_state = char_decoder_inits,
        output_layer = char_projection_layer)
    
    char_outputs, char_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder = char_decoder,
        maximum_iterations = max_char_iterations,
        scope = 'char_decoder_outter_scope'
        )
    
    char_sample_id = char_outputs.sample_id
        
    return char_sample_id, max_word_len, unk_ids