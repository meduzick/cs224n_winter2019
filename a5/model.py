# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:39:52 2020

@author: User
"""


import tensorflow as tf
from utils import vcb_utils, misc_utils
from utils.iterator_utils import UNK_ID
from layers import (cnn_encoder, highway_layer, distil_layer,
                    decoder_unroll_layer, char_decoder_unroll_layer)
from abc import ABC, abstractmethod
from collections import namedtuple

class TrainOutputTuple(namedtuple('train_output_tuple',
                                  ['train_summary',
                                   'global_step'])):
    
    pass

class EvalOutputTuple(namedtuple('eval_output_tuple',
                                 ['eval_loss'])):
    
    pass

class InferOutputTuple(namedtuple('infer_output_tuple',
                                  ['sample_chars',
                                   'sample_words'])):
    
    pass

class TranslationTokens(namedtuple('translations',
                                   ['word_ids',
                                    'char_ids',
                                    'unk_ids',
                                    'max_len'])):
    
    pass


class BaseModel(ABC):
    
    def __init__(self,
                 hparams,
                 iterator,
                 regime,
                 id2string_lookup_table = None,
                 id2char_lookup_table = None):
        
        self._init_parameters(hparams)
        
        translation_pack, loss = self._build_graph(regime,
                                                   iterator)
        
        self._set_train_or_infer(regime,
                                 loss,
                                 translation_pack,
                                 id2string_lookup_table,
                                 id2char_lookup_table)
        
    def _init_parameters(self, 
                         hparams):
        
        self.global_step = tf.get_variable('global_step',
                                            shape = [],
                                            initializer = tf.constant_initializer(0),
                                            trainable = False,
                                            dtype = tf.int32)
        
        self._summaries = []
                                
        self._trg_embeddings_matrix_file = hparams.trg_embeddings_matrix_file
        
        self._trg_char_vcb_file = hparams.trg_char_vcb_file
        
        self._src_char_vcb_file = hparams.src_char_vcb_file
        
        self._num_units = hparams.num_units
        
        self._learning_rate = hparams.learning_rate
        
        self._kernel_size = hparams.kernel_size
        
        self._word_emb_dim = hparams.word_emb_dim
        
        self._stride = hparams.stride
        
        self._char_emb_dim = hparams.char_emb_dim
        
        
        
    def _locate_variable(self,
                         name,
                         shape,
                         initializer,
                         trainable,
                         dtype,
                         wd = None):
        
        var = tf.get_variable(name = name,
                              shape = tf.TensorShape(shape),
                              initializer = initializer,
                              trainable = trainable,
                              dtype = dtype)
        
        if wd is not None:
            
            tf.add_to_collection('losses', wd * tf.reduce_sum(tf.square(var)))
            
        return var
    
    
    def _get_embedding_matricies(self):
        
        trg_embeddings_init, (self._trg_vcb_size, self._trg_embeddings_dim) = \
            vcb_utils.get_embeddings_initializer((self._trg_embeddings_matrix_file))
            
        
        self._trg_char_vcb_len = misc_utils.count_num_lines(self._trg_char_vcb_file)
        
        src_char_vcb_len = misc_utils.count_num_lines(self._src_char_vcb_file)
            
        
        trg_embeddings = self._locate_variable('trg_embeddings',
                                               shape = [self._trg_vcb_size,
                                                    self._trg_embeddings_dim],
                                               initializer = trg_embeddings_init,
                                               trainable = False, 
                                               dtype = tf.float32)
        
        trg_char_embeddings = self._locate_variable('trg_char_embeddings',
                                                    shape = [self._trg_char_vcb_len,
                                                             self._char_emb_dim],
                                                    initializer = tf.random_normal_initializer,
                                                    trainable = True,
                                                    dtype = tf.float32)
        
        src_char_embeddings = self._locate_variable('src_char_embeddings',
                                                    shape = [src_char_vcb_len,
                                                             self._char_emb_dim],
                                                    initializer = tf.random_normal_initializer,
                                                    trainable = True,
                                                    dtype = tf.float32)
        
        return src_char_embeddings, trg_char_embeddings, trg_embeddings
    
    
    def _build_encoder(self,
                       encoder_input,
                       input_lens):
        
        fw_rnn_cell = tf.nn.rnn_cell.LSTMCell(self._num_units)
        
        bw_rnn_cell = tf.nn.rnn_cell.LSTMCell(self._num_units)
        
        outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
                                                        cell_fw = fw_rnn_cell,
                                                        cell_bw = bw_rnn_cell,
                                                        inputs = encoder_input,
                                                        sequence_length = input_lens,
                                                        dtype = tf.float32
                                                               )
        
        return tf.concat(outputs,
                         axis = -1), \
            tf.nn.rnn_cell.LSTMStateTuple(c = tf.concat((final_state[0].c,
                                                        final_state[1].c), 
                                                       axis = -1),
                                         h = tf.concat((final_state[0].h,
                                                        final_state[1].h), 
                                                       axis = -1))
            
    
    @abstractmethod
    def _build_decoder(self,
                       regime,
                       iterator,
                       encoder_state,
                       hidden_states):
        
        pass
    
    
    @classmethod
    def _calculate_loss(self,
                        word_logits,
                        word_labels,
                        char_logits,
                        char_labels,
                        seq_lens,
                        word_lens):
        
        with tf.name_scope('word_component'):
            
            word_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                        logits = word_logits,
                                                        labels = word_labels
                                                                )
            
            word_mask = tf.cast(tf.sequence_mask(lengths = seq_lens,
                                         maxlen = tf.reduce_max(seq_lens)),
                                dtype = tf.float32)
            
            word_loss = tf.reduce_mean(tf.reduce_sum(word_loss * word_mask, axis = 1))
                
        
        with tf.name_scope('char_component'):
            
            max_word_len = tf.shape(char_labels)[-1]
            
            char_labels = tf.reshape(char_labels,
                                  shape = [-1, 
                                           max_word_len])
    
            word_lens = tf.reshape(word_lens,
                                   shape = [-1])
            
            
            mask_ids = tf.where(tf.greater(word_lens, tf.constant(0)))
            
            word_lens = tf.reshape(tf.gather(word_lens, mask_ids),
                               shape = [-1])
    
            char_labels = tf.reshape(tf.gather(char_labels,
                                               mask_ids),
                                          shape = [-1,
                                                   max_word_len])
            
            char_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                        logits = char_logits,
                                                        labels = char_labels
                                                                )
            
            char_mask = tf.cast(tf.sequence_mask(lengths = word_lens,
                                         maxlen = tf.reduce_max(word_lens)),
                                dtype = tf.float32)
            
            char_loss = tf.reduce_sum(char_loss * char_mask)
            
            
        return word_loss + char_loss + tf.reduce_sum(tf.get_collection('losses'))
        
    
    def _build_graph(self,
                     regime,
                     iterator):
        
        with tf.variable_scope('embeddings_matricies'):
            
            self.source_chars_embeddings_matrix, self.target_chars_embeddings_matrix, \
                self.target_embeddings_matrix = \
                    self._get_embedding_matricies()
                    
        
        with tf.variable_scope('encoder'):
            
            with tf.variable_scope('cnn_encoder'):
                
                filters = self._locate_variable('cnn_filters',
                                        shape = [self._kernel_size,
                                                 self._char_emb_dim,
                                                 self._word_emb_dim],
                                        initializer = tf.random_normal_initializer,
                                        trainable = True,
                                        dtype = tf.float32)
                
                conv_output = cnn_encoder(iterator.src_chars,
                                           self.source_chars_embeddings_matrix,
                                           filters,
                                           self._stride)
                
                
            with tf.variable_scope('highway_layer'):
                
                w_h = self._locate_variable('transform_weights',
                                            shape = [self._word_emb_dim,
                                                     self._word_emb_dim],
                                            initializer = tf.random_normal_initializer,
                                            trainable = True,
                                            dtype = tf.float32)
                
                w_t = self._locate_variable('transition_weights',
                                            shape = [self._word_emb_dim,
                                                     self._word_emb_dim],
                                            initializer = tf.random_normal_initializer,
                                            trainable = True,
                                            dtype = tf.float32)
                
                b_h = self._locate_variable('transform_bias',
                                            shape = [self._word_emb_dim,],
                                            initializer = tf.random_normal_initializer,
                                            trainable = True,
                                            dtype = tf.float32)
                
                transition_b_init = tf.constant_initializer([-1] * self._word_emb_dim)
                
                b_t = self._locate_variable('transition_bias',
                                            shape = [self._word_emb_dim,],
                                            initializer = transition_b_init,
                                            trainable = True,
                                            dtype = tf.float32)
                
                encoder_input = highway_layer(conv_output,
                                              w_h,
                                              w_t,
                                              b_h,
                                              b_t)
                
                
            encoder_outputs, encoder_state = self._build_encoder(encoder_input,
                                                          iterator.src_size)
            
        with tf.variable_scope('decoder'):
            
           (word_logits, char_logits, sample_id, char_sample_id, unk_ids, 
                max_word_len) = self._build_decoder(regime,
                                                    iterator,
                                                    encoder_state,
                                                    encoder_outputs)
                                         
        with tf.name_scope('loss'):
            
            if regime != 'TEST':
            
                loss = self._calculate_loss(word_logits,
                                            iterator.trg_words_out,
                                            char_logits,
                                            iterator.trg_chars_out,
                                            iterator.trg_size,
                                            iterator.trg_chars_lens)
                
            else:
                
                loss = None
            
        return (TranslationTokens(word_ids = sample_id, 
                                 char_ids = char_sample_id,
                                 unk_ids = unk_ids,
                                 max_len = max_word_len), loss)
                                         
        
    def _set_train_or_infer(self,
                            regime,
                            loss,
                            translation_pack,
                            id2string_lookup_table,
                            id2char_lookup_table):
        
        assert regime in ['TRAIN', 'DEV', 'TEST'], 'wrong regime {}'.format(regime)  
        
        if regime == 'TRAIN':
            
            with tf.name_scope('optimization'):
                
                optimizer = tf.train.AdamOptimizer(self._learning_rate)
                
                grads_vars = optimizer.compute_gradients(loss)
                
                self.optimization_step = optimizer.apply_gradients(grads_vars,
                                                global_step = self.global_step)
                
            with tf.name_scope('gradients'):
                
                for grad, var in grads_vars:
                    
                    if grad is not None:
                        
                        self._summaries.append((tf.summary
                                                .histogram(('grad_for_var_{v}'
                                                            .format(v = var.op.name)),
                                                    grad)))
                    

            with tf.name_scope('variables'):
                
                for var in tf.trainable_variables():
                    
                    self._summaries.append(tf.summary.histogram(('val_for_{v}'
                                                                 .format(v = var.op.name)),
                                                  var))
                    
            
            self.train_summary = tf.summary.merge(self._summaries + 
                                         [tf.summary.scalar('loss',
                                                           loss)])
            
        if regime == 'DEV':
            
            self.loss = loss
        
        
        if regime == 'TEST':
            
            words_sample_id = translation_pack.word_ids
            
            chars_sample_id = translation_pack.char_ids
                        
            self.sample_words = id2string_lookup_table.lookup(tf.cast(words_sample_id,
                                                            dtype = tf.int64))
            
            self.sample_chars = id2char_lookup_table.lookup(tf.cast(chars_sample_id,
                                                                    dtype = tf.int64))
            
        with tf.name_scope('saver'):
            
            self.saver = tf.train.Saver()
            
            
    def train(self, sess):
        
        output_tuple = TrainOutputTuple(train_summary = self.train_summary,
                                        global_step = self.global_step)
        
        return sess.run([self.optimization_step, output_tuple])
    
    
    def evaluate(self, sess):
        
        output_tuple = EvalOutputTuple(eval_loss = self.loss)
        
        return sess.run(output_tuple)
    
    def infer(self, sess):
        
        output_tuple = InferOutputTuple(sample_chars = self.sample_chars,
                                        sample_words = self.sample_words)
        
        return sess.run(output_tuple)
            
class AttentionHybridModel(BaseModel):
    
    def _build_decoder(self,
                       regime, 
                       iterator,
                       encoder_state,
                       hidden_states):
        
        assert regime in ['TRAIN', 'DEV', 'TEST'], 'wrong regime {}'.format(regime)
        
        with tf.variable_scope('word_level_decoder'):
            
            self.word_projection_layer = tf.layers.Dense(self._trg_vcb_size,
                                                         use_bias = False)
            
            transition_layer = tf.layers.Dense(2 * self._num_units,
                                               use_bias = False)
        
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                                            num_units = 2 * self._num_units, ### This is for memory layer
                                                                             ### for scoring function 
                                            memory = hidden_states,
                                            memory_sequence_length = iterator.src_size
                                            ### Memory seq lens is for _prepare_memory function
                                            ### inside attention mechanism, it zeros out
                                            ### output vectors for padded tokens in memory
                                            ### BUT, memory itself is output of 
                                            ### dynamic rnn, and it zeros out padded
                                            ### tokens, if seq lens provided
                                            ### 
                                            ### It is also used to replace all
                                            ### row scores in allignments ([bs, max_time])
                                            ### beyond proper length of seq
                                            ### with -np.inf (so the exp is 0)
                                            )
                    
            cell = tf.nn.rnn_cell.BasicLSTMCell(2 * self._num_units)
            
            ### Cell_input_fn is designed in the way to allow concatenation
            ### of the current inputs (embedding) and previous attention vector
            ### hence the dimensions of the matricies of the rnn cell are 
            ### [emb_dim + src_hidden_units + trg_hidden_units, trg_hidden_units]
            
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
                            attention_layer = tf.layers.Dense(2 * self._num_units,
                                                              activation = tf.math.tanh,
                                                              use_bias = False)
                                            ) 
            
            attention_state = attention_cell.zero_state(batch_size = tf.shape(iterator.trg_words_in)[0],
                                                        dtype = tf.float32)
            
            attention_state = attention_state.clone(cell_state = encoder_state)
                    
            if regime != 'TEST':
                
                outputs = decoder_unroll_layer(self.target_embeddings_matrix, 
                                               iterator.trg_words_in, 
                                               iterator.trg_size, 
                                               cell = attention_cell, 
                                               init_state = attention_state,
                                               projection_layer = self.word_projection_layer)
                
                word_logits = outputs.rnn_output 
                                
                sample_id = None
                
            else:
                
                start_tokens = tf.fill(tf.expand_dims(tf.size(iterator.src_size),
                                                  axis = 0),
                                   iterator.sos_token_id)
            
                maximum_iterations = tf.cast(tf.reduce_max(iterator.src_size) * 2,
                                             dtype = tf.int32)
                
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding = self.target_embeddings_matrix,
                    start_tokens = start_tokens,
                    end_token = iterator.eos_token_id)
                
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell = attention_cell,
                    helper = helper,
                    initial_state = attention_state,
                    output_layer = self.word_projection_layer)
                
                outputs, final_state, words_seq_lenghts = tf.contrib.seq2seq.dynamic_decode(
                    decoder = decoder,
                    maximum_iterations = maximum_iterations,
                    scope = 'decoder_outter_scope'
                    )
                 
                sample_id = outputs.sample_id
                 
                word_logits = outputs.rnn_output
                
            
            combined_outputs = transition_layer(word_logits)

                                
                
        with tf.variable_scope('char_level_decoder'):
            
            self.char_projection_layer = tf.layers.Dense(self._trg_char_vcb_len,
                                                         use_bias = False)
            
            char_decoder_cell = tf.nn.rnn_cell.LSTMCell(2 * self._num_units)

            
            if regime != 'TEST': 
            
                char_level_input = tf.nn.embedding_lookup(self.target_chars_embeddings_matrix,
                                                          iterator.trg_chars_in)
                
                char_decoder_input, char_decoder_lens, char_decoder_inits = \
                    distil_layer(char_level_input,
                                 iterator.trg_chars_lens,
                                 combined_outputs,
                                 2 * self._num_units,
                                 self._char_emb_dim)
                    
                
                char_decoder_init_states = tf.nn.rnn_cell.LSTMStateTuple(c = char_decoder_inits,
                                                                         h = char_decoder_inits)
                
                char_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs = char_decoder_input,
                    sequence_length = char_decoder_lens,
                    )
                
                char_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell = char_decoder_cell,
                    helper = char_helper,
                    initial_state = char_decoder_init_states,
                    output_layer = self.char_projection_layer 
                    )
                
                char_outputs, char_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder = char_decoder,
                    scope = 'char_decoder_outter_scope'
                    )
                
                char_logits = char_outputs.rnn_output
                
                char_sample_id = None
                
                unk_ids = None
                
                max_word_len = None
                
            else:
                                
                maximum_char_iterations = tf.cast(tf.shape(iterator.src_chars)[-1] * 2,
                                 dtype = tf.int32)
                                
                char_sample_id, max_word_len, unk_ids = char_decoder_unroll_layer(
                                            self.target_chars_embeddings_matrix,
                                            char_decoder_cell,
                                            self.char_projection_layer,
                                            combined_outputs,
                                            sample_id, 
                                            words_seq_lenghts, 
                                            self._num_units, 
                                            UNK_ID, 
                                            iterator.sos_token_id,
                                            iterator.eos_token_id,
                                            maximum_char_iterations)
                
                char_logits = None

            
        return (word_logits, char_logits, sample_id, char_sample_id, unk_ids, 
                max_word_len)