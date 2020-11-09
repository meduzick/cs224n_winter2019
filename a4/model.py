# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:00:35 2020

@author: User
"""


import tensorflow as tf
from utils import vcb_utils
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
                                  ['sample_ids',
                                   'sample_words'])):
    
    pass


class BaseModel(ABC):
    
    def __init__(self,
                 hparams,
                 iterator,
                 regime,
                 id2string_lookup_table = None):
        
        self._init_parameters(hparams)
        
        sample_id, loss = self._build_graph(regime,
                          iterator)
        
        self._set_train_or_infer(regime,
                                 loss,
                                 sample_id,
                                 id2string_lookup_table)
        
    
    def _init_parameters(self, 
                         hparams):
        
        self.global_step = self._locate_variable('global_step',
                                                 shape = [],
                                                 initializer = tf.constant_initializer(0),
                                                 trainable = False,
                                                 dtype = tf.int32)
        
        self._summaries = []
                
        self._src_embeddings_matrix_file = hparams.src_embeddings_matrix_file
        
        self._trg_embeddings_matrix_file = hparams.trg_embeddings_matrix_file
        
        self._num_units = hparams.num_units
        
        self._learning_rate = hparams.learning_rate
        
        
        
        
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
        
        src_embeddings_init, (src_vcb_size, self._src_embeddings_dim) = \
            vcb_utils.get_embeddings_initializer((self._src_embeddings_matrix_file))
            
        trg_embeddings_init, (self._trg_vcb_size, self._trg_embeddings_dim) = \
            vcb_utils.get_embeddings_initializer((self._trg_embeddings_matrix_file))
            
        src_embeddings = self._locate_variable('src_embeddings',
                                               shape = [src_vcb_size,
                                                    self._src_embeddings_dim],
                                               initializer = src_embeddings_init,
                                               trainable = False, 
                                               dtype = tf.float32)
        
        trg_embeddings = self._locate_variable('trg_embeddings',
                                               shape = [self._trg_vcb_size,
                                                    self._trg_embeddings_dim],
                                               initializer = trg_embeddings_init,
                                               trainable = False, 
                                               dtype = tf.float32)
        
        return src_embeddings, trg_embeddings
        
        
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
    
    def _return_loss_mask(self,
                          original_lens,
                          max_batch_len):
        
        indices = tf.expand_dims(tf.range(max_batch_len), axis = 0)

        batch_lens = tf.expand_dims(original_lens, axis = 1)
        
        weights = tf.cast(tf.math.greater(batch_lens, indices),
                          dtype = tf.float32)
        
        return weights
            
        
    def _build_graph(self, 
                     regime,
                     iterator):
        
        with tf.variable_scope('embeddings_matricies'):
            
            self.source_embeddings_matrix, self.target_embeddings_matrix = \
                    self._get_embedding_matricies()
                    
        
        ### Variable creation does not happen here (dense kernel)
        ### hence use of variable scope in this place is not valid
                                
        self.projection_layer = tf.layers.Dense(self._trg_vcb_size,
                                                    use_bias = False)
            
            
        with tf.variable_scope('encoder'):
            
            encoder_input = tf.nn.embedding_lookup(self.source_embeddings_matrix,
                                                   iterator.source)
            
            encoder_outputs, encoder_state = self._build_encoder(encoder_input,
                                                          iterator.source_size)
                        
        with tf.variable_scope('decoder'):
            
            
            logits, sample_id = self._build_decoder(regime,
                                         iterator,
                                         encoder_state,
                                         encoder_outputs)
            
        with tf.name_scope('loss'):
            
            weights = self._return_loss_mask(iterator.target_size,
                                             tf.shape(iterator.target_out)[1])
            
            batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                logits = logits,
                                                labels = iterator.target_out)
            
            loss = tf.reduce_mean(tf.reduce_sum(batch_loss * weights, axis = 1)) + \
                tf.reduce_sum(tf.get_collection('losses'))
            
        return sample_id, loss
        
    def _set_train_or_infer(self,
                            regime,
                            loss,
                            sample_id,
                            id2string_lookup_table):
        
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
            
            self.sample_id = sample_id
            
            self.sample_words = id2string_lookup_table.lookup(tf.cast(sample_id,
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
        
        output_tuple = InferOutputTuple(sample_ids = self.sample_id,
                                        sample_words = self.sample_words)
        
        return sess.run(output_tuple)
            
        
        
class SimpleModel(BaseModel):
    
    def _build_decoder(self,
                       regime,
                       iterator,
                       encoder_state,
                       hidden_states):
        
        assert regime in ['TRAIN', 'DEV', 'TEST'], 'wrong regime {}'.format(regime)
        
        cell = tf.nn.rnn_cell.BasicLSTMCell(2 * self._num_units)
        
        if regime != 'TEST':
            
            decoder_input = tf.nn.embedding_lookup(self.target_embeddings_matrix,
                                                    iterator.target_in)
                    
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = decoder_input,
                sequence_length = iterator.target_size,
                )
            
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = cell,
                helper = helper,
                initial_state = encoder_state,
                output_layer = self.projection_layer
                )
            
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder,
                scope = 'decoder_outter_scope'
                )
            
            sample_id = outputs.sample_id
            
            logits = outputs.rnn_output
            
        else:
            
            start_tokens = tf.fill(tf.expand_dims(tf.size(iterator.source_size),
                                                  axis = 0),
                                   iterator.sos_token_id)
            
            maximum_iterations = tf.cast(tf.reduce_max(iterator.source_size) * 2,
                                         dtype = tf.int32)
            
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding = self.target_embeddings_matrix,
                start_tokens = start_tokens,
                end_token = iterator.eos_token_id)
            
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = cell,
                helper = helper,
                initial_state = encoder_state,
                output_layer = self.projection_layer)
            
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder,
                maximum_iterations = maximum_iterations,
                scope = 'decoder_outter_scope'
                )
             
            sample_id = outputs.sample_id
             
            logits = outputs.rnn_output
             
        return logits, sample_id
    
    
class AttentionModel(BaseModel):
    
    def _build_decoder(self,
                       regime, 
                       iterator,
                       encoder_state,
                       hidden_states):
        
        
        assert regime in ['TRAIN', 'DEV', 'TEST'], 'wrong regime {}'.format(regime)
        
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                                        num_units = 2 * self._num_units,
                                        memory = hidden_states,
                                        memory_sequence_length = iterator.source_size
                                        )
                
        cell = tf.nn.rnn_cell.BasicLSTMCell(2 * self._num_units)
        
        ### Cell_input_fn is design in the way to allow concatenation
        ### of the current inputs (embedding) and previous attention vector
        ### hence the dimensions of the matricies of the rnn cell are 
        ### [emb_dim + src_hidden_units + trg_hidden_units, trg_hidden_units]
        
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                                        cell = cell,
                                        attention_mechanism = attention_mechanism,
                                        attention_layer_size = 2 * self._num_units,
                                        cell_input_fn = lambda inputs, attention: \
                                            tf.concat([inputs, attention],
                                                     axis = -1)
                                        )
        
        attention_state = attention_cell.zero_state(batch_size = tf.shape(iterator.target_in)[0],
                                                    dtype = tf.float32)
        
        attention_state = attention_state.clone(cell_state = encoder_state)
                
        if regime != 'TEST':
            
            decoder_input = tf.nn.embedding_lookup(self.target_embeddings_matrix,
                                                    iterator.target_in)
                    
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = decoder_input,
                sequence_length = iterator.target_size,
                )
            
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = attention_cell,
                helper = helper,
                initial_state = attention_state,
                output_layer = self.projection_layer
                )
            
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder,
                scope = 'decoder_outter_scope'
                )
            
            sample_id = outputs.sample_id
            
            logits = outputs.rnn_output
            
        else:
            
            start_tokens = tf.fill(tf.expand_dims(tf.size(iterator.source_size),
                                                  axis = 0),
                                   iterator.sos_token_id)
            
            maximum_iterations = tf.cast(tf.reduce_max(iterator.source_size) * 2,
                                         dtype = tf.int32)
            
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding = self.target_embeddings_matrix,
                start_tokens = start_tokens,
                end_token = iterator.eos_token_id)
            
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = attention_cell,
                helper = helper,
                initial_state = attention_state,
                output_layer = self.projection_layer)
            
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder,
                maximum_iterations = maximum_iterations,
                scope = 'decoder_outter_scope'
                )
             
            sample_id = outputs.sample_id
             
            logits = outputs.rnn_output
             
             
        return logits, sample_id