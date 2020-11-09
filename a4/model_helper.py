# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 15:59:21 2020

@author: User
"""


import tensorflow as tf
from model import SimpleModel, AttentionModel

class Model(object):
    
    def __init__(self,
                 model,
                 logdir,
                 graph,
                 infer_file_path = None):
        
        self.model = model
        
        self.writer = tf.summary.FileWriter(logdir,
                                            graph = graph)
                
        self.saver = model.saver
        
        self.infer_file_path = infer_file_path
        
       
    def train(self, sess):
        
        raise NotImplementedError('method has not been implemented yet')
        
    def evaluate(self, sess):
        
        raise NotImplementedError('method has not been implemented yet')
        
    def infer(self, sess):
        
        raise NotImplementedError('method has not been implemented yet')
        

class TrainModel(Model):
    
    def train(self, sess):
        
        _, res = self.model.train(sess)
        
        self.writer.add_summary(res.train_summary,
                                global_step = res.global_step)
        
    
class EvalModel(Model):
    
    def evaluate(self, sess):
        
        dev_output = self.model.evaluate(sess)
        
        return dev_output.eval_loss
    
    
class InferModel(Model):
    
    def infer(self, sess):
        
        res = self.model.infer(sess)
        
        with open(self.infer_file_path, 'a') as file:
            
            for translation in res.sample_words:
                
                text_translation = [elem.decode('utf8') for elem in translation]
                
                file.write(' '.join(text_translation) + '\n')
    
                
def BuildTrainModel(model_type,
                    hparams,
                    iterator,
                    graph):
    
    ### TO DO: add attention model
    
    if model_type == 'simple_model':
        
        model = SimpleModel(hparams = hparams,
                            iterator = iterator,
                            regime = 'TRAIN')
        
    if model_type == 'attention_model':
        
        model = AttentionModel(hparams = hparams,
                               iterator = iterator,
                               regime = 'TRAIN')
    
    return TrainModel(model,
                      hparams.logdir,
                      graph)


def BuildEvalModel(model_type,
                   hparams,
                   iterator,
                   graph):
    
    if model_type == 'simple_model':
        
        model = SimpleModel(hparams = hparams,
                            iterator = iterator,
                            regime = 'DEV')
        
    if model_type == 'attention_model':
        
        model = AttentionModel(hparams = hparams,
                               iterator = iterator,
                               regime = 'DEV')
        
    return EvalModel(model,
                     hparams.logdir,
                     graph)


def BuildInferModel(model_type,
                    hparams,
                    iterator,
                    graph,
                    infer_file_path):
    
    string2id_table = (tf.contrib.lookup.
                       index_to_string_table_from_file(hparams.filesobj.trg_vcb_file,
                                                    default_value = '<unk>'))
    
    if model_type == 'simple_model':
        
        model = SimpleModel(hparams = hparams,
                            iterator = iterator,
                            regime = 'TEST',
                            id2string_lookup_table = string2id_table)
        
    if model_type == 'attention_model':
        
        model = AttentionModel(hparams = hparams,
                               iterator = iterator,
                               regime = 'TEST',
                               id2string_lookup_table = string2id_table)
        
    return InferModel(model,
                     hparams.logdir,
                     graph,
                     infer_file_path)