# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 20:27:31 2020

@author: User
"""


import tensorflow as tf
from model_helper import BuildTrainModel, BuildEvalModel, BuildInferModel
from utils import misc_utils, iterator_utils
from tqdm import tqdm

def run():
    
    train_hparams = misc_utils.get_train_hparams()
    
    dev_hparams = misc_utils.get_dev_hparams()
    
    test_hparams = misc_utils.get_test_hparams()
    
    
    train_graph = tf.Graph()
    
    dev_graph = tf.Graph()
    
    test_graph = tf.Graph()
    
    with train_graph.as_default():
        
        train_iterator, train_total_num = iterator_utils.get_iterator(
                                        regime = train_hparams.regime,
                                        filesobj = train_hparams.filesobj,
                                        buffer_size = train_hparams.buffer_size,
                                        num_epochs = train_hparams.num_epochs,
                                        batch_size = train_hparams.batch_size)
        
        train_model = BuildTrainModel(hparams = train_hparams,
                                      iterator = train_iterator,
                                      graph = train_graph)
        
        train_vars_init_op = tf.global_variables_initializer()
        
        train_tables_init_op = tf.tables_initializer()
        
        
        tf.get_default_graph().finalize()
        
        
    with dev_graph.as_default():
        
        dev_iterator, dev_total_num = iterator_utils.get_iterator(
                                        dev_hparams.regime,
                                        filesobj = dev_hparams.filesobj,
                                        buffer_size = dev_hparams.buffer_size,
                                        num_epochs = dev_hparams.num_epochs,
                                        batch_size = dev_hparams.batch_size)
        
        dev_model = BuildEvalModel(hparams = dev_hparams,
                                   iterator = dev_iterator,
                                   graph = dev_graph)
        
        dev_tables_init_op = tf.tables_initializer()
        
        
        tf.get_default_graph().finalize()
        
        
    with test_graph.as_default():
        
        test_iterator, test_total_num = iterator_utils.get_iterator(
                                        test_hparams.regime,
                                        filesobj = test_hparams.filesobj,
                                        buffer_size = test_hparams.buffer_size,
                                        num_epochs = test_hparams.num_epochs,
                                        batch_size = test_hparams.batch_size)
        
        test_model = BuildInferModel(hparams = test_hparams,
                        iterator = test_iterator,
                        graph = test_graph,
                        infer_file_path = test_hparams.translation_file_path,
                        infer_chars_file_path = test_hparams.char_translation_file_path)
        
        test_tables_init_op = tf.tables_initializer()
        
        
        tf.get_default_graph().finalize()
        
    
    train_session = tf.Session(graph = train_graph)
    
    dev_session = tf.Session(graph = dev_graph)
    
    test_session = tf.Session(graph = test_graph)
    
    
    train_steps = misc_utils.count_num_steps(train_hparams.num_epochs,
                                             train_total_num,
                                             train_hparams.batch_size)
    
    eval_steps = misc_utils.count_num_steps(1,
                                             dev_total_num,
                                             dev_hparams.batch_size)
    
    num_test_steps = misc_utils.count_num_steps(1, 
                                                test_total_num,
                                                test_hparams.batch_size)
    
    eval_count = dev_hparams.num_steps_to_eval
    
    
    train_session.run(train_tables_init_op)
    
    train_session.run(train_iterator.initializer)
    
    train_session.run(train_vars_init_op)
                    
    
    with tqdm(total = train_steps) as prog:
        
        for step in range(train_steps):
        
            train_model.train(train_session)
            
            if step % eval_count == 0:
                
                dev_loss = misc_utils.eval_once(
                            train_model,
                            dev_model,
                            train_session,
                            dev_session,
                            step,
                            train_hparams.chkpts_dir,
                            dev_iterator,
                            eval_steps,
                            dev_tables_init_op)
                
                print('dev loss at step {} = {}'.format(step,
                                                        dev_loss))
                
                
            prog.update(1)   
            
    misc_utils.write_translations(train_model,
                                  train_session,
                                  train_hparams.chkpts_dir,
                                  step,
                                  test_model,
                                  test_session,
                                  test_tables_init_op,
                                  test_iterator,
                                  num_test_steps)
    
    

if __name__ == '__main__':
    
    run()
    