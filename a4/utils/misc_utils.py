# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 20:58:00 2020

@author: User
"""
from math import ceil
from numpy import mean
from collections import namedtuple
from tqdm import tqdm

class HParams(namedtuple('hparams',
                         ['regime',
                          'filesobj',
                          'buffer_size',
                          'num_epochs',
                          'batch_size',
                          'model_type',
                          'logdir',
                          'src_embeddings_matrix_file',
                          'trg_embeddings_matrix_file',
                          'num_units',
                          'learning_rate',
                          'translation_file_path',
                          'num_steps_to_eval',
                          'chkpts_dir'])):
    
    pass


class Files(namedtuple('files',
                       ['src_vcb_file',
                        'trg_vcb_file',
                        'src_train',
                        'trg_train',
                        'src_dev',
                        'trg_dev',
                        'src_test',
                        'trg_test'])):
    
    pass


def count_num_lines(file_path):
    
    count = 0
    
    with open(file_path) as file:
        
        for line in file:
            
            count += 1
            
    return count


def count_num_steps(num_epochs,
                    data_size,
                    batch_size,
                    include_last_batch = True):
    
    if include_last_batch:
        
        return ceil(data_size * num_epochs / batch_size)
    
    else:
    
        return data_size * num_epochs // batch_size
    
    
def eval_once(train_model, 
              dev_model,
              train_sess,
              dev_sess,
              current_step,
              chkpts_dir,
              dev_iterator,
              num_dev_steps,
              dev_table_init_op):
    
    losses = []
    
        
    current_chkpt_path = train_model.saver.save(train_sess,
                                               chkpts_dir,
                                               current_step)
    
    dev_model.saver.restore(dev_sess,
                            current_chkpt_path)
    
    
    dev_sess.run(dev_table_init_op)
    
    dev_sess.run(dev_iterator.initializer)
                 
    with tqdm(total = num_dev_steps) as prog:
        
        for eval_step in range(num_dev_steps):
            
            batch_loss = dev_model.evaluate(dev_sess)
            
            losses.append(batch_loss)
            
            prog.update(1)
        
    
    return mean(losses)


def write_translations(train_model,
                       train_sess,
                       chkpts_dir,
                       current_step,
                       test_model,
                       test_sess,
                       test_table_init_op,
                       test_iterator,
                       num_test_steps):
    
    current_chkpt_path = train_model.saver.save(train_sess,
                                               chkpts_dir,
                                               current_step)
    
    test_model.saver.restore(test_sess,
                            current_chkpt_path)
    
    
    test_sess.run(test_table_init_op)
    
    test_sess.run(test_iterator.initializer)
                 
    
    for step in range(num_test_steps):
        
        test_model.infer(test_sess)
        
        
def get_train_hparams():
    
    train_files = Files(src_vcb_file = './data/vcbs/src_vcb.txt',
                        trg_vcb_file = './data/vcbs/trg_vcb.txt',
                        src_train = './data/train/src.es',
                        trg_train = './data/train/trg.en',
                        src_dev = './data/dev/src.es',
                        trg_dev = './data/dev/trg.en',
                        src_test = './data/test/src.es',
                        trg_test = './data/test/trg.en')
    
    return HParams(regime = 'TRAIN',
                   filesobj = train_files,
                   buffer_size = 100,
                   num_epochs = 1,
                   batch_size = 100,
                   model_type = 'simple_model',
                   logdir = './logs/train_logs',
                   src_embeddings_matrix_file = './data/pretrained_embeddings/src_embeddings_matrix.p',
                   trg_embeddings_matrix_file = './data/pretrained_embeddings/trg_embeddings_matrix.p', 
                   num_units = 64,
                   learning_rate = 3e-04,
                   translation_file_path = None,
                   num_steps_to_eval = None,
                   chkpts_dir = './chkpts/'
                   )

def get_dev_hparams():
    
    dev_files = Files(src_vcb_file = './data/vcbs/src_vcb.txt',
                        trg_vcb_file = './data/vcbs/trg_vcb.txt',
                        src_train = './data/train/src.es',
                        trg_train = './data/train/trg.en',
                        src_dev = './data/dev/src.es',
                        trg_dev = './data/dev/trg.en',
                        src_test = './data/test/src.es',
                        trg_test = './data/test/trg.en')
    
    return HParams(regime = 'DEV',
                   filesobj = dev_files,
                   buffer_size = None,
                   num_epochs = None,
                   batch_size = 100,
                   model_type = 'simple_model',
                   logdir = './logs/dev_logs',
                   src_embeddings_matrix_file = './data/pretrained_embeddings/src_embeddings_matrix.p',
                   trg_embeddings_matrix_file = './data/pretrained_embeddings/trg_embeddings_matrix.p',  
                   num_units = 64,
                   learning_rate = 3e-04,
                   translation_file_path = None,
                   num_steps_to_eval = 500,
                   chkpts_dir = None
                   )

def get_test_hparams():
    
    test_files = Files(src_vcb_file = './data/vcbs/src_vcb.txt',
                        trg_vcb_file = './data/vcbs/trg_vcb.txt',
                        src_train = './data/train/src.es',
                        trg_train = './data/train/trg.en',
                        src_dev = './data/dev/src.es',
                        trg_dev = './data/dev/trg.en',
                        src_test = './data/test/src.es',
                        trg_test = './data/test/trg.en')
    
    return HParams(regime = 'TEST',
                   filesobj = test_files,
                   buffer_size = None,
                   num_epochs = None,
                   batch_size = 100,
                   model_type = 'simple_model',
                   logdir = './logs/test_logs',
                   src_embeddings_matrix_file = './data/pretrained_embeddings/src_embeddings_matrix.p',
                   trg_embeddings_matrix_file = './data/pretrained_embeddings/trg_embeddings_matrix.p', 
                   num_units = 64,
                   learning_rate = 3e-04,
                   translation_file_path = './translations/translations.txt',
                   num_steps_to_eval = None,
                   chkpts_dir = None
                   )