# -*- encoding: utf-8 -*-
'''
File    :   data_process.py
Time    :   2020/06/23 14:53:49
Author  :   Chao Wang 
Version :   1.0
Contact :   374494067@qq.com
@Desc    :   None
'''





import os
import re
import sys  
import random
import pandas as pd
import numpy as np
import time
# import gensim
# from gensim.models import Word2Vec
# from gensim.models.word2vec import LineSentence
# from keras.preprocessing import text, sequence
import pickle



def w2v_cal(df_train,df_test,maxlen_):
    ad_ids_victor_size = 200
    advertiser_ids_victor_size = 128
    product_ids_victor_size = 100
    industry_ids_victor_size = 50

#############################################################################
# #cal
#     print('function ad_id w2v_pad..')
#     ad_id_tokenizer = text.Tokenizer(num_words=3812202, lower=False,filters="")
#     ad_id_tokenizer.fit_on_texts(list(df_train['ad_ids'].values)+list(df_test['ad_ids'].values))

#     ad_id_train_ = sequence.pad_sequences(ad_id_tokenizer.texts_to_sequences(df_train['ad_ids'].values), maxlen=maxlen_)
#     ad_id_test_ = sequence.pad_sequences(ad_id_tokenizer.texts_to_sequences(df_test['ad_ids'].values), maxlen=maxlen_)
    
#     ad_id_word_index = ad_id_tokenizer.word_index
    
#     nb_words = len(ad_id_word_index)
#     print('ad_id nb_words:',nb_words)

#     print('function advertiser_ids w2v_pad..')
#     advertiser_id_tokenizer = text.Tokenizer(num_words=57870, lower=False,filters="")
#     advertiser_id_tokenizer.fit_on_texts(list(df_train['advertiser_ids'].values)+list(df_test['advertiser_ids'].values))

#     advertiser_id_train_ = sequence.pad_sequences(advertiser_id_tokenizer.texts_to_sequences(df_train['advertiser_ids'].values), maxlen=maxlen_)
#     advertiser_id_test_ = sequence.pad_sequences(advertiser_id_tokenizer.texts_to_sequences(df_test['advertiser_ids'].values), maxlen=maxlen_)
    
#     advertiser_id_word_index = advertiser_id_tokenizer.word_index
    
#     nb_words = len(advertiser_id_word_index)
#     print('adv nb_words:',nb_words)

#     print('function product_ids w2v_pad..')
#     product_id_tokenizer = text.Tokenizer(num_words=39057, lower=False,filters="")
#     product_id_tokenizer.fit_on_texts(list(df_train['product_ids'].values)+list(df_test['product_ids'].values))

#     product_id_train_ = sequence.pad_sequences(product_id_tokenizer.texts_to_sequences(df_train['product_ids'].values), maxlen=maxlen_)
#     product_id_test_ = sequence.pad_sequences(product_id_tokenizer.texts_to_sequences(df_test['product_ids'].values), maxlen=maxlen_)
    
#     product_id_word_index = product_id_tokenizer.word_index
    
#     nb_words = len(product_id_word_index)
#     print('product id nb_words:',nb_words)


##############################################################################

#     print('function ad_id w2v_pad..')
#     ad_id_tokenizer = text.Tokenizer(num_words=3812202, lower=False,filters="")
#     ad_id_tokenizer.fit_on_texts(list(df_train['ad_ids'].values)+list(df_test['ad_ids'].values))

#     ad_id_train_ = sequence.pad_sequences(ad_id_tokenizer.texts_to_sequences(df_train['ad_ids'].values), maxlen=maxlen_)
#     ad_id_test_ = sequence.pad_sequences(ad_id_tokenizer.texts_to_sequences(df_test['ad_ids'].values), maxlen=maxlen_)
    
#     ad_id_word_index = ad_id_tokenizer.word_index
    
#     nb_words = len(ad_id_word_index)
#     print('ad_id nb_words:',nb_words)


#     all_data=pd.concat([df_train['ad_ids'],df_test['ad_ids']])
#     file_name = '../../w2v_model/' + 'Word2Vec_intersection_ad_ids_window_15_start_'  +str(ad_ids_victor_size) + '_sg.model'
#     if not os.path.exists(file_name):
#         print('w2v model training...')
#         model = Word2Vec([[word for word in document.split(' ')] for document in all_data.values],
#                          size=ad_ids_victor_size, window=15, iter=10, workers=11, seed=2018, min_count=2,sg=1)
#         model.save(file_name)
#     else:
#         model = Word2Vec.load(file_name)
#     print("add word2vec finished....")    

#     ad_id_embedding_word2vec_matrix = np.zeros((nb_words + 1, ad_ids_victor_size))
#     for word, i in ad_id_word_index.items():
#         embedding_vector = model[word] if word in model else None
#         if embedding_vector is not None:
#             ad_id_embedding_word2vec_matrix[i] = embedding_vector
#         else:
#             unk_vec = np.random.random(ad_ids_victor_size) * 0.5
#             unk_vec = unk_vec - unk_vec.mean()
#             ad_id_embedding_word2vec_matrix[i] = unk_vec
#     print('finish ad_id w2v...')

 

    print('function advertiser_ids w2v_pad..')
    advertiser_id_tokenizer = text.Tokenizer(num_words=62965, lower=False,filters="")
    advertiser_id_tokenizer.fit_on_texts(list(df_train['advertiser_ids'].values)+list(df_test['advertiser_ids'].values))

    advertiser_id_train_ = sequence.pad_sequences(advertiser_id_tokenizer.texts_to_sequences(df_train['advertiser_ids'].values), maxlen=maxlen_)
    advertiser_id_test_ = sequence.pad_sequences(advertiser_id_tokenizer.texts_to_sequences(df_test['advertiser_ids'].values), maxlen=maxlen_)
    
    advertiser_id_word_index = advertiser_id_tokenizer.word_index
    
    nb_words = len(advertiser_id_word_index)
    print(nb_words)
    all_data=pd.concat([df_train['advertiser_ids'],df_test['advertiser_ids']])
    file_name = '../w2v_model/' + 'Word2Vec_advertiser_ids_window_15_start_advertiser_ids_' + str(advertiser_ids_victor_size) + '_mincount_1_sg.model'
    if not os.path.exists(file_name):
        print('w2v model training...')
        model = Word2Vec([[word for word in document.split(' ')] for document in all_data.values],
                         size=advertiser_ids_victor_size, window=15, iter=30, workers=30, seed=2020, min_count=1,sg=1)
        model.save(file_name)
    else:
        model = Word2Vec.load(file_name)
    print("add word2vec finished....")    

    advertiser_id_embedding_word2vec_matrix = np.zeros((nb_words + 1, advertiser_ids_victor_size))
    for word, i in advertiser_id_word_index.items():
        embedding_vector = model[word] if word in model else None
        if embedding_vector is not None:
            advertiser_id_embedding_word2vec_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(advertiser_ids_victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            advertiser_id_embedding_word2vec_matrix[i] = unk_vec
    print('finish advertiser_ids w2v...')

    print('function product_ids w2v_pad..')
    product_id_tokenizer = text.Tokenizer(num_words=44315, lower=False,filters="")
    product_id_tokenizer.fit_on_texts(list(df_train['product_ids'].values)+list(df_test['product_ids'].values))

    product_id_train_ = sequence.pad_sequences(product_id_tokenizer.texts_to_sequences(df_train['product_ids'].values), maxlen=maxlen_)
    product_id_test_ = sequence.pad_sequences(product_id_tokenizer.texts_to_sequences(df_test['product_ids'].values), maxlen=maxlen_)
    
    product_id_word_index = product_id_tokenizer.word_index
    
    nb_words = len(product_id_word_index)
    print(nb_words)
    all_data=pd.concat([df_train['product_ids'],df_test['product_ids']])
    file_name = '../w2v_model/' + 'Word2Vec_product_ids_window_15_start_product_ids_' + str(product_ids_victor_size) + '_mincount_1_sg.model'
    if not os.path.exists(file_name):
        print('w2v model training...')
        model = Word2Vec([[word for word in document.split(' ')] for document in all_data.values],
                         size=product_ids_victor_size, window=15, iter=30, workers=11, seed=2020, min_count=1,sg=1)
        model.save(file_name)
    else:
        model = Word2Vec.load(file_name)
    print("add word2vec finished....")    

    product_id_embedding_word2vec_matrix = np.zeros((nb_words + 1, product_ids_victor_size))
    for word, i in product_id_word_index.items():
        embedding_vector = model[word] if word in model else None
        if embedding_vector is not None:
            product_id_embedding_word2vec_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(product_ids_victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            product_id_embedding_word2vec_matrix[i] = unk_vec
    print('finish product_ids w2v...')

    # #######################################################
    print('function industry_ids w2v_pad..')
    industry_id_tokenizer = text.Tokenizer(num_words=1000, lower=False,filters="")
    industry_id_tokenizer.fit_on_texts(list(df_train['industry_ids'].values)+list(df_test['industry_ids'].values))

    industry_id_train_ = sequence.pad_sequences(industry_id_tokenizer.texts_to_sequences(df_train['industry_ids'].values), maxlen=maxlen_)
    industry_id_test_ = sequence.pad_sequences(industry_id_tokenizer.texts_to_sequences(df_test['industry_ids'].values), maxlen=maxlen_)
    
    industry_id_word_index = industry_id_tokenizer.word_index
    
    nb_words = len(industry_id_word_index)
    print('industry_ids',nb_words)
    all_data=pd.concat([df_train['industry_ids'],df_test['industry_ids']])
    file_name = '../w2v_model/' + 'Word2Vec_industry_ids_window_15_start_industry_ids_' + str(industry_ids_victor_size) + '_mincount_1_sg.model'
    if not os.path.exists(file_name):
        print('w2v model training...')
        model = Word2Vec([[word for word in document.split(' ')] for document in all_data.values],
                         size=industry_ids_victor_size, window=15, iter=30, workers=11, seed=2020, min_count=1,sg=1)
        model.save(file_name)
    else:
        model = Word2Vec.load(file_name)
    print("add word2vec finished....")    

    industry_id_embedding_word2vec_matrix = np.zeros((nb_words + 1, industry_ids_victor_size))
    for word, i in industry_id_word_index.items():
        embedding_vector = model[word] if word in model else None
        if embedding_vector is not None:
            industry_id_embedding_word2vec_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(industry_ids_victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            industry_id_embedding_word2vec_matrix[i] = unk_vec
    print('finish industry_ids w2v...')
    
#     save_obj(ad_id_train_,'../../embedding/embedding_200/ad_id_train_')
#     save_obj(ad_id_test_,'../../embedding/embedding_200/ad_id_test_')
    save_obj(advertiser_id_train_,'../embedding/embedding_diff/advertiser_id_train_dim128_mincount1')
    save_obj(advertiser_id_test_,'../embedding/embedding_diff/advertiser_id_test_dim128_mincount1')
    save_obj(product_id_train_,'../embedding/embedding_diff/product_id_train_dim100_mincount1')
    save_obj(product_id_test_,'../embedding/embedding_diff/product_id_test_dim100_mincount1')
    
    save_obj(industry_id_train_,'../embedding/embedding_diff/industry_id_train_dim50_mincount1')
    save_obj(industry_id_test_,'../embedding/embedding_diff/industry_id_test_dim50_mincount1')
    
    
#     save_obj(ad_id_embedding_word2vec_matrix,'../../embedding/embedding_200/ad_id_embedding_word2vec_matrix')
    save_obj(advertiser_id_embedding_word2vec_matrix,'../embedding/embedding_diff/advertiser_id_embedding_word2vec_matrix_dim128_mincount1')
    save_obj(product_id_embedding_word2vec_matrix,'../embedding/embedding_diff/product_id_embedding_word2vec_matrix_dim100_mincount1')
    
    save_obj(industry_id_embedding_word2vec_matrix,'../embedding/embedding_diff/industry_id_embedding_word2vec_matrix_dim50_mincount1')



    
    return ad_id_train_,ad_id_test_,advertiser_id_train_,advertiser_id_test_,product_id_train_,product_id_test_,ad_id_embedding_word2vec_matrix,advertiser_id_embedding_word2vec_matrix,product_id_embedding_word2vec_matrix






def w2v_train():
    print('load from pkl...')
    ad_id_train_ = load_obj('../embedding/embedding_200/ad_id_train_')
    
    creative_id_train_ = load_obj('../embedding/embedding_200/creative_id_train_')
    
    advertiser_id_train_ = load_obj('../embedding/embedding_diff/advertiser_id_train_dim128_mincount1')
    
    product_id_train_= load_obj('../embedding/embedding_diff/product_id_train_dim100_mincount1')
    
    industry_id_train_ = load_obj('../embedding/embedding_diff/industry_id_train_dim50_mincount1')
    
    
    return ad_id_train_,creative_id_train_,advertiser_id_train_,product_id_train_,industry_id_train_
def w2v_test():
    print('load from pkl...')
    ad_id_test_ =load_obj('../embedding/embedding_200/ad_id_test_')
    
    creative_id_test_ =load_obj('../embedding/embedding_200/creative_id_test_')
    
    advertiser_id_test_ = load_obj('../embedding/embedding_diff/advertiser_id_test_dim128_mincount1')
    
    product_id_test_= load_obj('../embedding/embedding_diff/product_id_test_dim100_mincount1')  
    
    industry_id_test_ = load_obj('../embedding/embedding_diff/industry_id_test_dim50_mincount1')
    
    return ad_id_test_,creative_id_test_,advertiser_id_test_,product_id_test_,industry_id_test_

def w2v_matrix():
    print('load from pkl...')
    ad_id_embedding_word2vec_matrix= load_obj('../embedding/embedding_200/ad_id_embedding_word2vec_matrix')
    
    creative_id_embedding_word2vec_matrix= load_obj('../embedding/embedding_200/creative_id_embedding_word2vec_matrix')
    
    advertiser_id_embedding_word2vec_matrix= load_obj('../embedding/embedding_diff/advertiser_id_embedding_word2vec_matrix_dim128_mincount1')
    
    product_id_embedding_word2vec_matrix = load_obj('../embedding/embedding_diff/product_id_embedding_word2vec_matrix_dim100_mincount1')    
    industry_id_embedding_word2vec_matrix = load_obj('../embedding/embedding_diff/industry_id_embedding_word2vec_matrix_dim50_mincount1')    

    
    return ad_id_embedding_word2vec_matrix,creative_id_embedding_word2vec_matrix,advertiser_id_embedding_word2vec_matrix,product_id_embedding_word2vec_matrix,industry_id_embedding_word2vec_matrix
  


    
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    pass