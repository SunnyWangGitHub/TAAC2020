
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import gc
from keras.utils import to_categorical
import time
from keras.callbacks import *
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
sys.path.append('../')
sys.path.append('../../')
from data_process import w2v_train,w2v_test,w2v_matrix
from model.model_7_12 import *
import multiprocessing
# import traceback
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



from keras.callbacks import *
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_acc', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)



word_seq_len = 190

batch_size = 512
lr_param = 0.002
weight_decay_param = 0.0001
factor = 0.8
classification = 10
seed = 2020
num_folds = 5
epochs = 40
my_opt="model_trans_v1"


def test_worker(i,my_opt):
        print('start test worker...')
        finder_path = "predict/"+str(i)
        model_path = "model/"+str(i)

        if not os.path.exists(finder_path):
            os.mkdir(finder_path)  

        if not os.path.exists(model_path):
            os.mkdir(model_path)

        my_opt=eval(my_opt)
        name = str(my_opt.__name__)

        ad_id_embedding_word2vec_matrix,advertiser_id_embedding_word2vec_matrix,product_id_embedding_word2vec_matrix = w2v_matrix()
        model = my_opt(word_seq_len,ad_id_embedding_word2vec_matrix,advertiser_id_embedding_word2vec_matrix,product_id_embedding_word2vec_matrix,classification)    
        
        del ad_id_embedding_word2vec_matrix
        del advertiser_id_embedding_word2vec_matrix
        del product_id_embedding_word2vec_matrix
        gc.collect()



        from keras.utils import multi_gpu_model 
        parallel_model = multi_gpu_model(model, 4)
        parallel_model.compile(optimizer=RAdam(lr=lr_param,weight_decay=weight_decay_param,), loss='categorical_crossentropy',
              loss_weights=[1., 0.5],metrics=['accuracy'])

        best_model_path = model_path + '/best_model.h5'


        checkpoint = ParallelModelCheckpoint(model, filepath=best_model_path, monitor='val_output_age_acc', verbose=1, save_best_only=True,save_weights_only='False') 
        
        cb = [  EarlyStopping(monitor='val_output_age_acc', patience=4, verbose=0),
                ReduceLROnPlateau(monitor='val_output_age_acc', factor=0.5, patience=2, min_lr=0.0001, verbose=1),
                checkpoint]


        # parallel_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid),callbacks = cb,shuffle=True)   


        # del X_train
        # del y_train
        # del X_valid
        # del y_valid
        # del parallel_model
        # K.clear_session()
        # tf.reset_default_graph()
        gc.collect()


        print('Start predicting...')
        print('load best model...')


        # parallel_model.load_weights(best_model_path)
        model.load_weights(best_model_path)



        test_data = pd.read_csv('../data_process/test_all_data_6_17.csv')
        test = pd.DataFrame(test_data)
#         test_tfidf = pd.read_csv('../data_process/all_old_all_new_test_data_ad_ids_tfidf.csv')
#         test_tfidf = pd.DataFrame(test_tfidf)
#         test_tf_idf = test_tfidf.drop(['user_id'],axis=1)
        countvec_test = pd.read_csv('../data_process/aggregate_features_df.csv')
        countvec_test = countvec_test.drop(['user_id'],axis=1)
        
        X_scaler = StandardScaler()
        countvec_test = X_scaler.fit_transform(countvec_test)
        ad_id_test_,advertiser_id_test_,product_id_test_ = w2v_test()

        print('predicting....')
#         test_model_pred =  model.predict([ad_id_test_,advertiser_id_test_,product_id_test_,test_tf_idf,countvec_test])
        test_model_pred =  model.predict([ad_id_test_,advertiser_id_test_,product_id_test_,countvec_test])
        test_model_age_pred = np.argmax(test_model_pred[0], axis=1) + 1
        test_model_gender_pred = np.argmax(test_model_pred[1], axis=1) + 1

        test['predicted_age'] = test_model_age_pred
        test['predicted_gender'] = test_model_gender_pred
        res = test[['user_id','predicted_age','predicted_gender']]
        

    
        res.to_csv(finder_path+'/submission.csv',index=0)
        print('success! save to '+ finder_path+'/submission.csv')

        del parallel_model
        del model
        del test
#         del test_tfidf
        del countvec_test
        del ad_id_test_
        del advertiser_id_test_
        del product_id_test_
        gc.collect()
        K.clear_session()
        tf.reset_default_graph()
        print('test... fold ' +str(i)+' finish!')
        print('####################################################')



def training_worker(i,my_opt,X_train,X_valid,y_train,y_valid):
        print('start train worker...')
        finder_path = "predict/"+str(i)
        model_path = "model/"+str(i)

        if not os.path.exists(finder_path):
            os.mkdir(finder_path)  

        if not os.path.exists(model_path):
            os.mkdir(model_path)

        my_opt=eval(my_opt)
        name = str(my_opt.__name__)

        ad_id_embedding_word2vec_matrix,advertiser_id_embedding_word2vec_matrix,product_id_embedding_word2vec_matrix = w2v_matrix()
        model = my_opt(word_seq_len,ad_id_embedding_word2vec_matrix,advertiser_id_embedding_word2vec_matrix,product_id_embedding_word2vec_matrix,classification)    
        
        del ad_id_embedding_word2vec_matrix
        del advertiser_id_embedding_word2vec_matrix
        del product_id_embedding_word2vec_matrix
        gc.collect()



        from keras.utils import multi_gpu_model 
        parallel_model = multi_gpu_model(model, 4)
        parallel_model.compile(optimizer=RAdam(lr=lr_param,weight_decay=weight_decay_param,), loss='categorical_crossentropy',
              loss_weights=[1., 0.5],metrics=['accuracy'])

        best_model_path = model_path + '/best_model.h5'


        checkpoint = ParallelModelCheckpoint(model, filepath=best_model_path, monitor='val_output_age_acc', verbose=1, save_best_only=True,save_weights_only='False') 
        
        cb = [  EarlyStopping(monitor='val_output_age_acc', patience=4,verbose=0),
                ReduceLROnPlateau(monitor='val_output_age_acc', factor=0.5, patience=2, min_lr=0.0001, verbose=1),
                checkpoint]


        parallel_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid),callbacks = cb,shuffle=True)   


        del X_train
        del y_train
        del X_valid
        del y_valid
        del parallel_model
        # K.clear_session()
        # tf.reset_default_graph()
        gc.collect()

        gc.collect()
        K.clear_session()
        tf.reset_default_graph()
        print('train...fold ' +str(i)+' finish!')
        print('####################################################')







if __name__ == '__main__':
    print('prepare dataset...')


    train_data = pd.read_csv('../data_process/all_old_all_new_train_data.csv')
    train = pd.DataFrame(train_data)
    del train_data
#     train_tfidf_csv = pd.read_csv('../data_process/all_old_all_new_train_data_ad_ids_tfidf.csv')
#     train_tfidf = pd.DataFrame(train_tfidf_csv)
#     train_tf_idf = train_tfidf.drop(['user_id'],axis=1)
#     del train_tfidf_csv


    countvec_train = pd.read_csv('../data_process/all_old_all_new_train_data_aggregate_features_df.csv')
    countvec_train = countvec_train.drop(['user_id'],axis=1)
    X_scaler = StandardScaler()
    countvec_train = X_scaler.fit_transform(countvec_train)


    print('finish prepare dataset...')

    # print([column for column in train])


  
    


    train_label = train[['age','gender']]
    train_label['age'] = train_label['age'].apply(lambda x : x - 1)
    train_label['gender'] = train_label['gender'].apply(lambda x : x - 1)
    train_label = train_label.values




    ad_id_train_,advertiser_id_train_,product_id_train_ = w2v_train()

#     train_data = np.hstack((ad_id_train_,advertiser_id_train_,product_id_train_,train_tf_idf,countvec_train))
    train_data = np.hstack((ad_id_train_,advertiser_id_train_,product_id_train_,countvec_train))

    gc.collect()

    splits = list(KFold(n_splits=num_folds, shuffle=True, random_state=seed).split(train_data, train_label))

    for i, (train_fold, val_fold) in enumerate(splits):
        print('#######################################################')
        print('KFold '+str(i)+'.....')
        gc.collect()

        X_train, X_valid, = train_data[train_fold, :], train_data[val_fold, :]
        y_train, y_valid = train_label[train_fold], train_label[val_fold]

#         X_train = np.split(X_train,[190,190*2,190*3,190*4],axis = 1)
#         X_valid = np.split(X_valid,[190,190*2,190*3,190*4],axis = 1)

        X_train = np.split(X_train,[190,190*2,190*3],axis = 1)
        X_valid = np.split(X_valid,[190,190*2,190*3],axis = 1)
        
        y_train = np.split(y_train,2,axis = 1)
        y_valid = np.split(y_valid,2,axis = 1)
        y_train = [to_categorical(y_train[0]),to_categorical(y_train[1])]
        y_valid = [to_categorical(y_valid[0]),to_categorical(y_valid[1])]



        training_process = multiprocessing.Process(target=training_worker, args = (i,my_opt,X_train,X_valid,y_train,y_valid))
        training_process.start()
        training_process.join()
        test_process = multiprocessing.Process(target=test_worker, args = (i,my_opt))
        test_process.start()
        test_process.join()
        print('########################################')
        print('finish!!!')








































