#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 00:40:13 2017

@author: badrawy
"""
import numpy as np
import feature_extract as f
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from sklearn.metrics import accuracy_score

import to_word2vec as w
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)



def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0]/batch_size)

    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        #X_batch = X[batch_index,:]
       
        y_batch = y[batch_index]  #52083 9370 59700
        counter += 1       
        print("len",X_batch.shape[0])
        yield np.reshape(X_batch,(batch_size,1,f)), y_batch
        #yield X_batch,y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0
            


"""
try random bactch gernator
use a fixed number of steps per epoch 
"""
def kerasMLP():
    featuers_train,feauters_test,labels_train,labels_test,feature_names =f.split_data(False)
    #featuers_train,feauters_test,feature_names=f.featureSelection(featuers_train,feauters_test,labels_train)

    #tr=featuers_train.shape[0]
    #te=feauters_test.shape[0]
    #dims = len(feature_names)
    
    dims=2000
    #dims=52380
    
    le = LabelEncoder()
    transfoemed_labels_train = le.fit_transform(labels_train)
    hot_labels_train = np_utils.to_categorical(transfoemed_labels_train, 8)
    transformed_labels_test=le.fit_transform(labels_test)
    hot_labels_test = np_utils.to_categorical(transformed_labels_test, 8)
    
    
    
    #nerual  model
    model = Sequential()
    model.add(Dense(500, input_dim=dims, init="uniform",
    	activation="softmax")) #input layer
    model.add(Dense(64)) #hidden layerS
    model.add(Dense(32)) #hidden layerS

    model.add(Dense(8)) #output layer
    model.add(Activation("softmax"))
    
    
    #ftting model
    print("[INFO] compiling model...")
    model.compile(loss="categorical_crossentropy", optimizer="adam",
    	metrics=["accuracy"])
    
    """model.fit_generator(generator=batch_generator(featuers_train,hot_labels_train,10000,False),
                    epochs=30,
                   steps_per_epoch=tr/10000)"""
    
   
    model.fit(featuers_train, hot_labels_train, nb_epoch=30, batch_size=300)


    #testing 
    print("[INFO] evaluating on testing set...")
    
    #(loss, accuracy) =model.evaluate_generator(generator=batch_generator(feauters_test, hot_labels_test,1000,False),steps=te/1000)
    (loss, accuracy) =  model.evaluate(feauters_test, hot_labels_test,
    	batch_size=50, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
    	
    accuracy * 100))

def kerasRNN():
  
    featuers_train,feauters_test,labels_train,labels_test,f_names=f.split_data(False)
    #featuers_train,feauters_test,feature_names=f.featureSelection(featuers_train,feauters_test,labels_train)

    featuers_train=featuers_train[:30000]
    feauters_test=feauters_test[:8685]
    labels_train=labels_train[:30000]
    labels_test=labels_test[:8685]

    tr=featuers_train.shape[0]
    te=feauters_test.shape[0]
    #print("theree",tr,te)
    #featuers_train,feauters_test,f_names=f.victorize( featuers_train,feauters_test)
    dims = len(f_names)    #dims=500
    #dims=52380
    #dims=2000
    global f
    f=dims
    #print("theree",tr,te,dims)

    #featuers_train=np.reshape( featuers_train,(len(featuers_train),1,dims))
    #feauters_test=np.reshape( feauters_test,(len(feauters_test),1,dims))
    
   
    le = LabelEncoder()
    transfoemed_labels_train = le.fit_transform(labels_train)
    hot_labels_train = np_utils.to_categorical(transfoemed_labels_train,8)
    transformed_labels_test=le.fit_transform(labels_test)
    hot_labels_test = np_utils.to_categorical(transformed_labels_test, 8)
    
    
    
    
    #model
    model = Sequential()
    model.add(LSTM(300, return_sequences=True,
               input_shape=(None,dims)))  # returns a sequence of vectors of dimension 32

    model.add(Bidirectional(LSTM(64,return_sequences=True)))  # returns a sequence of vectors of dimension 32
    model.add(Bidirectional(LSTM(32))) # return a single vector of dimension 32
    model.add(Dense(8, activation='softmax'))
    
    #ftting model
    print("[INFO] compiling model...")
    #sgd = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer='RMSProp',
    	metrics=["accuracy"])

   
    model.fit_generator(generator=batch_generator(featuers_train,hot_labels_train,5000,False),
                    epochs=30,
                   steps_per_epoch=tr/5000)   
    #model.fit(featuers_train, hot_labels_train, nb_epoch=30, batch_size=300)
    #testing 
    print("[INFO] evaluating on testing set...")

    (loss, accuracy) =model.evaluate_generator(generator=batch_generator(feauters_test, hot_labels_test,1737,False),steps=te/1737)
    #(loss, accuracy) =  model.evaluate(feauters_test, hot_labels_test,
    	#batch_size=50, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
    	
    accuracy * 100))

def kerasMLPV():
    transformed_featuers_train,labels_trainn,transformed_featuers_test,labels_testn=w.prepare_data()
    transformed_featuers_train=np.array(transformed_featuers_train)
    transformed_featuers_test=np.array(transformed_featuers_test)
    
    #dims = len(feature_names)
    
    dims=300
    le = LabelEncoder()
    transfoemed_labels_train = le.fit_transform(labels_trainn)
    hot_labels_train = np_utils.to_categorical(transfoemed_labels_train,8)
    transformed_labels_test=le.fit_transform(labels_testn)
    hot_labels_test = np_utils.to_categorical(transformed_labels_test, 8)
    
    #nerual  model
    model = Sequential()
    model.add(Dense(dims, input_dim=dims, init="uniform",
    	activation="softmax")) #input layer
    #model.add(Dense(128)) #hidden layerS
    model.add(Dense(64,use_bias=True)) #hidden layerS
    #model.add(Dense(32)) #hidden layerS
    #model.add(Dense(16)) #hidden layerS
    model.add(Dense(8)) #output layer
    model.add(Activation("softmax"))
    
    #ftting model
    print("[INFO] compiling model...")
    #sgd = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer='adam',
    	metrics=["accuracy"])
    model.fit(transformed_featuers_train, hot_labels_train, nb_epoch=30, batch_size=300,shuffle=False)
    
    #testing 
    print("[INFO] evaluating on testing set...")
    (loss, accuracy) =  model.evaluate(transformed_featuers_test, hot_labels_test,
    	batch_size=50, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
    	
    accuracy * 100))
    
  


    
"""
for train_f ,train_l in  zip(transformed_featuers_train,labels_trainn):
       hot_labels_train=[]
       train_re= np.reshape( train_f,(1,len(train_f),dims))
       temp=le.fit_transform(train_l)
       hot_labels_train=hot_labels_train.append(np_utils.to_categorical(temp,8))
       #model.train_on_batch(train_re,hot_labels_train)
       model.fit(train_re,hot_labels_train, epochs=1, batch_size=1,shuffle=False)
       #model.reset_states()
    #test the model
    print("[INFO] evaluating on testing set...")
    #(loss, accuracy) = model.evaluate(test_re, hot_labels_test,
    	#batch_size=1, verbose=1)
    #model.reset_states()
    for test_f ,test_l in  zip(transformed_featuers_test,labels_testn):
       hot_labels_test=[]
       test_re= np.reshape( test_f,(1,len(test_f),dims))
       temp=le.fit_transform(test_l)
       hot_labels_test=hot_labels_test.append(np_utils.to_categorical(temp,8))
       (loss, accuracy)=model.evaluate(test_re, hot_labels_test,
    	batch_size=1, verbose=1)
"""



def kerasRNN_batch():  
   
    featuers_train,labels_train,feauters_test,labels_test=w.prepare_data()
    featuers_train=np.array(featuers_train)
    feauters_test=np.array(feauters_test)
   
    """
    featuers_train,feauters_test,labels_train,labels_test =f.split_data(False)
    featuers_train,feauters_test,feature_names=f.victorize( featuers_train,feauters_test)
    """
    #dims=300
    dims=300
    #time_steps=265
    #batch=1
    """
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,stateful=True,
               batch_input_shape=(batch,None,dims)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, stateful=True,return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(dims))  # return a single vector of dimension 32
    model.add(Dense(8, activation='softmax'))
    """
    model = Sequential()
    model.add(LSTM(dims, return_sequences=True,
               input_shape=(None,dims))) # returns a sequence of vectors of dimension 32
    model.add(Bidirectional(LSTM(64,return_sequences=True)))  # returns a sequence of vectors of dimension 32
    model.add(Bidirectional(LSTM(32,return_sequences=True))) # return a single vector of dimension 32
    model.add(TimeDistributed(Dense(8,activation='softmax')))

             #model.add(Dense(8, activation='softmax'))
   
    #fit the model
    print("fiting")
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    
    le = LabelEncoder()
    count=0
    train_data=[]
    train_labels=[]
    for sent_tr,label_tr in zip(featuers_train,labels_train):
        hot_labels_train=[]
        if(len(sent_tr)!=0):
            train_re= np.reshape( sent_tr,(1,len(sent_tr),dims))
            temp=le.fit_transform(label_tr)
            hot_labels_train=np_utils.to_categorical(temp,8)
            hot_labels_train=np.reshape(hot_labels_train,(1,len(hot_labels_train),8))
            #model.train_on_batch(train_re,hot_labels_train)
            train_data.append(train_re)
            train_labels.append(hot_labels_train)
            else:
                count+=1
    
    
    model.fit_generator(one_batch_generator(train_data,train_labels),steps_per_epoch=len(train_data), epochs=30, verbose=1)
    
    #test the model
    print("[INFO] evaluating on testing set...")
    
    test_data=[]
    test_labels=[]
    for sent_te,label_te in zip(feauters_test,labels_test):
        hot_labels_test=[]
        if(len(sent_te)!=0):
            test_re= np.reshape( sent_te,(1,len(sent_te),dims))
            temp=le.fit_transform(label_te)
            hot_labels_test=np_utils.to_categorical(temp,8)
            hot_labels_test=np.reshape(hot_labels_test,(1,len(hot_labels_test),8))
            test_data.append(test_re)
            test_labels.append(hot_labels_test)
          
          #(loss, accuracy) = model.test_on_batch(test_re, hot_labels_test)

    (loss, accuracy) = model.evaluate_generator(one_batch_generator(test_data,test_labels), steps=len(test_data), max_q_size=10, workers=1)

    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
    	
    accuracy * 100))
    
    
def one_batch_generator(X, y):
    
    while True:
        for counter in range(len(X)):
            X_batch = X[counter]
            y_batch = y[counter]
          
            yield X_batch,y_batch


def kerasRNNV():
    dims=300
    #dims=500
    """
    featuers_train,feauters_test,labels_train,labels_test =f.split_data(False)
    featuers_train,feauters_test,feature_names=f.victorize( featuers_train,feauters_test)
   """
    featuers_train,labels_train,feauters_test,labels_test=w.prepare_data()
    
    """
    transformed_featuers_train=[item for sublist in transformed_featuers_train for item in sublist]    
    labels_trainn=[item for sublist in labels_trainn for item in sublist]    
    transformed_featuers_test=[item for sublist in transformed_featuers_test for item in sublist]    
    labels_testn=[item for sublist in labels_testn for item in sublist]    
    """
    featuers_train=np.reshape( featuers_train,(len(featuers_train),1,dims))
    feauters_test=np.reshape( feauters_test,(len(feauters_test),1,dims))
    
    le = LabelEncoder()
    transfoemed_labels_train = le.fit_transform(labels_train)
    hot_labels_train = np_utils.to_categorical(transfoemed_labels_train,8)
    transformed_labels_test=le.fit_transform(labels_test)
    hot_labels_test = np_utils.to_categorical(transformed_labels_test, 8)
    
    
    
    
    #model
    model = Sequential()
    model.add(LSTM(dims, return_sequences=True,
               input_shape=(None,dims)))  # returns a sequence of vectors of dimension 32
    model.add(Bidirectional(LSTM(64,return_sequences=True)))  # returns a sequence of vectors of dimension 32
    model.add(Bidirectional(LSTM(32))) # return a single vector of dimension 32
    model.add(Dense(8, activation='softmax'))
    
    #ftting model
    print("[INFO] compiling model...")
    #sgd = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer='adam',
    	metrics=["accuracy"])
    model.fit(featuers_train, hot_labels_train, nb_epoch=30, batch_size=1000,shuffle=False)
    
    #testing 
    print("[INFO] evaluating on testing set...")
    (loss, accuracy) =  model.evaluate(feauters_test, hot_labels_test,
    	batch_size=300, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
    	
    accuracy * 100))
            
#kerasMLP()
#kerasMLPV()
kerasRNN()
#kerasRNN_batch()
