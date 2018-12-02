# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:46:25 2017

@author: aisha
"""

import random
import PADTparsing as p
import to_word2vec as wv
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_extraction import DictVectorizer
#from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import normalize


def features(sentence,index,history,hybrid):
  
       
   
        """
        dic=[]
        dic.extend(wv.word(sentence[index]))
        dic.append(len(sentence[index]))
        dic.extend(zero() if index<=1 else word(sentence[index-2]))
        dic.extend(zero() if index==0 else word(sentence[index-1]))
        #print("history",history)
        dic.append(-1 if index==0 else history[index-1])
        dic.append(-1 if index<=1 else history[index-2])

        dic.extend(zero() if index==len(sentence)-1 else word(sentence[index+1]))
        dic.extend(zero() if index>=len(sentence)-2 else word(sentence[index+2]))

        """
       
        dic={
            'word':sentence[index],
            'word_length':len(sentence[index]),
            'is_first':index==0,
            'is_last':index==len(sentence)-1,
            'prefix-1': '' if len(sentence[index])==0 else sentence[index][0],
            'prefix-2':sentence[index][:2],
            'prefix-3':sentence[index][:3],
            'suffix-1':'' if len(sentence[index])==0 else sentence[index][-1],
            'suffix-2':sentence[index][-2:],
            'suffix-3':sentence[index][-3:],
            'p_prev_word':''if index<=1 else sentence[index-2],
            'prev_word':'' if index==0 else sentence[index-1],
            'prev_tag':'' if index==0 else history[index-1],
            'p_pev_tag':'' if index<=1 else history[index-2],                                  
            'next_word':'' if index==len(sentence)-1 else sentence[index+1],
            'n_next_word':'' if index>=len(sentence)-2 else sentence[index+2]
        }
        
        if(hybrid):
            
            word=wv.word(sentence[index])

            for i in range(300):
                dic.update({'wordv'+str(i):word[i]})

            if(index<=1):
                for i in range(300):
                    dic.update({'p_prev_wordv'+str(i):-1})
            else :
                p_prev_word=wv.word(sentence[index-2])
                for i in range(300):
                    dic.update({'p_prev_wordv'+str(i):p_prev_word[i]})

            if(index==0):
                for i in range(300):
                    dic.update({'prev_wordv'+str(i):-1})
            else :
                prev_word= wv.word(sentence[index-1])
                for i in range(300):
                    dic.update({'prev_wordv'+str(i):prev_word[i]})

            if(index==len(sentence)-1):
                for i in range(300):
                    dic.update({'next_wordv'+str(i):-1})
            else :
                next_word=wv.word(sentence[index+1])
                for i in range(300):
                    dic.update({'next_wordv'+str(i):next_word[i]})

            if(index>=len(sentence)-2):
                for i in range(300):
                    dic.update({'n_next_wordv'+str(i):-1})
            else :
                n_next_word=wv.word(sentence[index+2])
                for i in range(300):
                    dic.update({'n_next_wordv'+str(i):n_next_word[i]})
               
 
       
        
        
        return dic


def zero():
    vec=[]
    for i in range(300):
         vec.append(-1)
    return vec
def neg():
    vec=[]
    for i in range(8):
         vec.append(None)
    return vec
         
def untag(sent):
    return [w for (t,w) in sent]

def prepare_features_set(tagged_sentences,hybrid):
    global history
    history=[]
    features_set=[]
    labels_set=[]
    tagged_sentences=wv.out_vocab(tagged_sentences)
    for sent in tagged_sentences:
        for index in range(len(sent)):
            features_set.append(features(untag(sent),index,labels_set,hybrid))
            labels_set.append(sent[index][0])
            history.append(sent[index][0])
           
    return features_set,labels_set
  
def num_tags(d_tag):
        #print("fine",d_tag)
        if d_tag.startswith("PRON"):
              return 1;
        elif d_tag.startswith("ADJ"):
             return 2;
        elif d_tag.startswith("ADP"):
              return 3;
        elif d_tag.startswith("ADV"):
              return 4;
        elif d_tag.startswith("CONJ"):
              return 5;
        elif d_tag.startswith("NOUN"):
             return 6;
        elif d_tag.startswith("VERB"):
             return 7;
        else: 
            
             return 8;
            
    


def split_data(hybrid):
    tagged_sentences = p.tagged_sents()
    random.seed(5)

    random.shuffle(tagged_sentences)
    features_set,labels_set=prepare_features_set(tagged_sentences,hybrid)
    features_set,f_names=victorize(features_set)
    
    split_factor=int(.75*features_set.shape[0])
    featuers_train=features_set[:split_factor]
    labels_train=labels_set[:split_factor]
    feauters_test=features_set[split_factor:]
    labels_test=labels_set[split_factor:]
    
    
    

    return featuers_train,feauters_test,labels_train,labels_test,f_names 


def victorize(featuers):
    #standard_scaler = StandardScaler(with_mean=False)
    #svd = TruncatedSVD(n_components=500,random_state=0)
    #enc = OneHotEncoder()
    vec = DictVectorizer()
    featuers= vec.fit_transform(featuers)
    #train,test= standard_scaler.fit_transform(train),standard_scaler.fit_transform(test)
    #featuers=svd.fit_transform(featuers)
    #train,test=enc.fit_transform(train),enc.fit_transform(test)
    #train,test = normalize(train, norm='max'),normalize(test, norm='max')

    return featuers,vec.feature_names_


    


def featureSelection(featuers_train,feauters_test,labels_train):
    support =SelectPercentile(chi2,percentile=90).fit(featuers_train,labels_train)
    new_transformed_featuers_train=support.transform(featuers_train)
    selected_features=support.get_support()
    print(len(selected_features))
    new_transformed_featuers_test=support.transform(feauters_test)
    return new_transformed_featuers_train,new_transformed_featuers_test,selected_features


def visuializa_data():
    featuers_train,feauters_test,labels_train,labels_test =split_data()
    featuers_train,x_test,_=victorize(featuers_train,feauters_test)
    #class_labels = 8
    label_encoder = LabelEncoder()
    labels_train = label_encoder.fit_transform(labels_train)
    y_test=label_encoder.fit_transform(labels_test)
    svd = TruncatedSVD(n_components=2,algorithm="arpack")
    x_test_2d=svd.fit_transform(x_test)
    #x_test_2d=x_test_2d.toarray()
    # scatter plot the sample points among 5 classes
    markers=('s', 'd', 'o', '^', 'v','p','P','*')
    color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan',5:'yellow',6:'black',7:'magenta'}
    plt.figure()
    for idx, cl in enumerate(np.unique(y_test)):
        plt.scatter(x=x_test_2d[y_test==cl,0], y=x_test_2d[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
    plt.xlabel('X in t-SVD')
    plt.ylabel('Y in t-SVD')
    plt.legend(loc='upper left')
    plt.title('t-SVD visualization of test data')
    plt.show()
    

    

