# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 19:30:39 2017

@author: aisha
"""
import os
import random
from time import time
import gensim
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings("ignore",category=DeprecationWarning)
import PADTparsing as p
from gensim.models.keyedvectors import KeyedVectors
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler



unfound_words=[]
found_words=[]
sent_len=[]
model = gensim.models.Word2Vec.load(os.path.realpath('..')+'/word2vecModels/updated_model')
#model=gensim.models.Word2Vec.load(os.path.realpath('..')+'/word2vecModels/word2vec_1000')
#print("vocacb",len(model.wv.vocab))



def prepare_data():
    tagged_data=p.tagged_sents()
    random.seed(5)
    random.shuffle(tagged_data)
    
    sentences,labels=vec_data(tagged_data)  
    """if(flat):
     sentences,labels=flat_sents,flat_labels
    if(norm):
     sentences=norm_vec_data(sentences)"""

    #sentences,labels=pad(sentences,labels)
    
    #sentences,labels=clean(sentences,labels) 
 
    split_factor=int(.75*len(sentences))
    sentences_train=sentences[:split_factor]
    labels_train=labels[:split_factor]
    sentences_test=sentences[split_factor:]
    labels_test=labels[split_factor:]
    return  sentences_train,labels_train,sentences_test,labels_test

def vec_data(tagged_data):
    sentences=[]
    labels=[]
    tagged_data=out_vocab(tagged_data)
    #re_train(tagged_data)
    model = gensim.models.Word2Vec.load(os.path.realpath('..')+'/word2vecModels/updated_model')
    #model=gensim.models.Word2Vec.load(os.path.realpath('..')+'/word2vecModels/word2vec_1000')
    print("new  vocacb",len(model.wv.vocab))

    for sent in tagged_data:
        sentence=[]
        label=[]
        for idx,(tag,word) in enumerate(sent):
            #if word  in model.wv.vocab:
            word_vec=model[word]
            sentences.append(word_vec)
            labels.append(tag)
           
            
        #if(len(sentence)!=0): 
        sent_len.append(len(sentence))   
        #sentences.append(normalize(sentence, norm='max'))
        #labels.append(label)
    #sentences=standard_scaler.fit_transform(sentences)
    sentences=normalize(sentences, norm='max')


    return sentences,labels

def norm_vec_data(sentences):
    n_sentences=normalize(sentences, norm='max')
    return n_sentences

def re_train(tagged_data):
    more_sent=[]
    train_sent=[]
    for sent in tagged_data:
        for (tag,word) in sent:
            if(word=="OOV"):
                more_sent.append(sent)
                break
    for sent in more_sent:
         train_sent.append([t[1] for t in sent])
    print(len(train_sent))
    print(train_sent[1])
    
    #model.build_vocab(train_sent, update=False)

    model.train(train_sent)

    model.build_vocab(train_sent, update=True)
    model.train(train_sent)
   
    model.save(os.path.realpath('..')+'/word2vecModels/updated_model')

def out_vocab(tagged_data):
    for sent in tagged_data:
        for idx,(tag,word) in enumerate(sent):
             if word not in model.wv.vocab:
                    sent[idx]=(tag,"OOV")
    return tagged_data     

def zero_vec():
    vec=[]
    for i in range(300):
        vec.append(2147483647)
    return vec

def word(w):
    
     return  normalize(model[w], norm='max').reshape(300)

       
def pad(sentences,labels):
    z_vec=zero_vec()
    for idx,sent in enumerate(sentences):
        for i in range(265-len(sent)):
            sent.append(z_vec)
            labels[idx].append("pad")
    return sentences,labels

def clean(sentences,labels):
    for idx,sent in enumerate(sentences):
         if(len(sent)==0):
                count=count+1
                del sentences[idx]
                del labels[idx]
    return sentences,labels
    
    
"""          
t0 = time()
print("start")   
sentences_train,labels_train,sentences_test,labels_test=prepare_data()
print ("Training time:", round(time()-t0, 3), "s")
print("len train",len(sentences_train),"len test",len(sentences_test))  
print("not found",len(unfound_words))
print("found",len(found_words))
print("sent_len",len(set(sent_len)))
print("max of list",max(sent_len))
#print("unfound words",unfound_words)   

"""
