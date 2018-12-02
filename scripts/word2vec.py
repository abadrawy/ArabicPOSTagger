# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 19:49:39 2017

@author: aisha
"""
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gzip
import gensim
import glob
from time import time
import multiprocessing
import re
import os
assert gensim.models.word2vec.FAST_VERSION >-1
cores = multiprocessing.cpu_count()




#fname='D:\LDC2009T30-Arabic_Gigaword_Fourth_Edition\dist\data\\aaw_arb\\aaw_arb_200611.gz'
#with open('word_list', 'wb') as fp:
     #pickle.dump(sentences, fp)
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in glob.glob(self.dirname+'/*/*'): 
            with gzip.open(fname, mode='rt',encoding='utf-8') as f:
                for line in f:
                    l= re.sub('[a-zA-Z]+', ' ', line)
                    li = re.sub('[_=/@&;!.،:"؟><,»«)(*-]', '', l)
                    sent=li.strip().split(" ")
                    if len(sent)!=1:
                        yield sent
            print("fname",fname)
 

                
        

t0 = time()
print("start")
print("testing",len(glob.glob(os.path.realpath('..')+'/LDC2009T30-Arabic_Gigaword_Fourth_Edition/dist/data'+'/*/*')))

sentences = MySentences(os.path.realpath('..')+'/LDC2009T30-Arabic_Gigaword_Fourth_Edition/dist/data') 
model = gensim.models.Word2Vec(sentences,workers=cores,size=1000,min_count=10)
print ("Training time:", round(time()-t0, 3), "s")
model.save(os.path.realpath('..')+'/word2vecModels/word2vec_1000')
