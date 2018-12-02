# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 22:57:58 2017

@author: aisha

2 cases
stuff stuck from the front
C,P, ignore it and takes the word from them and the tag of next word
if next word is an S, do not ignore it
stuck from the back 
S  ignoew it ,
another case if 2 special cases foloow each other
cae1 2 2 conj or 1 conj and adp or prt,ignore the seconed one
"""
import re
import os
list_sentences=[]
sentennce =[]
sent_no=0
next_index=0
saved_word=""
saved_tag=""
suffix=['هن','ها','هما','ه','ما','كم','ني','هم','نا','ك','ا','ي']
prefix=['ال','ما','']
pron=['ني', 'كيف', 'ه', 'هن', 'هذان', 'من', 'هي', 'نحن', 'ا', 'ذا', 'ماهو', 'أين', 'ها', 'أولئك', 'أية', 'هٰؤلاء', 'ذاك', 'الذين', 'نا', 'هما', 'أنا', 'هذه', 'هم', 'للاتي', 'التي', 'أيا', 'ذلك', 'هو', 'هٰذه', 'ذوي', 'الذي', 'هٰذا', 'متى', 'هذا', 'أي', 'اللذان', 'هاتان', 'هٰذين', 'اللواتي', 'اللتين', 'هؤلاء', 'كم', 'ك', 'ذي', 'ما', 'ذٰلك', 'ي', 'تلك']
adp=['تجاه', 'حوالى', 'فور', 'ب', 'مع', 'إزاء', 'وراء', 'سوا', 'لدي', 'من', 'داخل', 'أما', 'رغم', 'بشأن', 'وفق', 'حوالي', 'بلا', 'خارج', 'نحو', 'عبر', 'بإزاء', 'إثر', 'تحت', 'عوض', 'قيد', 'حيال', 'بسبب', 'سوى', 'ضد', 'جراء', 'عقب', 'مثل', 'خلف', 'لدى', 'بعيد', 'بدون', 'قبل', 'أمام', 'خلال', 'طوال', 'ضمن', 'منمن', 'علي', 'دون', 'ل', 'حين', 'وسط', 'بين', 'إلى', 'عدا', 'قرب', 'شرق', 'فوق', 'قبالة', 'في', 'ك', 'عند', 'إلي', 'بعد', 'على', 'منذ', 'حتى', 'حول', 'عن', 'أثناء']
conj=['حيث', 'مما', 'أن', 'كأنما', 'مهما', 'لكي', 'لكن', 'هكذا', 'فيما', 'طال', 'مالم', 'لعل', 'عندما', 'لأن', 'حتى', 'حينما', 'ولو', 'ف', 'لذٰلك', 'إنما', 'و', 'لٰكن', 'إذا', 'لو', 'إن', 'لولا', 'ألا', 'أنلا', 'بل', 'كذا', 'حسبما', 'مثل', 'إذ', 'لما', 'طالما', 'كي', 'أو', 'متى', 'كما', 'حسب', 'أي', 'حين', 'بينما', 'أينما', 'أم', 'إما', 'عل', 'كل', 'كلما', 'عند', 'إلا', 'منذ', 'لذلك', 'مثلما', 'لذا', 'ل']
prt=['إيا', 'أن', 'ماذا', 'فور', 'أ', 'ناهيك', 'من', 'س', 'هلا', 'لماذا', 'إن', 'هل', 'عما', 'لا', 'أيا', 'سوى', 'لقد', 'قد', 'لما', 'كلا', 'شبعا', 'لئن', 'سوف', 'أي', 'لن', 'عل', 'إلا', 'ما', 'لم', 'غير', 'ل']


                      
def deNoise(text):
    noise = re.compile("""  ّ    | # Tashdid
                            َ    | # Fatha
                            ً    | # Tanwin Fath
                            ُ    | # Damma
                            ٌ    | # Tanwin Damm
                            ِ    | # Kasra
                            ٍ    | # Tanwin Kasr
                            ْ    | # Sukun
                            \u0670|
                              ـ     # Tatwil/Kashida
                                """, re.VERBOSE)
    text = re.sub(noise, '', text)
    return text       


                 
    
def remove_index(sentennce):
    sent=[]
    for (tag,index,word) in sentennce:
        sent.append((tag,word))
    return sent



def map_tag(d_tag):
    if d_tag.startswith("S"):
         return "PRON"
    elif d_tag.startswith("A"):
          return "ADJ"
    elif d_tag.startswith("P"):
          return "ADP"
    elif d_tag.startswith("D"):
          return "ADV"
    elif d_tag.startswith("C"):
          return "CONJ"
    elif d_tag.startswith("N") or d_tag.startswith("Z"):
         return "NOUN"
    elif d_tag.startswith("Q"):
         return "NUM"
    elif d_tag.startswith("V"):
         return "VERB"
    elif d_tag.startswith("G"):
         return "PUNCT"
    elif d_tag.startswith("F"):
         return "PRT"
    else:
        return "X"

paths=['PADT_LDC/data/ALH/syntax/','PADT_LDC/data/UMH/syntax/','PADT_LDC/data/XIN/syntax/','PADT_LDC/data/XIA/syntax/','PADT_LDC/data/ANN/syntax/','PADT_LDC/data/AFP/syntax/']

for path in paths:
    files = (file for file in os.listdir(os.path.join(os.path.realpath('..'),path)))
    for file in files:
        if os.path.isfile(os.path.join(os.path.realpath('..'),path, file)):
            with open(os.path.join(os.path.realpath('..'),path)+file,
            encoding='utf-8') as f:
                for line in f:
                    splitcomma=re.split(',',line)
                    if(len(splitcomma)>4):
                        if(splitcomma[0].startswith('[#')):
                            if(sent_no !=0 ):
                                sentennce.sort(key=lambda tup: tup[1])
                                sortedd=remove_index(sentennce)
                                list_sentences.append(sortedd)
                                sentennce =[]
                            sent_no=splitcomma[0][2]
                        else:
                            for inn in splitcomma :
                                if(inn.startswith('ord=')):
                                    index=inn[4:]
                            if(('PADT_LDC/data/UMH/syntax/') in path or ('PADT_LDC/data/XIN/syntax/' in path)):
                                tag=map_tag(splitcomma[2][4:])
                                sword=splitcomma[3]
                            else:
                                tag=map_tag(splitcomma[3])
                                sword=splitcomma[4]
                            word=deNoise(splitcomma[0][1:])
                            tag_word=(tag,int(index),word)
                            sentennce.append(tag_word)
                   
                   
        
        
                            
                   
                 
                    
                   
                   
sentennce.sort(key=lambda tup: tup[1])
sortedd=remove_index(sentennce)
list_sentences.append(sortedd)
print(list_sentences[5])
print(len(list_sentences))


     
def new_prefix(sent):
    for idx,(tag,word) in enumerate(sent):
        
        if((tag=="PRT" or tag=="ADP" or tag=="CONJ") and len(word)==1) and idx<len(sent)-2:
            (t,w)=sent[idx+1]
            if(w.startswith("ال")and word=='ل'):
                sent[idx+1]=(t,word+w[1:])
                del sent[idx]
            else:
                sent[idx+1]=(t,word+w)
                del sent[idx]
             
        if(tag=="PRON" and len(word)==1) and idx>1:
            (t,w)=sent[idx-1]
            if(w.endswith("ة")):
                sent[idx-1]=(t,w[:-1]+"ت"+word)
                del sent[idx]
            elif (w.endswith("ى")):
                sent[idx-1]=(t,w[:-1]+"ي"+word)
                del sent[idx]
            else: 
                sent[idx-1]=(t,w+word)
                del sent[idx]
              
    return sent
            

def clean(tagged):
    for sent in tagged:
        if(len(sent)==0):
            tagged.remove(sent)
    return tagged

def tagged_sents():
    tagged_sents=[]
    tokB1=0

    tokB=0
    tokA=0
    for sent in list_sentences:
        tokB1=tokB1+len(sent)

        fixed_sent=new_prefix(sent)
        tokB=tokB+len(fixed_sent)
        fixed_sent = [(t,w) for (t,w) in sent if (t!="X" and t!="PUNCT" and t!="NUM")]
        tokA=tokA+len(fixed_sent)
        tagged_sents.append(fixed_sent)  
    print("tokB",tokB,"tokA",tokA,"tokbb1",tokB1)
    return clean(tagged_sents)



