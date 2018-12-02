# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 01:49:32 2017

@author: aisha
"""

 
from time import time
import numpy as np
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import  mutual_info_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import accuracy_score
import to_word2vec as w
import feature_extract as f

def sklearnClassification(clf,featureSelection,hybrid):
    """
    Ishould victoize then split 
    """
    
    featuers_train,feauters_test,labels_train,labels_test,f_names =f.split_data(hybrid)
    
    print("featuer train shape",featuers_train.shape[1])
    
    if(featureSelection):
        features_train,feauters_test,_=f.featureSelection(featuers_train,feauters_test,labels_train)
        
    t0 = time()
    print("start fit")
    clf.fit(featuers_train,labels_train)
    print ("Training time:", round(time()-t0, 3), "s")
    print("start pred")
    pred=clf.predict(feauters_test)
    print(accuracy_score(labels_test, pred))


def sklearnClassification():
    """
    i should victoize then split 
    """
    
    featuers_train,feauters_test,labels_train,labels_test,f_names =f.split_data(True)
    
    print("featuer train shape",featuers_train.shape[1])
    
    #featuers_train=np.reshape(featuers_train,(len(featuers_train),300))
    #feauters_test=np.reshape(feauters_test,(len(feauters_test),300))
    
    
    #new_transformed_featuers_train,new_transformed_featuers_test,_=f.featureSelection(featuers_train,feauters_test,labels_train)
  
    #sentences_train,labels_trainn,sentences_test,labels_testn=w.prepare_data()
  
    
    #clf = svm.SVC(kernel="poly",C=1,gamma=1000)
    #print("vec size",len(fn))
    #clf = svm.LinearSVC(C=1)
    #clf = tree.DecisionTreeClassifier(criterion="entropy")
    #clf=RandomForestClassifier(criterion="entropy",max_features=None,n_estimators=50,min_samples_leaf=3,n_jobs=4)
    #clf=AdaBoostClassifier()
    #clf = GaussianNB()
    clf=BernoulliNB()
    t0 = time()
    print("start fit")
    clf.fit(featuers_train,labels_train)
    print ("Training time:", round(time()-t0, 3), "s")
    print("start pred")
    pred=clf.predict(feauters_test)
    print(accuracy_score(labels_test, pred))
    
def SKparam_tunning(k):
    featuers_train,feauters_test,labels_train,labels_test =f.split_data()
    featuers_train,feauters_test,_=f.victorize( featuers_train,feauters_test)
    Cs = [0.001, 0.01, 0.1, 1, 10,100,1000,10000]
    gammas = [0.001, 0.01, 0.1, 1,10,100,1000,10000]
    param_grid = {'C': Cs,'gamma':gammas}
    grid_search =GridSearchCV(svm.SVC(kernel=k), param_grid)
    grid_search.fit(featuers_train,labels_train)
    print("best",grid_search.best_params_)
    
    


    
    
sklearnClassification()

    

