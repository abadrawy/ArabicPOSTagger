# Arabic POS Tagger

POS Tagging is the problem of assigning tags, using a defined Tag set to un-tagged new words given a previously tagged Corpus. 
A tag set can be as general as (Noun, Verb, Adverb) and as detailed as (male or female, singular or plural, past or present).
    
    
## Classifiers Used In This Project

* Naïve Bayes 
* Support Vector Machines (SVM)
* Random Forest
* Neural Networks

## Libraries and Tools
The Arabic POS tagger is implemented in Python, using different python libraries .

* Scikit-Learn.
* Keras.
* Gensim.

## Motivation
* Research interest in processing Arabic language compared to English language is very limited .

* POS tagging  is an important NLP task that serve as a subtask for building larger NLP applications .

* The aim of this bachelor project is to implement a Supervised  Arabic POS Tagger, experimenting with different machine learning algorithms.

## English POS

A review of state of the art English POS taggers.

![alt text](https://github.com/abadrawy/ArabicPOSTagger/blob/master/graphs%26Images/EnglihsPOS.png)


# Arabic POS

A review of state of the art Arabic POS taggers.

![alt text](https://github.com/abadrawy/ArabicPOSTagger/blob/master/graphs%26Images/ArabicPOS.png)

## DataSet

* The data set used in implementing the system, is the Prague Arabic Dependency Treebank (PADT).

* It is based on Modern Standard Arabic (MSA), and was developed to be used in general NLP tasks.

* It is comprised of news texts and articles  from 6 sources.

* The PADT data set  consists of 113,000 tokens and has a tag set of 21 tags.

![alt text](https://github.com/abadrawy/ArabicPOSTagger/blob/master/graphs%26Images/pos.png)

## Data Processing

* The PADT tags were reduced to to the most common tags according universal tagset.

* The Universal Tagset consist of 12 universal POS categories that are common between all languages.

* PADT tags were reduced from 21 tags to 11 tags.


![alt text](https://github.com/abadrawy/ArabicPOSTagger/blob/master/graphs%26Images/tagset.png)

* Removing diacritics as some words had diacritics like subscript Alef.

* PADT is a morphologically analyzed data set, thus it treats the word shown as 2  separate words.Thus, words with such pattern were concatenated and the tag of the prefix or the suffix was removed

![alt text](https://github.com/abadrawy/ArabicPOSTagger/blob/master/graphs%26Images/dataProcessing.png)

* The last step is filtering tokens, by removing unknown words, punctuations and numerals.

* The final shape of the data is 8 tags and 42,632 tokens. 


## Feature Extraction

The process of Feature Extraction consists of first, constructing the features that are informative and distinctive.
Constructed features are:

* The Word itself, it’s Length, whether it’s first or last in the sentences.

* The first n and last character of the word with n ranging from 1 to 3.

* Previous word and it’s tag.

* The word Previous to the previous word and it’s tag.

* The Next word.

* The word next to the next word.

After constructing the features, features selection is used to to select the features that are most informative, and relevant. 
Using Sickit-learn’s SelectPercentile, where features are selected according to a given percentile using a scoring function.
One-hot encoding is then used to transform categorical data to a vector with a series of ones and zeros. 

## Prameter Tuning

* In this stage the machine learning algorithm that is going to be used is chosen, as there are various algorithms experimented with.
Parameter tuning, is the task of finding the best parameters for a machine learning model, given a specific problem. 

* These parameters improve the overall performance of the model. 

* Using Sickit-Learn GridSearchCV method, which takes as an input the machine learning algorithm,  and a set of values for a given parameter.

* It tries all these values and compares the results of each one, and finally returns the best performing value for the given parameter and algorithm. 

## Training

After choosing ML algorithm and tuning it is parameter, we pass 75% of input data to  train it.
This process is repeated for different machine learning algorithms such as Naïve Bayes ,SVM and Neural Networks.
After training the classifier a  model is generated, which is later tested.

## Experimental Results:

Now, we are going to test the trained models and compare their accuracies.

![alt text](https://github.com/abadrawy/ArabicPOSTagger/blob/master/graphs%26Images/classifiers.png)

## Basline System

![alt text](https://github.com/abadrawy/ArabicPOSTagger/blob/master/graphs%26Images/baseline.png)

## Applying Feature Selection

![alt text](https://github.com/abadrawy/ArabicPOSTagger/blob/master/graphs%26Images/featureSelection.png)

## High Dimensionality Problem 

Since our vocab is around 40,000, this results in a feature vector having high dimensionality, leading to large processing time and space inefficiency. 


There are 2 approaches to solving this  problem:
* Dimensionality Reduction
* Word2vec 

## Dimensionality Reduction

Dimensionality Reduction is projecting high dimensional data to a lower dimensional space, while preserving as much information as possible.


Using Sickit-learn’s TruncatedSVD, which takes as an input the feature vectors, and number of dimensions, it outputs a vector of the specified dimensions using matrices factorization.

## Applying Dimensionality Reduction

![alt text](https://github.com/abadrawy/ArabicPOSTagger/blob/master/graphs%26Images/dimensionalityReduction.png)

## Using Word Vectors

![alt text](https://github.com/abadrawy/ArabicPOSTagger/blob/master/graphs%26Images/usingWordVectors.png)

## Using Normalized Word Vectors

![alt text](https://github.com/abadrawy/ArabicPOSTagger/blob/master/graphs%26Images/usingNormalizedWordVectors.png)


## Conclusion

From experimenting with different input representations, it is observed that it affected the accuracy of classifiers, either increasing the performance or decreasing it. 


The overall highest performing classifier is the MLP, with an accuracy of 96.2%, when using normalized word vectors, producing near state of the art results. 

## References

* Trent Hauck. scikit-learn Cookbook. Packt Publishing Ltd, 2014. 
Daniel Jurafsky and H. James. Speech and language processing an introduction to natural language processing, computational linguistics, and speech. 2000. 

* Otakar Smrz, Jan Šnaidauf, and Petr Zemánek. Prague dependency treebank for arabic: Multi-level annotation of arabic corpus. In Proc. of the Intern. Symposium on Processing of Arabic, pages 147–155, 2002. 

* Vandana Korde and C.Namrata Mahender.Text classification and classifiers: A survey. International Journal of Artificial Intelligence & Applications, 3(2):85, 2012. 



















