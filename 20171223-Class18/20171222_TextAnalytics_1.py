# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#nltk.download() # download popular packages

pos_tweets=[('I love this car','positive'), 
('This view is amazing','positive'),
('I feel great this morning','positive'),
('I am so excited about the concert','positive'),
('He is my best friend','positive')]

neg_tweets=[('I do not like this car','negative'),
('This view is horrible','negative'),
('I feel tired this morning','negative'),
('I am not looking forward to the concert','negative'),
('He is my enemy','negative')]

test_tweets=[('I feel happy this morning','positive'), 
('Larry is my friend','positive'),
('I do not like that man','negative'),
('This view is horrible','negative'),
('The house is not great','negative'),
('Your song is annoying','negative')]

all_tweets = pos_tweets + neg_tweets + test_tweets
tweets = [] # IDV
sentiment = [] # DV
for tw in all_tweets:
    tweets.append(tw[0])
    sentiment.append(tw[1])

### FEATURE EXTRACTION
feature_algo = TfidfVectorizer()
text_features = feature_algo.fit_transform(tweets)
print(text_features)
text_features_raw_matrix = text_features.toarray()
feature_algo.vocabulary_

# Removing stop words
feature_algo_wo_stopwords = TfidfVectorizer(stop_words='english')
text_features_wo_stopwords = feature_algo_wo_stopwords.fit_transform(tweets)
text_features__wo_stopwords_raw_matrix = text_features_wo_stopwords.toarray()
feature_algo_wo_stopwords.vocabulary_

# Getting term frequency matrix without idf weighting
feature_algo_tf = TfidfVectorizer(use_idf=False,norm=None)
text_features_tf = feature_algo_tf.fit_transform(tweets)
text_features_tf_raw_matrix = text_features_tf.toarray()
feature_algo_tf.vocabulary_

# IDV
X_train = text_features[:10,:] # extracting first 10 tweets for training
X_test = text_features[10:,:] # extracting remaining 6 tweets for testing
# DV
y_train = sentiment[:10]
y_test = sentiment[10:]

# Building model on training data
# Multinomial Naive Baye's CLassification Algorithm
nb_model = MultinomialNB().fit(X_train,y_train)

# EValuate the model on test data
y_predicted = nb_model.predict(X_test)

# Confusion matrix
pd.crosstab(y_predicted,np.array(y_test),
            rownames = ["Predicted Sentiment"],
            colnames = ["Actual Sentiment"])
4/6 #66.66% accuracy
accuracy_score(y_test,y_predicted)


## Generic classification function
def text_classification(training_features,training_class,test_features,test_class,algo =  MultinomialNB() ):
    text_classifier_model = algo.fit(training_features,training_class)
    predicted_class = text_classifier_model.predict(test_features)
    conf_matrix = pd.crosstab(predicted_class,np.array(test_class),
                              rownames = ["Predicted Sentiment"],
                              colnames = ["Actual Sentiment"])
    print(conf_matrix)
    print("Accuracy = ",accuracy_score(test_class,predicted_class)*100)

text_classification(X_train,y_train,X_test,y_test) #66.6% accuracy
text_classification(X_train,y_train,X_test,y_test,KNeighborsClassifier()) #50% accuracy
text_classification(X_train,y_train,X_test,y_test,DecisionTreeClassifier()) #83.3% accuracy

# IDV without stopwords
X_train_wo_stopwords = text_features_wo_stopwords[:10,:]
X_test_wo_stopwords = text_features_wo_stopwords[10:,:] 

text_classification(X_train_wo_stopwords,y_train,X_test_wo_stopwords,y_test) #83.33%
text_classification(X_train_wo_stopwords,y_train,X_test_wo_stopwords,y_test,DecisionTreeClassifier()) #83.33%

# IDV with term frequncy
X_train_tf= text_features_tf[:10,:]
X_test_tf = text_features_tf[10:,:] 
text_classification(X_train_tf,y_train,X_test_tf,y_test)  #83.3


###############  Assignment #########################
# Happy training
f = open("data/happy.txt","r",encoding='utf8')
happy_train = f.readlines()
f.close()
# Happy test data
f = open("data/happy_test.txt","r",encoding='utf8')
happy_test = f.readlines()
f.close()
# Sad training
f = open("data/sad.txt","r",encoding='utf8')
sad_train = f.readlines()
f.close()
# Sad test
f = open("data/sad_test.txt","r",encoding='utf8')
sad_test = f.readlines()
f.close()

# Build a sentiment analysis classifier using training data
# Test the model on test data

###### Solution ############################################
tweets = happy_train + sad_train + happy_test + sad_test
sentiment = ["happy"]*80 + ["sad"]*80 + ["happy"]*10 + ["sad"]*10

# features without stop words
text_feature_tf_withoutstopwords = TfidfVectorizer(stop_words='english',
                                  use_idf = False,
                                  norm = None).\
                                  fit_transform(tweets)
text_feature_tf_withoutstopwords_mat = text_feature_tf_withoutstopwords.toarray()

# features with stop words
text_feature_tf_withstopwords = TfidfVectorizer(stop_words=None,
                                  use_idf = None,
                                  norm = None).\
                                  fit_transform(tweets)                                   
text_feature_tf_withstopwords_mat = text_feature_tf_withstopwords.toarray()


# tfidf features 
text_feature_tfidf_withstopwords = TfidfVectorizer(stop_words=None,
                                  use_idf = True,
                                  norm = None).\
                                  fit_transform(tweets)                                   
text_feature_tfidf_withstopwords_mat = text_feature_tfidf_withstopwords.toarray()


text_classification(text_feature_tf_withoutstopwords[:160,:],
                    sentiment[:160],
                    text_feature_tf_withoutstopwords[160:180,:],
                    sentiment[160:180])
text_classification(text_feature_tf_withstopwords[:160,:],
                    sentiment[:160],
                    text_feature_tf_withstopwords[160:180,:],
                    sentiment[160:180])
text_classification(text_feature_tfidf_withstopwords[:160,:],
                    sentiment[:160],
                    text_feature_tfidf_withstopwords[160:180,:],
                    sentiment[160:180])

########### NLTK ################################################
# Natural Language Toolkit
### word tokenization
text = " this is Python class"
text_tokenized = word_tokenize(text)
# converting to lower case
text_tokenized_lower = [token.lower() for token in text_tokenized] 
print (text_tokenized_lower)

### removing stop word
eng_stop_words = stopwords.words('english')
text_without_stopwords = \
    [t for t in text_tokenized if not t in eng_stop_words]
print (text_without_stopwords)

#### stemming
PorterStemmer().stem("winning")
PorterStemmer().stem("wins")
PorterStemmer().stem("winner")
PorterStemmer().stem("victorious")
PorterStemmer().stem("victory")


