# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import os

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
#Multinomial Naive Bayes classification Algorithm
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
    nb_model = algo.fit(training_features,training_class)
    predicted_class = nb_model.predict(test_features)
    conf_matrix = pd.crosstab(predicted_class,np.array(test_class),
                              rownames = ["Predicted Sentiment"],
                              colnames = ["Actual Sentiment"])
    print(conf_matrix)
    print("Accuracy = ",accuracy_score(test_class,predicted_class)*100)

text_classification(X_train,y_train,X_test,y_test) #66.6% accuracy

# IDV without stopwords
X_train_wo_stopwords = text_features_wo_stopwords[:10,:]
X_test_wo_stopwords = text_features_wo_stopwords[10:,:] 

text_classification(X_train_wo_stopwords,y_train,X_test_wo_stopwords,y_test) #83.33%

#Idv with Term frequency
x_train_tf = text_features_tf[:10,:]
x_test_tf = text_features_tf[10:,:]

text_classification(x_train_tf,y_train,x_test_tf,y_test) #83.33%

#Using other models

#DecisionTreeClassifier
text_classification(x_train_tf,y_train,x_test_tf,y_test,DecisionTreeClassifier())

###############  Assignment #########################
os.chdir("E:\Python Class")
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

All_data = happy_train + sad_train + happy_test + sad_test
ss = ["Happy"]*90
ss_1 = ["Sad"]*90
Sentiment = ss[0:80] + ss_1[0:80] + ss[80:90] + ss_1[80:90]

### FEATURE EXTRACTION
feature_algo = TfidfVectorizer()
text_features_1 = feature_algo.fit_transform(All_data)
print(text_features_1)
feature_algo.vocabulary_

# Removing stop words
feature_algo_wo_stopwords = TfidfVectorizer(stop_words='english')
text_features_wo_stopwords = feature_algo_wo_stopwords.fit_transform(All_data)
feature_algo_wo_stopwords.vocabulary_

# IDV
X_train = text_features_wo_stopwords[:160,:] # extracting first 160 text for training
X_test = text_features_wo_stopwords[160:,:] # extracting remaining 20 text for testing
# DV
y_train = Sentiment[:160]
y_test = Sentiment[160:]

nb_model = MultinomialNB().fit(X_train,y_train)
y_predicted = nb_model.predict(X_test)

pd.crosstab(y_predicted,np.array(y_test),
            rownames = ["Predicted Sentiment"],
            colnames = ["Actual Sentiment"])

accuracy_score(y_test,y_predicted)

#90% accuracy

def text_classification(training_features,training_class,test_features,test_class,algo =  MultinomialNB() ):
    model = algo.fit(training_features,training_class)
    predicted_class = nb_model.predict(test_features)
    conf_matrix = pd.crosstab(predicted_class,np.array(test_class),
                              rownames = ["Predicted Sentiment"],
                              colnames = ["Actual Sentiment"])
    print(conf_matrix)
    print("Accuracy = ",accuracy_score(test_class,predicted_class)*100)

#Decision Tree
text_classification(X_train,y_train,X_test,y_test,DecisionTreeClassifier())
# 90% accuracy

#KNN Classifier
text_classification(X_train,y_train,X_test,y_test,KNeighborsClassifier())
# 90% accuracy

########### NLTK ################################################

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


