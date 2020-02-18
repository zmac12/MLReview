# Convolutional Neural Network w/Snowflake Source POC #

#Snowflake
import snowflake.connector

#Keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer

#Other
import os
import nltk
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from numpy import array

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression





# Snowflake Credentials from env variables #
PASSWORD = os.getenv('SNOWSQL_PWD')
WAREHOUSE = os.getenv('SNOWWAREHOUSE')
ACCOUNT = os.getenv('SNOWACCT')
USER = os.getenv('SNOWUSER')


# Connection Manager for Snowflake Instance #
con = snowflake.connector.connect(
    user=USER,
    password=PASSWORD,
    account=ACCOUNT,
    warehouse=WAREHOUSE,
    schema='PUBLIC',
    database='CNNPOC'
)

cur = con.cursor()

sql = 'SELECT * FROM CNNPOC.PUBLIC.LABELLEDREVIEWSNEW'

cur.execute(sql)

df = cur.fetch_pandas_all()

# Start of Convolutional Models #

# Yelp Log Reg for baseline test #
df_yelp = df[df['RECORDSOURCE'] == 'yelp']

sentences = df_yelp['SENTENCE'].values

y = df_yelp['LABEL'].values
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print('Accuracy of Log Reg: ', score)

# Log Reg for each unique source in dataframe #
for source in df['RECORDSOURCE'].unique():
    df_source = df[df['RECORDSOURCE'] == source]
    sentences = df_source['SENTENCE'].values
    y = df_source['LABEL'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test  = vectorizer.transform(sentences_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print('Accuracy for {} data: {:.4f}'.format(source, score))

