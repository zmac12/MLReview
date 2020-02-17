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
from sklearn.model_selection import train_test_split


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

