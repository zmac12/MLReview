# Convolutional Neural Network w/Snowflake Source POC #

# Snowflake
import snowflake.connector

# Keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras import layers

# Other
import os
import nltk
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from numpy import array
import matplotlib.pyplot as plt

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Functions #
def plot_history(history):
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, "b", label="Training acc")
    plt.plot(x, val_acc, "r", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, "b", label="Training loss")
    plt.plot(x, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()


# Snowflake Credentials from env variables #
PASSWORD = os.getenv("SNOWSQL_PWD")
WAREHOUSE = os.getenv("SNOWWAREHOUSE")
ACCOUNT = os.getenv("SNOWACCT")
USER = os.getenv("SNOWUSER")


# Connection Manager for Snowflake Instance #
con = snowflake.connector.connect(
    user=USER,
    password=PASSWORD,
    account=ACCOUNT,
    warehouse=WAREHOUSE,
    schema="PUBLIC",
    database="CNNPOC",
)

cur = con.cursor()

sql = "SELECT * FROM CNNPOC.PUBLIC.LABELLEDREVIEWSNEW"

cur.execute(sql)

df = cur.fetch_pandas_all()

#### Start of Models ####

# Yelp Log Reg for baseline test #
df_yelp = df[df["RECORDSOURCE"] == "yelp"]

sentences = df_yelp["SENTENCE"].values

y = df_yelp["LABEL"].values
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000
)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy of Log Reg: ", score)

# Log Reg for each unique source in dataframe #
for source in df["RECORDSOURCE"].unique():
    df_source = df[df["RECORDSOURCE"] == source]
    sentences = df_source["SENTENCE"].values
    y = df_source["LABEL"].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000
    )

    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test = vectorizer.transform(sentences_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print("Accuracy for {} data: {:.4f}".format(source, score))


# Start of Convolutional Neural Net #

input_dim = X_train.shape[1]

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    verbose=False,
    validation_data=(X_test, y_test),
    batch_size=10,
)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy: {:.4f}".format(accuracy))


plt.style.use("ggplot")
plot_history(history)
