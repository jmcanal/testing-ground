"""
Code from https://nlpforhackers.io/keras-intro/
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
import spacy
import numpy as np
import en_core_web_md
nlp = en_core_web_md.load()


# Data cleaning

def clean_review(text):
    # Strip HTML tags
    text = re.sub('<[^<]+?>', ' ', text)

    # Strip escaped quotes
    text = text.replace('\\"', '')

    # Strip quotes
    text = text.replace('"', '')

    return text


df = pd.read_csv('labeledTrainData.tsv', sep='\t', quoting=3)
df['cleaned_review'] = df['review'].apply(clean_review)
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2)


# Build count vectorizer

vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, max_features=5000)
X_train_onehot = vectorizer.fit_transform(X_train)


# transform the texts into sequences of word ids
# train an embeddings layer
# use the word ids to fetch the corresponding word vector

word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
tokenize = vectorizer.build_tokenizer()
preprocess = vectorizer.build_preprocessor()


def to_sequence(tokenizer, preprocessor, index, text):
    words = tokenizer(preprocessor(text))
    indexes = [index[word] for word in words if word in index]
    return indexes


print(to_sequence(tokenize, preprocess, word2idx, "This is an important test!"))  # [2269, 4453]
X_train_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_train]
print(X_train_sequences[0])


# Pad sequences to normalize the lengths
# Compute the max lenght of a text
MAX_SEQ_LENGTH = len(max(X_train_sequences, key=len))
print("MAX_SEQ_LENGTH=", MAX_SEQ_LENGTH)

N_FEATURES = len(vectorizer.get_feature_names())
X_train_sequences = pad_sequences(X_train_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)
print(X_train_sequences[0])


# Model definition

model = Sequential()
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    64,  # Embedding size
                    input_length=MAX_SEQ_LENGTH))
model.add(LSTM(64))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# Use Spacy embeddings

EMBEDDINGS_LEN = len(nlp.vocab['apple'].vector)
print("EMBEDDINGS_LEN=", EMBEDDINGS_LEN)  # 300

embeddings_index = np.zeros((len(vectorizer.get_feature_names()) + 1, EMBEDDINGS_LEN))
for word, idx in word2idx.items():
    try:
        embedding = nlp.vocab[word].vector
        embeddings_index[idx] = embedding
    except:
        pass


# Define the network with a pre-trained embedding layer

model = Sequential()
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    EMBEDDINGS_LEN,  # Embedding size
                    weights=[embeddings_index],
                    input_length=MAX_SEQ_LENGTH,
                    trainable=False))
model.add(LSTM(300))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# Transform the test data to sequences then pad them

X_test_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_test]
X_test_sequences = pad_sequences(X_test_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)


# Run model and evaluate

model.fit(X_train_sequences[:-100], y_train[:-100],
          epochs=1, batch_size=128, verbose=1,
          validation_data=(X_train_sequences[-100:], y_train[-100:]))

scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1])  # 0.8508

