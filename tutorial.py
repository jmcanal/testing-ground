"""
Test code from https://nlpforhackers.io/deep-learning-introduction/
"""

import re
import pandas as pd


def clean_review(text):
    # Strip HTML tags
    text = re.sub('<[^<]+?>', ' ', text)

    # Strip escaped quotes
    text = text.replace('\\"', '')

    # Strip quotes
    text = text.replace('"', '')

    return text


df = pd.read_csv('labeledTrainData.tsv', sep='\t', quoting=3)

# Create a cleaned_review column
df['cleaned_review'] = df['review'].apply(clean_review)

# Check out how the cleaned review compares to the original one
# print(df['review'][0])
# print("\n\n")
# print(df['cleaned_review'][0])


# Logistic regression section

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Shuffle the data and then split it, keeping 20% aside for testing
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2)

vectorizer = CountVectorizer(lowercase=True)
vectorizer.fit(X_train)

# classifier = LogisticRegression()
classifier = MLPClassifier(hidden_layer_sizes=(100,))
classifier.fit(vectorizer.transform(X_train), y_train)

print("Score:", classifier.score(vectorizer.transform(X_test), y_test))  # Score: 0.8778
