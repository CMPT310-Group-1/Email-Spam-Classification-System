# train_binary.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('combinned_data.csv')

# clean
data = data.dropna(subset=['mod_text', 'spam'])
data['mod_text'] = data['mod_text'].astype(str)
data['spam'] = data['spam'].astype(int)

X = data['mod_text']
y = data['spam']

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=44,
    stratify=y
)

# word count feature vector
vectorizer = CountVectorizer(
    lowercase=True, # Change to lowercase
    stop_words='english', # remove meaningless words e.g) a, the, is, at
    max_features=5000 # use only top 5000 words
)

X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

# binary classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train_count, y_train)

# prediction
y_pred = model.predict(X_test_count)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))