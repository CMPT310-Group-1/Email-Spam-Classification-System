# train_category.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
import joblib

# load dataset
data = pd.read_csv('cleaning_data.csv')

# clean
data = data.dropna(subset=['mod_text', 'category'])
data['mod_text'] = data['mod_text'].astype(str)

# train only on spam emails
data = data[data['spam'] == 1]

X = data['mod_text']
y = data['category']

# split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=44,
    stratify=y
)

# word count feature vector
vectorizer = CountVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=5000
)

# multiclass classifier
model = make_pipeline(
    vectorizer,
    SGDClassifier(
        loss='log_loss',
        penalty='l2',
        max_iter=1000,
        random_state=44
    )
)

# cross validation
cv_score = cross_val_score(model, X_train, y_train, cv=10)

print("Cross Validation Accuracy:", cv_score.mean())

# train
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# save model
#joblib.dump(model, 'stage2.pkl')