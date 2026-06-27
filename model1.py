# train_binary.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

# to save the model for stage 2
import joblib

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

# binary classifier
model = make_pipeline(vectorizer,
                      SGDClassifier(        # SGD
                        loss='log_loss',    # logistic loss function
                        penalty='l2',       # regularization
                        max_iter=1000,
                        random_state=44
                      )
)
                    
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# Accuracy / Zero-one loss
print("Accuracy:", accuracy_score(y_test, y_pred))

# general report 
print('report:')
print(classification_report(y_test, y_pred))

# save the model
joblib.dump(model, 'stage1.pkl')