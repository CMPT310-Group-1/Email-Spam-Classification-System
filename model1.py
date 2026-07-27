# train_binary.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# to save the model for stage 2
import joblib

data = pd.read_csv('combinned_data.csv')

# added stage 2 data back into stage 1
stage2_data = pd.read_csv('cleaning_data.csv')
stage2_data = stage2_data.dropna(subset=['mod_text', 'spam'])
stage2_data = stage2_data[['mod_text', 'spam']]

stage2_data['mod_text'] = stage2_data['mod_text'].astype(str)
stage2_data['spam'] = stage2_data['spam'].astype(int)

# clean
data = data.dropna(subset=['mod_text', 'spam'])
data['mod_text'] = data['mod_text'].astype(str)
data['spam'] = data['spam'].astype(int)

data = pd.concat([data, stage2_data])

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

# does a cross validation
cv_score = cross_val_score(model, X_train, y_train, cv=10)

print(cv_score.mean())
                    
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# confusion matrix for display
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot().figure_.savefig('model1_confusion_matrix.png')

# Accuracy / Zero-one loss
print("Accuracy:", accuracy_score(y_test, y_pred))

# general report 
print('report:')
print(classification_report(y_test, y_pred))

# save the model
joblib.dump(model, 'stage1.pkl')