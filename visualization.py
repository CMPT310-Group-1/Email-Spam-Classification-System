import pandas as pd
import joblib
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
# python visualization.py

# load model
stage1 = joblib.load('stage1.pkl')
stage2 = joblib.load('stage2.pkl')

# load data
stage1_data = pd.read_csv('combinned_data.csv')
stage1_data['mod_text'] = stage1_data['mod_text'].astype(str)
stage1_data['spam'] = stage1_data['spam'].astype(int)

stage2_data = pd.read_csv('cleaning_data.csv')
stage2_data['mod_text'] = stage2_data['mod_text'].astype(str)
stage2_data['spam'] = stage2_data['spam'].astype(int)

# add part 2 data to part 1
stage1_data = pd.concat([stage1_data, stage2_data[['mod_text', 'spam']]])

stage2_data = stage2_data[stage2_data['spam'] == 1]

# first prints out where the models made errors 
predict1 = stage1.predict(stage1_data['mod_text'])

model1_incorrect = stage1_data
model1_incorrect['predict'] = predict1

model1_incorrect = model1_incorrect[model1_incorrect['predict'] != model1_incorrect['spam']]

predict2 = stage2.predict(stage2_data['mod_text'])

model2_incorrect = stage2_data
model2_incorrect['predict'] = predict2

model2_incorrect = model2_incorrect[model2_incorrect['predict'] != model2_incorrect['category']]

print('model 1 errors')
print(model1_incorrect)

print('model 2 errors')
print(model2_incorrect)

# additional image visualization

RocCurveDisplay.from_estimator(stage1, stage1_data['mod_text'], stage1_data['spam'])
plt.show()

RocCurveDisplay.from_estimator(stage2, stage2_data['mod_text'], stage2_data['category'])
plt.show()