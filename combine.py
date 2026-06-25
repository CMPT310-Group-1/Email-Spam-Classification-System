import numpy as np
import pandas as pd

# python combine.py

# loads the two datasets
spam_dataset = pd.read_csv('base_datasets/spam_v2.csv', encoding_errors='replace')
email_dataset = pd.read_csv('base_datasets/emails.csv.zip')

# sliced of the subject 
email_dataset['mod_text'] = email_dataset['text'].str[9:]

# into the some format for spam as other
spam_dataset['spam'] = (spam_dataset['v1'] == 'spam').astype(int)

spam_dataset.columns = ['v1', 'mod_text', 'spam']

print(spam_dataset)

# combining 
new_spam = spam_dataset[['mod_text', 'spam']].copy()
new_email = email_dataset[['mod_text', 'spam']].copy()

new_data = pd.concat([new_email, new_spam], ignore_index=True)

new_data.to_csv('combinned_data.csv', index=False, errors='replace')