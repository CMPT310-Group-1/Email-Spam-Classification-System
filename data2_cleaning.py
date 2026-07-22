import pandas as pd

# Load dataset
email_dataset = pd.read_csv("base_datasets/sentinel_dataset_multiclass.csv")

# Keep only the columns we need
email_dataset = email_dataset[["text", "label", "category_name"]]

# Rename columns
email_dataset.columns = ["mod_text", "spam", "category"]

email_dataset.to_csv("cleaning_data.csv", index=False)