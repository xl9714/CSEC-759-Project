import pandas as pd

from data_preprocessing import  train_data, val_data, test_data
from model_training import train_tokens, val_tokens, test_tokens

train_labels = train_tokens['labels'].numpy()
val_labels = val_tokens['labels'].numpy()
test_labels = test_tokens['labels'].numpy()

print("Training label distribution:", pd.Series(train_labels).value_counts())
print("Validation label distribution:", pd.Series(val_labels).value_counts())
print("Test label distribution:", pd.Series(test_labels).value_counts())
