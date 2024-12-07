from sklearn.model_selection import train_test_split

from normalize_data import combined_dataset

# Remove duplicates
combined_dataset = combined_dataset.drop_duplicates(subset=['code']).dropna()

# Split into train, validation, and test sets
train_data, test_data = train_test_split(combined_dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Inspect splits
print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Test data size: {len(test_data)}")
