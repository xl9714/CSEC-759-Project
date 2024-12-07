import torch
from transformers import AutoTokenizer
import pandas as pd

from data_preprocessing import train_data, val_data, test_data

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")


# Function to tokenize and save data in batches
def tokenize_and_save_in_batches(data, tokenizer, save_path, batch_size=100, max_length=512):
    """
    Tokenize and save a dataset in batches to avoid memory issues.
    :param data: DataFrame with 'code' and 'label' columns
    :param tokenizer: Pre-trained tokenizer
    :param save_path: Path to save the tokenized data
    :param batch_size: Number of rows to process at once
    :param max_length: Maximum token length
    """
    if data.empty:
        print(f"Data for {save_path} is empty! Skipping save.")
        return

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        tokenized_batch = tokenizer(
            list(batch['code']),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        all_input_ids.append(tokenized_batch['input_ids'])
        all_attention_masks.append(tokenized_batch['attention_mask'])
        all_labels.append(torch.tensor(list(batch['label'])))
        print(f"Processed batch {i // batch_size + 1}/{(len(data) + batch_size - 1) // batch_size}")

    # Combine all batches
    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_masks, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Save the tokenized data
    tokenized_data = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    torch.save(tokenized_data, save_path)
    print(f"Tokenized data successfully saved to {save_path}")


# Tokenize and save train, validation, and test datasets
tokenize_and_save_in_batches(train_data, tokenizer, "tokenized_train_data.pt")
tokenize_and_save_in_batches(val_data, tokenizer, "tokenized_val_data.pt")
tokenize_and_save_in_batches(test_data, tokenizer, "tokenized_test_data.pt")
