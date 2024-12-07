import torch

# Load the tokenized data
tokenized_data = torch.load("tokenized_train_data.pt")

print("Example Input IDs:")
print(tokenized_data['input_ids'][:2])

print("\nExample Attention Masks:")
print(tokenized_data['attention_mask'][:2])

print("\nExample Labels:")
print(tokenized_data['labels'][:2])