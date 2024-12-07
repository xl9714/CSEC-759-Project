from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the fine-tuned model
model_path = "./vulnerability_detection_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Model and tokenizer loaded successfully!")

# Load tokenized datasets
train_tokens = torch.load("tokenized_train_data.pt", weights_only=True)
val_tokens = torch.load("tokenized_val_data.pt", weights_only=True)
test_tokens = torch.load("tokenized_test_data.pt", weights_only=True)

# Dataset class
class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = tokenized_data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

# Wrap tokenized data
train_dataset = TokenizedDataset(train_tokens)
val_dataset = TokenizedDataset(val_tokens)
test_dataset = TokenizedDataset(test_tokens)

training_args = TrainingArguments(
    output_dir="./results",                # Directory to save results
    evaluation_strategy="epoch",          # Evaluate at the end of each epoch
    save_strategy="epoch",                # Save model at the end of each epoch
    learning_rate=1e-5,                   # Learning rate
    per_device_train_batch_size=8,        # Batch size for training
    per_device_eval_batch_size=8,         # Batch size for evaluation
    num_train_epochs=5,                   # Number of epochs
    weight_decay=0.01,                    # Weight decay for optimization
    logging_dir="./logs",                 # Directory for logs
    logging_steps=10,                     # Log every 10 steps
    save_total_limit=2,                   # Save the two best models
    load_best_model_at_end=True,          # Load the best model at the end of training
    metric_for_best_model="f1",           # Use F1-score to select the best model
    greater_is_better=True,               # Higher F1-score is better
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./vulnerability_detection_model_finetuned")

# Evaluate the model on the test dataset
results = trainer.evaluate(test_dataset)
print("Test Results:", results)