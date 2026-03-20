import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import os

class SarcasmDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_len=128):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])
        label = self.labels[item]
        encoding = self.tokenizer(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model():
    train_df = pd.read_csv("data/train.csv")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Use a small subset for fine-tuning in this environment (1000 samples for speed)
    train_subset = train_df.sample(n=min(1000, len(train_df)), random_state=42)
    
    train_dataset = SarcasmDataset(
        train_subset['comment'].to_numpy(),
        train_subset['label'].to_numpy(),
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Simple training loop (1 epoch for demonstration/speed)
    model.train()
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
    # Save model
    if not os.path.exists("models/sarcasm_bert"):
        os.makedirs("models/sarcasm_bert")
    model.save_pretrained("models/sarcasm_bert")
    tokenizer.save_pretrained("models/sarcasm_bert")
    print("Fine-tuned model saved successfully.")

if __name__ == "__main__":
    train_model()
