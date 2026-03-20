import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, f1_score, accuracy_score

def evaluate():
    test_df = pd.read_csv("data/test.csv")
    test_subset = test_df.sample(n=min(500, len(test_df)), random_state=42)
    comments = test_subset['comment'].tolist()
    labels = test_subset['label'].tolist()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "models/sarcasm_bert"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for comment in comments:
            inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)
            
    print("Fine-tuned BERT Classification Report:")
    print(classification_report(labels, predictions))
    
    # Simple comparison summary
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    with open("results/comparison_metrics.txt", "w") as f:
        f.write(f"Fine-tuned BERT Accuracy: {accuracy:.4f}\n")
        f.write(f"Fine-tuned BERT F1 Score: {f1:.4f}\n")
    print("Evaluation results saved to results/comparison_metrics.txt")

if __name__ == "__main__":
    evaluate()
