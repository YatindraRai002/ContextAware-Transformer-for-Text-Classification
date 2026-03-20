from transformers import pipeline
import pandas as pd
from sklearn.metrics import classification_report

def run_baseline_inference(test_file):
    df = pd.read_csv(test_file)
    # Take a small sample for evaluation to avoid long execution
    df_eval = df.sample(n=min(500, len(df)), random_state=42)
    
    # Using standard sentiment analysis pipeline as a baseline for sarcasm
    # Sarcasm often flips sentiment, so a general sentiment model might struggle
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    texts = df_eval['comment'].tolist()
    results = classifier(texts, truncation=True)
    
    # Map 'POSITIVE' to label-appropriate or just use for comparison
    # Sarcasm detection usually expects 1 (Sarcastic) or 0 (Not Sarcastic)
    # SST-2 maps POSITIVE to 1 (which we treat as Sarcastic) and NEGATIVE to 0
    predictions = [1 if r['label'] == 'POSITIVE' else 0 for r in results]
    
    print("Baseline (Pre-trained Sentiment) Classification Report:")
    report = classification_report(df_eval['label'], predictions)
    print(report)
    
    with open("results/baseline_metrics.txt", "w") as f:
        f.write("Baseline (Pre-trained Sentiment) Report:\n")
        f.write(report)
    
    # Save a few examples
    import os
    if not os.path.exists("results"):
        os.makedirs("results")
    df_eval['baseline_pred'] = predictions
    df_eval.head(10).to_csv("results/baseline_examples.csv", index=False)

if __name__ == "__main__":
    run_baseline_inference("data/test.csv")
