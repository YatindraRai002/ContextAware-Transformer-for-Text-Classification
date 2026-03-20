import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def preprocess_data(file_path, output_dir):
    df = pd.read_csv(file_path)
    df = df[['label', 'comment']].dropna()
    df['comment'] = df['comment'].apply(clean_text)
    df = df[df['comment'] != ""]
    
    df_sample = df.sample(n=min(20000, len(df)), random_state=42)
    
    train, test = train_test_split(df_sample, test_size=0.2, random_state=42, stratify=df_sample['label'])
    
    train.to_csv(f"{output_dir}/train.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)
    print(f"Preprocessed data saved to {output_dir}")

if __name__ == "__main__":
    raw_data_path = "data/train-balanced-sarcasm.csv"
    output_directory = "data"
    # Ensure directory exists
    import os
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    preprocess_data(raw_data_path, output_directory)
