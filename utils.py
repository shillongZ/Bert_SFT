import torch
import pandas as pd
from torch.utils.data import Dataset

def load_data():
    sentences = []
    with open('stanfordSentimentTreebank\datasetSentences.txt', 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2: 
                sentences.append({'sentence_index': int(parts[0]), 'text': parts[1]})

    sentences_df = pd.DataFrame(sentences)
    print("Sentences loaded:", len(sentences_df))

    splits = []
    with open('stanfordSentimentTreebank\datasetSplit.txt', 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2: 
                splits.append({'sentence_index': int(parts[0]), 'splitset_label': int(parts[1])})

    splits_df = pd.DataFrame(splits)
    print("Splits loaded:", len(splits_df))

    labels = []
    with open('stanfordSentimentTreebank\sentiment_labels.txt', 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2: 
                labels.append({'phrase_id': int(parts[0]), 'sentiment_value': float(parts[1])})

    labels_df = pd.DataFrame(labels)
    print("Labels loaded:", len(labels_df))

    merged_df = pd.merge(sentences_df, splits_df, on='sentence_index', how='left')
    merged_df = pd.merge(merged_df, labels_df, left_on='sentence_index', right_on='phrase_id', how='left')

    merged_df.drop(['phrase_id'], axis=1, inplace=True)

    return merged_df

def categorize_sentiment(score):
    if score < 0.5:
        return 0
    else:
        return 1
    
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }