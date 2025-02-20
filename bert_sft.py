import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from utils import load_data, categorize_sentiment, SentimentDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

merged_df = load_data()

print("\nMerged DataFrame:")
print(merged_df.head())

merged_df['sentiment_label'] = merged_df['sentiment_value'].apply(categorize_sentiment)

train_data = merged_df[merged_df['splitset_label'] == 1]
test_data = merged_df[merged_df['splitset_label'] == 2]

print("\nData Statistics:")
print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
train_texts = train_data['text'].tolist()
train_labels = train_data['sentiment_label'].tolist()

test_texts = test_data['text'].tolist()
test_labels = test_data['sentiment_label'].tolist()

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 3

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()

    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            val_correct += (preds == labels).sum().item()

    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = val_correct / len(test_dataset)
    print(f'Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.4f}')

    avg_loss = total_loss / len(train_loader)

    print(f"Training loss: {avg_loss}")

model.eval()
total_correct = 0
total_samples = 0

for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    predictions = torch.argmax(outputs.logits, dim=1)
    # print('pre:', predictions, 'labels:', labels)
    total_correct += (predictions == labels).sum().item()
    total_samples += labels.size(0)

accuracy = total_correct / total_samples
print(f"Accuracy: {accuracy * 100}%")

model.save_pretrained('sentiment_bert_model')
tokenizer.save_pretrained('sentiment_bert_model')

loaded_model = BertForSequenceClassification.from_pretrained('sentiment_bert_model')
loaded_tokenizer = BertTokenizer.from_pretrained('sentiment_bert_model')

def predict_sentiment(text):
    inputs = loaded_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = loaded_model(**inputs)
    
    prediction = torch.argmax(outputs.logits, dim=1).item()
    if prediction == 0:
        return "Negative"
    else:
        return "Positive"
    
test_sentence = "I love this product!"
print("Prediction:", predict_sentiment(test_sentence))