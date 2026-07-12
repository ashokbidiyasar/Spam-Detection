import os
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

MAX_LEN    = 100       # max tokens per message
VOCAB_SIZE = 15000     # top-N words to keep
EMBED_DIM  = 64        # embedding dimension
HIDDEN_DIM = 64        # BiLSTM hidden units per direction
NUM_LAYERS = 2         # number of BiLSTM layers
BATCH_SIZE = 64
EPOCHS     = 10
LR         = 1e-3

PAD_IDX = 0
UNK_IDX = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


#  TEXT CLEANING

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)           # remove HTML tags
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  # remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)           # remove emails
    text = re.sub(r'[^a-z\s]', ' ', text)          # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str):
    return clean_text(text).split()


#  LOAD DATA

DATA_PATH = os.path.join('dataset', 'spam_Emails_data.csv')


df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]
df['label'] = df['label'].str.strip().str.lower().map({'spam': 1, 'ham': 0})
df = df[['text', 'label']].dropna()
df.drop_duplicates(subset=['text'], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\nTotal samples : {len(df):,}")
print(f"Spam (1)      : {df['label'].sum():,}")
print(f"Ham  (0)      : {(df['label'] == 0).sum():,}")


#  TRAIN / TEST SPLIT

train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df['label']
)
print(f"\nTrain: {len(train_df):,}  |  Test: {len(test_df):,}")


#  BUILD CUSTOM VOCABULARY  (from training data only)

counter = Counter()
for text in train_df['text'].astype(str):
    counter.update(tokenize(text))

# <PAD>=0, <UNK>=1, then most common words
vocab = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
for word, _ in counter.most_common(VOCAB_SIZE - 2):
    vocab[word] = len(vocab)

print(f"Vocabulary size: {len(vocab):,} tokens")


#  ENCODE SEQUENCES

def encode(text: str) -> list:
    tokens = tokenize(text)[:MAX_LEN]
    ids = [vocab.get(t, UNK_IDX) for t in tokens]
    ids += [PAD_IDX] * (MAX_LEN - len(ids))   # pad to MAX_LEN
    return ids


#  DATASET

class SpamDataset(Dataset):
    def __init__(self, dataframe):
        texts = dataframe['text'].astype(str).tolist()
        self.input_ids = torch.tensor(
            [encode(t) for t in texts], dtype=torch.long
        )
        self.labels = torch.tensor(
            dataframe['label'].values, dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]

train_loader = DataLoader(SpamDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(SpamDataset(test_df),  batch_size=BATCH_SIZE, shuffle=False)


#  BiLSTM MODEL

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(x)
        combined = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(combined)


model = BiLSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")


#  TRAINING

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

best_acc = 0.0
best_model_state = None

print(f"\n{'Epoch':>6} {'Loss':>10} {'Accuracy':>10}")
print("-" * 30)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for ids, labels in train_loader:
        ids, labels = ids.to(DEVICE), labels.to(DEVICE)
        pred = model(ids)
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += ((pred >= 0.5).float() == labels).sum().item()
        total      += len(labels)

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for ids, labels in test_loader:
            ids, labels = ids.to(DEVICE), labels.to(DEVICE)
            pred = model(ids)
            val_correct += ((pred >= 0.5).float() == labels).sum().item()
            val_total   += len(labels)

    val_acc = val_correct / val_total
    scheduler.step(val_acc)
    print(f"{epoch:>6}   {total_loss/total:>8.4f}   {correct/total*100:>8.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}


#  EVALUATION

model.load_state_dict(best_model_state)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for ids, labels in test_loader:
        pred = model(ids.to(DEVICE))
        all_preds.extend((pred.cpu() >= 0.5).int().numpy().flatten())
        all_labels.extend(labels.int().numpy().flatten())


print(f"Accuracy  : {accuracy_score(all_labels, all_preds)*100:.2f}%")
print(f"Precision : {precision_score(all_labels, all_preds)*100:.2f}%")
print(f"Recall    : {recall_score(all_labels, all_preds)*100:.2f}%")
print(f"F1 Score  : {f1_score(all_labels, all_preds):.4f}")
print()
print(classification_report(all_labels, all_preds, target_names=['Ham', 'Spam']))


os.makedirs('model', exist_ok=True)
torch.save(best_model_state, os.path.join('model', 'bilstm_model.pt'))

with open(os.path.join('model', 'vocab.pkl'), 'wb') as f:
    pickle.dump(vocab, f)

config = {
    'vocab_size': len(vocab),
    'embed_dim':  EMBED_DIM,
    'hidden_dim': HIDDEN_DIM,
    'num_layers': NUM_LAYERS,
    'max_len':    MAX_LEN,
}
with open(os.path.join('model', 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)

print("  Saved: model/bilstm_model.pt")
print("  Saved: model/vocab.pkl")

