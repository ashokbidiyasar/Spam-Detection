# 📧 Spam Detection — BiLSTM

A binary text classifier that identifies **spam vs. ham (legitimate) emails** using a Bidirectional LSTM (BiLSTM) built from scratch in PyTorch. No pretrained embeddings or transformers — everything from the vocabulary to the weights is trained on the dataset itself.

---

## 🧠 Why BiLSTM?

Traditional approaches like Naive Bayes or TF-IDF + Logistic Regression treat text as a **bag of words** — they ignore word order entirely. A BiLSTM reads the sequence **left-to-right and right-to-left simultaneously**, capturing contextual meaning in both directions.

This matters for spam detection because phrases like:
- *"You have won a prize"* vs *"I won the argument"*

...share words but carry completely different intent. The BiLSTM learns these patterns through sequence context rather than just word frequency.

---

## 📦 Dataset

| Property | Value |
|---|---|
| File | `spam_Emails_data.csv` |
| Columns used | `text`, `label` |
| Labels | `spam` → 1, `ham` → 0 |
| Train / Test Split | 80% / 20% (stratified) |

The dataset is loaded, cleaned, and deduplicated before any vocabulary is built. The vocabulary is constructed **only from training data** to prevent data leakage into the test set.

> **Note:** The dataset is not included in this repository (it's listed in `.gitignore`). You must download it manually before running the training script.

### ⬇️ Download

The dataset is available on **Kaggle**:

👉 [Spam Emails Dataset on Kaggle](https://www.kaggle.com/datasets/search?q=spam+emails) ← *(replace with the exact dataset URL)*

**Steps:**
1. Go to the Kaggle dataset page
2. Click **Download** (you'll need a free Kaggle account)
3. Extract the ZIP and place `spam_Emails_data.csv` inside the `dataset/` folder:

```
Spam Detection/
└── dataset/
    └── spam_Emails_data.csv   ← place it here
```

**Or via Kaggle API:**
```bash
pip install kaggle
kaggle datasets download -d <author>/<dataset-name> --unzip -p dataset/
```
*(Replace `<author>/<dataset-name>` with the actual dataset slug from the Kaggle URL)*

---


## ⚙️ Pipeline

### Step 1 — Text Cleaning
Each raw message goes through the following preprocessing:
- Convert to **lowercase**
- Strip **HTML tags** (`<br>`, `<p>`, etc.)
- Remove **URLs** (`http://...`, `www....`)
- Remove **email addresses**
- Remove all **non-alphabetic characters** (numbers, punctuation, symbols)
- Collapse multiple spaces into one

### Step 2 — Vocabulary Building
A custom vocabulary of the **top 15,000 most frequent tokens** is built from the training set only.
Two special tokens are always reserved:
- `<PAD>` (index 0) — used to pad shorter sequences to a fixed length
- `<UNK>` (index 1) — used for words not seen during training

### Step 3 — Sequence Encoding
Each message is tokenized and converted to a list of integer IDs using the vocabulary.
- Sequences longer than **100 tokens** are truncated
- Shorter sequences are **zero-padded** to length 100

### Step 4 — Model Training
The BiLSTM is trained for up to **10 epochs** using the Adam optimizer with a `ReduceLROnPlateau` learning rate scheduler that halves the LR when validation accuracy plateaus.

Gradient clipping (`max norm = 1.0`) is applied to prevent exploding gradients — a common issue with deep RNN stacks.

The **best model checkpoint** (by validation accuracy) is saved during training and restored for final evaluation.

### Step 5 — Evaluation
Final metrics are reported on the held-out test set:
- Accuracy, Precision, Recall, F1 Score
- Full per-class classification report (Ham / Spam)

---

## 🏗️ Model Architecture

```
Input (token IDs)  [batch × 100]
    ↓
Embedding Layer    [15000 vocab × 64 dims]  — trained from scratch
    ↓
BiLSTM             [64 hidden units per direction, 2 layers]
    ↓
Final Hidden State [forward[-1] + backward[-1]  →  128-dim vector]
    ↓
Linear(128 → 64) → ReLU → Linear(64 → 1) → Sigmoid
    ↓
Spam (1) / Ham (0)
```

The final hidden state concatenates the **last forward** and **last backward** hidden states, giving the model a summary of the entire sequence from both reading directions before making a prediction.

---

## 🔧 Hyperparameters

| Parameter | Value |
|---|---|
| Max sequence length | 100 tokens |
| Vocabulary size | 15,000 |
| Embedding dimension | 64 |
| BiLSTM hidden units | 64 per direction (128 total) |
| BiLSTM layers | 2 |
| Batch size | 64 |
| Epochs | 10 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| LR scheduler | ReduceLROnPlateau (patience=2, factor=0.5) |
| Loss function | Binary Cross-Entropy (BCELoss) |

---

## 📁 Project Structure

```
Spam Detection/
├── dataset/
│   └── spam_Emails_data.csv
├── model/                  # Created after training
│   ├── bilstm_model.pt     # Saved model weights (best checkpoint)
│   ├── vocab.pkl           # Custom vocabulary (word → index)
│   └── config.pkl          # Model config for inference
├── train.py                # Full training pipeline
└── requirements.txt
```

---

## 🚀 Setup & Usage

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Train the model:**
```bash
python train.py
```

After training, the best model weights, vocabulary, and config are automatically saved to the `model/` directory.

---

## 📊 Results

> Fill in after running `train.py`. Typical BiLSTM performance on email spam datasets:

| Metric | Expected Range |
|---|---|
| Accuracy | 97 – 99% |
| Precision | 96 – 99% |
| Recall | 95 – 98% |
| F1 Score | 0.96 – 0.99 |

