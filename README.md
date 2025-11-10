# 🧠 Legal Clause Similarity Using Siamese BiLSTM & Attention Encoder

This project implements deep learning models to identify **semantic similarity between legal clauses**. Two architectures — **Siamese BiLSTM** and **Attention Encoder** — are trained and compared to evaluate their ability to understand legal text semantics.

---

## 🚀 Project Overview
Legal documents often contain clauses with similar meanings but different wording. This project builds models that can automatically determine whether two clauses are semantically similar.  
We use **Siamese neural networks** with **BiLSTM** and **Attention-based** encoders to capture contextual and semantic relationships between clauses.

---

## 📂 Dataset
- **Input:** `final_file.csv`  
- **Columns:**  
  - `clause_text` — Legal clause text  
  - `clause_type` — Type/category of clause  
- **Preprocessing:**  
  - Lowercasing and punctuation removal  
  - Tokenization using `torchtext`’s `basic_english` tokenizer  
  - Balanced positive (same type) and negative (different type) clause pairs  

### Dataset Split
| Split | Percentage | Example Count |
|--------|-------------|---------------|
| Train | 80% | ~92,704 pairs |
| Test | 20% | ~23,176 pairs |

---

## 🧩 Model Architectures

### 🌀 Siamese BiLSTM
- Embedding layer: `vocab_size × 128`  
- BiLSTM: hidden size = 64 (bidirectional)  
- Fully connected layers: 128×64 → 64×1  
- Combines absolute difference and elementwise product of encoded vectors.

### 🎯 Attention Encoder
- Embedding layer: `vocab_size × 128`  
- BiLSTM: hidden size = 64 (bidirectional)  
- Attention layer: Linear(128 → 1)  
- Fully connected layers: 128×64 → 64×1  
- Attention highlights important words for deeper semantic understanding.

---

## ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross-Entropy |
| Batch Size | 32 |
| Epochs | 3 |
| Device | GPU/CPU |

---

## 📊 Performance Results

| Model | Epoch 1 Acc | Final Accuracy | F1-score | ROC-AUC |
|-------|--------------|----------------|-----------|----------|
| **Siamese BiLSTM** | 0.9988 | **0.9997** | 0.9997 | 0.99999 |
| **Attention Encoder** | 0.9670 | **0.9998** | 0.9998 | 0.99999 |

Both models achieved near-perfect accuracy and F1-scores, indicating outstanding ability to distinguish semantically similar and dissimilar clauses.

---

## 🧠 Comparative Analysis

- **Siamese BiLSTM:**  
  Efficient and simple architecture that captures general semantic similarity. Performs well but may not fully capture long or complex clause dependencies.

- **Attention Encoder:**  
  Achieved the highest ROC-AUC and F1-score (0.9998).  
  Better captures key terms in lengthy legal clauses.  
  Slightly higher computational cost due to the attention layer.

### Summary:
| Criterion | Best Model |
|------------|-------------|
| Accuracy & F1 | **Attention Encoder** |
| Speed & Efficiency | **Siamese BiLSTM** |

---
