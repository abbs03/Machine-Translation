# English-Tamil Machine Translation Model

This repository contains a machine translation model built from scratch, inspired by the Transformer architecture introduced in the paper *"Attention is All You Need"*. The model is trained on an English-Tamil parallel dataset and translates English text to Tamil.

---

## Dataset

The dataset used for training is sourced from [this Kaggle repository](https://www.kaggle.com/datasets/hemanthkumar21/englist-tamil-parallel-sent). It consists of English sentences paired with their Tamil translations, ideal for sequence-to-sequence modeling.

Key statistics of the dataset:
- Number of sentence pairs: Approximately 100,000
- Average sentence length: ~10 words
- Languages: English (source) and Tamil (target)

---

## Model Architecture

The model implements the Transformer architecture as described in the paper *"Attention is All You Need"*. Key components include:

1. **Encoder:** Processes the input English sentence and generates a sequence of contextual embeddings.
2. **Decoder:** Uses the encoder output and generates the translated Tamil sentence token by token.
3. **Attention Mechanisms:** Self-attention and cross-attention layers to capture contextual relationships.
4. **Tokenization:** Implement Byte Pair Encoding (BPE) for subword tokenization, using a vocabulary size of 8,000 for English and 16,000 for Tamil, reflecting Tamil's higher linguistic diversity..
