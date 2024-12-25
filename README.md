# Aspect-Based-Sentiment-Analysis-Classifying-Key-Product-Attributes

# BERT Layer Freezing with Trainable LSTM for Text Classification

## Introduction
This project explores a hybrid approach to text classification using a pretrained BERT model where its layers are frozen, and a custom LSTM network is trained on top. By leveraging BERT's feature extraction capabilities and combining it with the sequential modeling power of LSTM, this approach aims to balance performance and computational efficiency. The notebook implements this architecture, processes datasets, and provides tools for fine-tuning.

## Description
The notebook focuses on:
- Freezing all layers of a pretrained BERT model.
- Training a custom LSTM network on top of BERT's output.
- Implementing multi-label classification for textual data.
- Visualizing the training process and evaluating performance on validation data.

This architecture is particularly useful when you want to utilize BERT's pretrained embeddings without the computational cost of fine-tuning its layers, while still capturing sequential dependencies in the data with an LSTM layer.

## Features
- **Data Preprocessing**: Tokenization using `BertTokenizer` and data transformation for multi-label classification.
- **Model Architecture**: Combines BERT as a feature extractor with a trainable Bidirectional LSTM.
- **Training Optimization**: Uses techniques like model checkpoints and validation splitting.
- **Visualization**: Includes plots for training loss and accuracy.

## Installation

### Dependencies
Ensure you have the following Python libraries installed:
