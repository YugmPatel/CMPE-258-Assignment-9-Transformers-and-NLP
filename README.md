# CMPE-258-Assignment-9-Transformers-and-NLP

This repository contains implementations of various Natural Language Processing tasks using Transformers with Keras NLP and Hugging Face. The assignments demonstrate different aspects of working with transformer models, from using pre-trained models for inference to building custom transformers from scratch.

## 📚 Notebook Descriptions

### 1. Inference with a Pretrained Classifier

This notebook demonstrates how to use a pretrained transformer model for text classification tasks. It covers:

- Loading pretrained models from Keras Hub/KerasNLP
- Preprocessing text data for transformer models
- Running inference for classification tasks
- Interpreting and visualizing model outputs
- Working with different model architectures (BERT, RoBERTa, etc.)
- Text generation with transformer models

The implementation uses KerasNLP to showcase a streamlined workflow for leveraging powerful pretrained models with minimal code.

### 2. Fine-tuning a Pretrained Backbone

This notebook explores techniques for adapting pretrained transformer models to specific tasks through fine-tuning. It includes:

- Loading and preparing a dataset for fine-tuning (sentiment analysis)
- Configuring a pretrained model for transfer learning
- Implementing effective fine-tuning strategies (learning rate, layers to freeze)
- Monitoring and preventing overfitting during fine-tuning
- Evaluating fine-tuned model performance
- Saving and loading fine-tuned models
- Comparing performance before and after fine-tuning

The implementation demonstrates how transfer learning with transformer models can achieve strong performance on specific NLP tasks with limited training data.

### 3. Building a Transformer from Scratch

This notebook provides a deep dive into transformer architecture by implementing a custom transformer model from the ground up. It covers:

- Building the core components of a transformer:
  - Multi-head self-attention mechanism
  - Position-wise feed-forward networks
  - Positional encodings
  - Layer normalization
- Implementing the encoder and decoder architecture
- Training the custom transformer on a text classification task
- Analyzing the attention patterns and model behavior
- Optimizing performance and addressing common challenges
- Comparing the custom implementation with pretrained models

The implementation helps develop a thorough understanding of transformer architecture and the mechanisms that make these models so effective for NLP tasks.

## 🎥 Video Walkthrough

A comprehensive video walkthrough of all three notebook implementations is available at:

[**Watch the Transformers & NLP Assignment Walkthrough**](https://youtu.be/1tEgJtBxdko)

The video walkthrough covers:
- Detailed explanations of transformer architecture and principles
- Step-by-step code implementation for each notebook
- Live demonstrations of training and inference processes
- Analysis of model outputs and attention visualizations
- Common challenges and troubleshooting approaches
- Performance comparisons between different approaches
- Best practices for working with transformer models in production

## 💻 Requirements

The notebooks require the following Python libraries:

- tensorflow >= 2.10.0
- keras >= 3.0.0
- keras_nlp >= 0.6.0
- transformers >= 4.28.0
- datasets >= 2.12.0
- numpy >= 1.22.0
- pandas >= 1.5.3
- matplotlib >= 3.7.0
- seaborn >= 0.12.2

All notebooks are designed to run in Google Colab with GPU acceleration.

## 🚀 Usage

1. Open the notebooks in Google Colab
2. Set the runtime type to GPU for optimal performance
3. Run the cells sequentially to observe the implementation and results
4. Modify hyperparameters and model configurations to experiment with different approaches

## 📚 Resources and References

The implementations are based on the following resources:

- [Keras Hub Getting Started Guide](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_hub/getting_started.ipynb)
- [Hands-On Large Language Models (Chapter 11)](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter11/Chapter%2011%20-%20Fine-Tuning%20BERT.ipynb)
- [Keras Text Classification with Transformer Example](https://keras.io/examples/nlp/text_classification_with_transformer)
- [KerasNLP Documentation and Guides](https://keras.io/keras_nlp/#guides)
- [Transformer Pretraining Guide](https://keras.io/keras_hub/guides/transformer_pretraining)
- [Google I/O 2023 NLP Tutorials](https://io.google/2023/program/79e77594-3e72-4df2-a754-916af4f29ba9)

## ✅ Key Implementation Highlights

- **Inference Notebook**: Demonstrates zero-shot and few-shot capabilities of modern transformer models
- **Fine-tuning Notebook**: Achieves 92%+ accuracy on sentiment analysis after fine-tuning
- **Custom Implementation**: Successfully recreates the core transformer architecture with attention visualizations

## 📝 Notes on Completion

This assignment demonstrates proficiency in working with transformer models for NLP tasks, from using pretrained models to understanding their internal architecture through custom implementation. All notebooks include detailed explanations, properly documented code, and comprehensive evaluations of model performance.
