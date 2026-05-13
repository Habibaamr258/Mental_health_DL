# Hybrid Multimodal Drug Recommendation System

## Overview

The Hybrid Multimodal Drug Recommendation System is an AI-powered healthcare project that combines Natural Language Processing (NLP) and Computer Vision techniques to analyze patient-related textual information and chest X-ray images in order to predict diseases and recommend suitable medications.

This project was designed as a multimodal deep learning system capable of integrating information from different medical data sources using a deep fusion neural network architecture.

The system combines:
- Text-based disease analysis
- Medical image classification
- Deep multimodal fusion
- Drug recommendation
- GUI-based interaction

The goal of this project is to explore how multimodal artificial intelligence can improve healthcare decision support systems.

---

# Problem Statement

Most traditional medical AI systems rely on only one data modality, such as text or medical images. However, real healthcare environments involve multiple types of medical information.

This project attempts to simulate a multimodal healthcare assistant capable of analyzing:
- Patient symptoms and textual medical data
- Chest X-ray images

and combining both sources to improve disease prediction and generate drug recommendations.

---

# Project Objectives

The main objectives of this project are:

- Scrape and preprocess medical datasets
- Build NLP-based disease classification models
- Build image classification models for X-ray analysis
- Implement multimodal fusion techniques
- Generate drug recommendations based on disease predictions
- Develop a GUI application for user interaction
- Explore deep learning techniques in healthcare AI

---

# Datasets

## Text Dataset
The textual dataset contains approximately:
- 600 patient medical records

The dataset includes:
- Symptoms
- Disease-related information
- Medical descriptions

The data was collected using web scraping and preprocessing techniques.

---

## Image Dataset
The image dataset contains approximately:
- 1200 chest X-ray images

The image dataset contains multiple disease classes:
- COVID-19
- Pneumonia
- Tuberculosis
- Lung Opacity
- Normal

---

# Dataset Challenges

One of the main challenges in this project was that:
- The textual records and X-ray images did not belong to the same patients.
- No real paired multimodal medical dataset was available.

To solve this problem, a:
# Synthetic Class-Based Pairing Strategy

was implemented during the fusion stage.

Each textual sample was paired with a randomly selected X-ray image from the same disease class.

---

# Data Preprocessing

## Text Preprocessing
The textual data preprocessing pipeline included:
- Lowercasing
- Stopword removal
- Tokenization
- Text cleaning
- Feature extraction
- Embedding generation

---

## Image Preprocessing
The image preprocessing pipeline included:
- Image resizing
- Normalization
- Data augmentation
- Noise reduction
- Class balancing

---

# Models Used

## NLP Models

Several NLP approaches were explored, including:
- DistilBERT
- BiLSTM
- Machine Learning classifiers

The best-performing NLP model was selected based on evaluation metrics.

---

## Image Classification Models

Multiple deep learning architectures were tested, including:
- EfficientNet
- Raisnet50
- Transfer Learning techniques

The image models were trained on chest X-ray datasets for disease classification.

---

# Multimodal Fusion

The final stage of the project uses:
# Deep Multimodal Fusion

instead of traditional weighted averaging.

The fusion model combines:
- Text embeddings
- Image embeddings

using:
- Concatenation
- Dense layers
- Batch normalization
- Dropout layers

to create a unified multimodal representation.

---

# Fusion Architecture

```text
Text Input
↓
Text Embeddings
↓
-------------------
        Fusion Layer
-------------------
↓
Image Embeddings
↓
Dense Layers
↓
Dropout
↓
Final Disease Prediction
↓
Drug Recommendation
