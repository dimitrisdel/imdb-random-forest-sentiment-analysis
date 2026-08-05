# IMDb Sentiment Analysis using a Custom Random Forest

## Overview

This project implements a custom Random Forest classifier in Python for sentiment analysis on the IMDb movie reviews dataset.

The implementation includes a custom ID3 decision tree algorithm, bootstrap sampling, Information Gain feature selection, and majority voting. The model is evaluated on the IMDb dataset and compared with the Random Forest implementation provided by scikit-learn.

---

## Features

- Custom Random Forest implementation
- Custom ID3 Decision Tree
- Information Gain based feature selection
- Bootstrap sampling for tree generation
- Binary Bag-of-Words text representation
- Automatic vocabulary creation from training data
- Sentiment classification of IMDb movie reviews
- Performance comparison with scikit-learn's Random Forest
- Evaluation using Accuracy, Precision, Recall and F1-score
- Learning curve generation for different training set sizes

---

## Technologies

- Python
- NumPy
- Pandas
- TensorFlow / Keras
- Scikit-learn
- Matplotlib

---

## Dataset

The project uses the IMDb Movie Reviews dataset available through TensorFlow/Keras.

The dataset is downloaded automatically when the program is executed for the first time.

---

## Project Workflow

1. Load the IMDb dataset.
2. Convert encoded reviews into text.
3. Build a custom vocabulary from the training data.
4. Transform reviews into binary feature vectors using Bag-of-Words.
5. Train a custom Random Forest classifier.
6. Predict the sentiment of unseen reviews.
7. Compare the results with scikit-learn's Random Forest implementation.
8. Display evaluation metrics and learning curves.

---

## Evaluation

The following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1-score

The project also generates learning curves showing the model's performance.
