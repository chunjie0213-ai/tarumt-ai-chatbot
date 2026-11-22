# TAR UMT AI Chatbot

Desktop campus assistant chatbot for Tunku Abdul Rahman University of Management and Technology (TAR UMT). Built in Python using TF-IDF text vectorization, a TensorFlow/Keras Deep Neural Network (DNN) classifier, typo auto-correction, and a Tkinter desktop interface.

## Overview

The application answers campus queries about TAR UMT facilities, locations, and student services from a curated `intents.json` knowledge base. During development, I compared the Keras DNN against Support Vector Machine (SVM) and Naive Bayes baselines using 5-fold cross-validation and hyperparameter tuning.

## Evaluation results (5-fold CV)

Results from `code/model graph/cv_tuning_report.txt` on the best DNN configuration (`[256 -> 128 -> 64]`, dropout `[0.5 / 0.4 / 0.3]`, batch size `32`, learning rate `0.001`, label smoothing `0.1`):

- **Mean Accuracy:** 93.10% (+/- 0.59%)
- **Mean F1 Score:** 93.03% (+/- 0.62%)
- **Mean Precision:** 93.44% (+/- 0.59%)
- **Mean Recall:** 93.10% (+/- 0.59%)

![Model comparison chart](code/model%20graph/model_comparison_v2.png)

## Features

- **Intent classification:** TF-IDF vectorization paired with a trained TensorFlow/Keras neural network (`tarumt_chatbot_model.keras`)
- **Typo tolerance:** fuzzy word matching via `difflib` to recover misspelled user queries before inference
- **Desktop interface:** multithreaded Tkinter chat window with message timestamps and non-blocking model inference
- **Benchmarking scripts:** training, cross-validation tuning, baseline comparisons (SVM and Naive Bayes), and UAT scripts included under `code/`

## Project structure

```text
tarumt-ai-chatbot/
├── README.md
├── Chatbot Report.pdf                 Project report
└── code/
    ├── tarumtAiBot.py                 Main Tkinter desktop chatbot
    ├── DNN_model_train.py             Keras DNN training script
    ├── SVM_and_NB_model_train.py      SVM and Naive Bayes baseline training
    ├── cv_tuning.py                   5-fold cross-validation tuning
    ├── tfidf_algorithm.py             TF-IDF preprocessing helpers
    ├── UAT.py                         User acceptance testing script
    ├── intents.json                   Campus Q&A intents dataset
    ├── tarumt_chatbot_model.keras     Saved Keras model
    ├── tokenizer.pickle               Saved TF-IDF vectorizer
    ├── label_encoder.pickle           Saved label encoder
    └── model graph/                   Evaluation plots and CV reports
```

## Setup and usage

Requires Python 3.8 or newer (with Tkinter installed).

```bash
git clone https://github.com/chunjie0213-ai/tarumt-ai-chatbot.git
cd tarumt-ai-chatbot/code
pip install tensorflow numpy scikit-learn
```

Run the desktop chatbot using the pre-trained model artifacts:

```bash
python tarumtAiBot.py
```

Retrain the model after modifying `intents.json`:

```bash
python DNN_model_train.py
```

## Note

Built as an academic NLP and deep learning project at TAR UMT.
