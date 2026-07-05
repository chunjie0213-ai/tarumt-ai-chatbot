# TAR UMT AI Chatbot

![TAR UMT AI Chatbot](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

A sleek, modern, deep-learning-powered desktop application providing intelligent, instant responses regarding Tunku Abdul Rahman University of Management and Technology (TAR UMT) facilities and services.

## 📖 Overview

The TAR UMT AI Chatbot leverages Natural Language Processing (NLP) and a Deep Neural Network (DNN) to understand and respond to user queries accurately. Built with Python and TensorFlow/Keras, it features a custom Tkinter Graphical User Interface (GUI) inspired by ChatGPT, designed for an intuitive and interactive user experience.

## ✨ Features

- **Intelligent NLP Engine:** Utilizes TF-IDF vectorization coupled with a trained Deep Neural Network for high-accuracy intent classification.
- **Interactive GUI:** A responsive, aesthetically pleasing desktop interface built entirely in Python using Tkinter.
- **Auto-Correction System:** Features built-in typo handling to intelligently match misspelled queries with the closest matching vocabulary words.
- **Campus Assistant:** Pre-trained on `intents.json` to promptly answer queries regarding TAR UMT campus facilities, locations, and essential services.

## 🛠️ Prerequisites

- **Python 3.8** or higher
- **TensorFlow / Keras** (for running the `.keras` model)
- **NumPy & Scikit-Learn** (for vectorization and prediction handling)

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/tarumt-ai-chatbot.git
   cd tarumt-ai-chatbot/code
   ```

2. **Install dependencies:**
   Make sure you have installed the required machine learning packages:
   ```bash
   pip install tensorflow numpy scikit-learn
   ```

3. **Train the Model (Optional):**
   If you wish to update the chatbot's knowledge by modifying `intents.json`, you will need to retrain the model before running the bot:
   ```bash
   python DNN_model_train.py
   ```

4. **Run the Chatbot:**
   Start the interactive GUI application:
   ```bash
   python tarumtAiBot.py
   ```

## 🧠 Model Architecture

The bot relies on a few key saved assets (generated from the training scripts):
- `tarumt_chatbot_model.keras` — The compiled deep learning model.
- `tokenizer.pickle` — The TF-IDF vectorizer mapping text to numerical vectors.
- `label_encoder.pickle` — The label encoder for matching model output to predefined intent tags.
- `intents.json` — The core dataset containing conversational patterns and expected responses.

## 📜 License

This project was built for educational and academic purposes.
