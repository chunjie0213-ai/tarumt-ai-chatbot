"""
TAR UMT AI Chatbot — TF-IDF Algorithm Module
===============================================
This file contains ONLY the TF-IDF text preprocessing pipeline:
  - Load and clean training data from intents.json
  - Encode labels
  - Split data (train / test)
  - Data augmentation (6 techniques)
  - TF-IDF vectorization (unigrams + bigrams)
  - Compute class weights

The DNN (neural network) is in DNN_model_train.py.

Usage:
    from tfidf_algorithm import (
        load_intents, encode_labels, split_data,
        augment_data, create_tfidf_vectorizer, compute_class_weights,
    )
"""

import json
import numpy as np
import random
import string

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
#  DEFAULT SETTINGS — TF-IDF Vectorizer Configuration
# ============================================================

# Maximum number of features (unique words/n-grams) to keep in the vocabulary
# Only the top 7000 most frequent terms will be retained after fitting
# A larger value captures more vocabulary but increases memory and risk of noise
DEFAULT_MAX_WORDS   = 7000

# N-gram range for the TF-IDF vectorizer
# (1, 2) means both unigrams (single words) and bigrams (two-word pairs) are used
# Example: 'sports complex' is captured as both ['sports', 'complex'] and ['sports complex']
# Trigrams (1, 3) were tested but caused overfitting on the relatively small dataset
DEFAULT_NGRAM_RANGE = (1, 2)       # Unigrams + Bigrams only (trigrams overfit on small data)


# ============================================================
#  DATA AUGMENTATION — 5 Text Transformation Techniques
# ============================================================
# Data augmentation synthetically creates new training samples by applying
# small transformations to existing sentences. This improves model robustness
# by exposing it to varied phrasings without collecting new real-world data.
# Each function takes a list of words and returns a modified sentence string.
# If the sentence is too short to meaningfully augment, None is returned.

def _augment_word_drop(words):
    """
    Randomly remove one word from the sentence.
    Purpose: Simulates users omitting words, e.g., 'what are library hours' → 'what library hours'
    Minimum length: 3 words (dropping from shorter sentences yields meaningless noise)
    """
    if len(words) < 3:  # Skip sentences that are too short to drop meaningfully
        return None
    idx = random.randint(0, len(words) - 1)  # Pick a random word position to remove
    return ' '.join(words[:idx] + words[idx+1:])  # Reconstruct sentence without that word


def _augment_word_swap(words):
    """
    Swap two randomly selected adjacent words.
    Purpose: Simulates different word order, e.g., 'booking sports' ↔ 'sports booking'
    Minimum length: 2 words (need at least two words to swap)
    """
    if len(words) < 2:  # Must have at least 2 words for swapping
        return None
    idx = random.randint(0, len(words) - 2)  # Pick position to swap with the next word
    swapped = words[:]                        # Make a copy to avoid mutating original
    swapped[idx], swapped[idx+1] = swapped[idx+1], swapped[idx]  # Swap adjacent words
    return ' '.join(swapped)


def _augment_truncate(words):
    """
    Keep only the first N words of the sentence (partial input simulation).
    Purpose: Simulates users sending incomplete queries, e.g., 'how do i borrow' instead of full sentence
    Minimum length: 4 words (need enough words to produce meaningful truncation)
    """
    if len(words) < 4:  # Skip very short sentences
        return None
    keep = random.randint(2, len(words) - 1)  # Randomly decide how many words to keep
    return ' '.join(words[:keep])             # Return only the first 'keep' words


def _augment_repeat_keyword(words):
    """
    Duplicate a random word within the sentence (emphasis simulation).
    Purpose: Simulates users emphasizing a keyword, e.g., 'library library hours'
    Minimum length: 2 words
    """
    if len(words) < 2:  # Must have at least 2 words
        return None
    idx = random.randint(0, len(words) - 1)             # Pick a random word
    new_words = words[:idx] + [words[idx], words[idx]] + words[idx+1:]  # Insert duplicate
    return ' '.join(new_words)


def _augment_random_insert(words):
    """
    Insert a word from the same sentence at a random position.
    Purpose: Adds controlled noise while keeping words relevant to the original intent.
    Minimum length: 3 words
    """
    if len(words) < 3:  # Must have enough words to pick and reinsert
        return None
    word_to_insert = random.choice(words)            # Pick a word already in the sentence
    pos = random.randint(0, len(words))              # Pick a random position to insert
    new_words = words[:pos] + [word_to_insert] + words[pos:]  # Insert and reconstruct
    return ' '.join(new_words)


# NOTE: _augment_shuffle was intentionally removed.
# Shuffling all words destroys bigram features (e.g., 'sports complex' becomes
# 'complex sports'), which are critical to our TF-IDF model. Word order matters
# for n-gram-based models, so random shuffling introduces harmful noise.

# Registry of all active augmentation functions
# Called in sequence during augment_data() for each training sample
_AUG_FUNCTIONS = [
    _augment_word_drop,
    _augment_word_swap,
    _augment_truncate,
    _augment_repeat_keyword,
    _augment_random_insert,
]


# ============================================================
#  PUBLIC API — TF-IDF Pipeline Functions
# ============================================================

def load_intents(filepath='intents.json'):
    """
    Load and preprocess all training patterns from the intents JSON file.

    The intents.json file contains a list of 'intent' objects, each with:
      - 'tag'      : The intent name (e.g., 'library_hours', 'citc_info')
      - 'patterns' : List of example user queries for this intent
      - 'responses': List of possible bot replies for this intent

    Text preprocessing applied to each pattern:
      1. Lowercase  — standardizes case (e.g., 'Library' → 'library')
      2. Strip punctuation — removes commas, question marks, etc.
      3. Normalize whitespace — removes extra spaces between words

    Parameters:
        filepath (str): Path to the intents JSON file (default: 'intents.json')

    Returns:
        sentences  (list[str]) : Cleaned training sentences
        labels     (list[str]) : Intent tag corresponding to each sentence
        label_list (list[str]) : Unique intent tags (order of first appearance)
        responses  (dict)      : Mapping of {tag: [response_strings]}
    """
    # Open the JSON file using UTF-8 encoding to support special characters
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)  # Parse the JSON structure into a Python dictionary

    sentences  = []   # Will hold all cleaned training patterns
    labels     = []   # Will hold the corresponding intent tag for each pattern
    label_list = []   # Will hold the unique list of all intent tags (deduplicated)
    responses  = {}   # Will map tag → list of possible response strings

    # Iterate through each intent entry in the intents JSON file
    for intent in data['intents']:
        for pattern in intent['patterns']:   # Each pattern is a training example
            # Step 1: Lowercase and strip leading/trailing whitespace
            cleaned = pattern.lower().strip()
            # Step 2: Remove all punctuation characters (! ? , . ' " etc.)
            cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
            # Step 3: Collapse multiple spaces into a single space
            cleaned = ' '.join(cleaned.split())
            sentences.append(cleaned)        # Store the cleaned pattern
            labels.append(intent['tag'])     # Store the matching intent label

        # Store all response strings for this intent (used at prediction time)
        responses[intent['tag']] = intent['responses']

        # Track unique intent tags in the order they appear
        if intent['tag'] not in label_list:
            label_list.append(intent['tag'])

    return sentences, labels, label_list, responses


def encode_labels(labels):
    """
    Convert string intent labels into integer class indices.

    The model cannot process string labels directly — they must be numeric.
    LabelEncoder assigns a unique integer to each unique label alphabetically.
    For example: ['library', 'citc', 'sports'] → [1, 0, 2]

    Parameters:
        labels (list[str]): List of intent tag strings (one per training pattern)

    Returns:
        encoder (LabelEncoder): Fitted encoder (saved for inverse transform later)
        encoded (np.ndarray):   Integer-encoded label array
    """
    encoder = LabelEncoder()              # Initialize the encoder
    encoded = encoder.fit_transform(labels)  # Fit on unique labels and transform to integers
    return encoder, encoded


def split_data(sentences, encoded_labels, test_size=0.20, random_state=42):
    """
    Split the dataset into training and testing subsets using stratified sampling.

    Stratification ensures the class distribution (proportion of each intent)
    is preserved in both the training and testing sets. This prevents the model
    from being evaluated on an unrepresentative subset.

    Parameters:
        sentences      (list[str])    : All cleaned training patterns
        encoded_labels (np.ndarray)   : Integer-encoded intent labels
        test_size      (float)        : Fraction of data for testing (default = 20%)
        random_state   (int)          : Random seed for reproducibility (default = 42)

    Returns:
        train_sentences (list[str])   : 80% of patterns used for training
        test_sentences  (list[str])   : 20% of patterns used for testing
        train_labels    (np.ndarray)  : Labels for training patterns
        test_labels     (np.ndarray)  : Labels for testing patterns
    """
    return train_test_split(
        sentences, encoded_labels,
        test_size=test_size,          # Reserve 20% for the test set
        random_state=random_state,    # Ensures the same split every run
        stratify=encoded_labels,      # Maintain class proportions in both sets
    )


def augment_data(train_sentences, train_labels, rounds=2):
    """
    Expand the training dataset by applying all 5 augmentation techniques.

    For each round, every training sentence is passed through each augmentation
    function. Only valid (non-None) outputs are added to the augmented dataset.
    The augmented data is SEPARATE from the original — it is appended to the
    original training set, not used as a replacement.

    This approach is only applied to training data — NEVER to the test set.
    Augmenting test data would cause data leakage and inflate accuracy metrics.

    Parameters:
        train_sentences (list[str]) : Original cleaned training sentences
        train_labels    (array-like): Corresponding integer-encoded labels
        rounds          (int)       : Number of augmentation passes (default = 2)
                                      Higher rounds = more data but slower training

    Returns:
        augmented_sentences (list[str]): Newly generated training samples
        augmented_labels    (list)     : Corresponding labels for new samples
    """
    augmented_sentences = []  # Collect all newly generated sentences
    augmented_labels    = []  # Collect corresponding labels for new sentences

    # Repeat augmentation for the specified number of rounds
    # Each round generates a fresh set of augmented samples from the originals
    for _ in range(rounds):
        for sent, label in zip(train_sentences, train_labels):
            words = sent.split()          # Split sentence into list of words
            if len(words) < 2:            # Skip single-word sentences (can't augment meaningfully)
                continue
            for aug_fn in _AUG_FUNCTIONS: # Apply each augmentation technique
                result = aug_fn(words)    # Get augmented sentence (or None if skipped)
                if result:                # Only add if the augmentation returned a valid string
                    augmented_sentences.append(result)
                    augmented_labels.append(label)  # New sample inherits the same intent label

    return augmented_sentences, augmented_labels


def create_tfidf_vectorizer(sentences, max_words=DEFAULT_MAX_WORDS,
                             ngram_range=DEFAULT_NGRAM_RANGE):
    """
    Fit a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer
    and convert sentences into a numerical feature matrix.

    TF-IDF assigns a weight to each word based on:
      - TF  (Term Frequency): How often the word appears in a sentence
      - IDF (Inverse Doc Freq): How rare the word is across all sentences
    Words that appear frequently in many documents (e.g., 'the', 'is') get
    lower weights, while domain-specific words (e.g., 'arena', 'badminton')
    get higher weights.

    Vectorizer settings explained:
      max_features  : Keep only the top 7000 most informative features
      ngram_range   : Include unigrams AND bigrams (e.g., 'sports', 'sports complex')
      sublinear_tf  : Apply log(1 + TF) to reduce the dominance of repeated words
      min_df        : Include a word if it appears in at least 1 document
      max_df        : Ignore words that appear in more than 95% of all documents
                      (these are too common to be informative, like stop words)
      strip_accents : Normalize accented characters (e.g., 'café' → 'cafe')

    Parameters:
        sentences   (list[str]) : Training sentences to fit the vectorizer on
        max_words   (int)       : Maximum vocabulary size
        ngram_range (tuple)     : N-gram range, e.g. (1, 2) for unigrams + bigrams

    Returns:
        vectorizer (TfidfVectorizer) : Fitted vectorizer (saved for transforming new inputs)
        X          (np.ndarray)      : Dense TF-IDF feature matrix (n_samples x max_words)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_words,        # Limit vocabulary to top N most important terms
        ngram_range=ngram_range,       # Capture both single words and word pairs
        sublinear_tf=True,             # Use log-scaled TF to reduce impact of frequent terms
        min_df=1,                      # Include word even if it appears in just 1 sentence
        max_df=0.95,                   # Exclude words appearing in >95% of sentences (too common)
        strip_accents='unicode',       # Normalize unicode/accented characters 
    )
    # fit_transform: learns the vocabulary AND transforms sentences into vectors
    # .toarray(): converts sparse matrix to dense numpy array (required by DNN)
    X = vectorizer.fit_transform(sentences).toarray()
    return vectorizer, X


def compute_class_weights(labels):
    """
    Compute balanced class weights to address class imbalance in training data.

    In the TAR UMT dataset, some intents have many more training patterns than
    others (e.g., 'library' may have 30 patterns, 'clubhouse' may have only 8).
    Without correction, the model would be biased toward frequent classes.

    'balanced' mode computes weights as:
        weight[i] = total_samples / (n_classes * count_of_class_i)
    This gives higher weight to underrepresented classes during training.

    Parameters:
        labels (array-like): Integer-encoded training labels

    Returns:
        class_weight_dict (dict): {class_index: weight_value}
                                  Passed directly to model.fit(class_weight=...)
    """
    weights = compute_class_weight(
        class_weight='balanced',       # Automatically infer weights for all classes
        classes=np.unique(labels),     # All unique class indices present in labels
        y=labels,                      # The full label array to count class frequencies
    )
    # Convert numpy array to dictionary format expected by Keras: {0: w0, 1: w1, ...}
    return dict(enumerate(weights))