"""
TAR UMT AI Chatbot — DNN Training Script (Facility Information)
================================================================
This file contains the DNN (Deep Neural Network) algorithm:
  - Build the neural network model
  - Train with callbacks (EarlyStopping, LR reduction, progress bar)
  - Evaluate performance (accuracy, F1, precision, recall)
  - Generate visual reports (via graph_generator.py)
  - Save trained model + TF-IDF vectorizer + label encoder

The dataset (intents.json) focuses on TAR UMT facility information:
  Library, CITC, Sports Complex, Clubhouse, Arena, and related rules.

The TF-IDF preprocessing is imported from tfidf_algorithm.py.
The charts are imported from graph_generator.py.

Usage:
  python DNN_model_train.py
"""

import numpy as np       # For numerical array operations (argmax, concatenation, etc.)
import pickle             # For serializing and saving the TF-IDF vectorizer and label encoder
import sys                # For sys.stdout.write to print progress bar on the same line
import time               # For measuring and displaying elapsed training time

import tensorflow as tf   # Main deep learning framework for building and training the DNN
from tensorflow.keras.models import Sequential          # Linear stack of neural network layers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
# Dense          — Fully connected (linear) layer with activation
# Dropout        — Randomly disables neurons to prevent overfitting
# BatchNormalization — Normalizes layer inputs to stabilize and speed up training
# Input          — Explicitly defines the shape of the input to the model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
# EarlyStopping  — Stops training when validation accuracy stops improving
# ReduceLROnPlateau — Reduces learning rate when validation loss plateaus
# Callback       — Base class for creating custom training callbacks
from tensorflow.keras import regularizers               # L2 weight regularization to reduce overfitting
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
# classification_report — Generates Precision/Recall/F1 per intent class
# roc_auc_score         — Computes area under the ROC curve (one-vs-rest)
# confusion_matrix      — Shows predicted vs actual class counts (used for Specificity)

# Import preprocessing utilities from the TF-IDF module
# Keeping these separate promotes modularity and code reuse across files
from tfidf_algorithm import (
    load_intents,              # Load and clean patterns from intents.json
    encode_labels,             # Convert string intent labels to integers
    split_data,                # Stratified 80/20 train/test split
    augment_data,              # Text augmentation to expand training set
    create_tfidf_vectorizer,   # Build TF-IDF feature matrix
    compute_class_weights,     # Handle class imbalance with weighted training
    DEFAULT_MAX_WORDS,         # Vocabulary size constant (7000)
    DEFAULT_NGRAM_RANGE,       # N-gram range constant (1, 2)
)

# Import chart generation functions from the visualization module
from graph_generator import (
    plot_training_history,   # Training accuracy and loss curves over epochs
    plot_confusion_matrix,   # Heatmap of predicted vs actual intent classes
    plot_f1_scores,          # Per-class F1 score bar chart
    plot_overall_metrics,    # Dashboard of all weighted average metrics + pie chart
)


# ============================================================
#  DNN TRAINING SETTINGS
# ============================================================
EPOCHS          = 300    # Maximum number of training iterations (EarlyStopping may stop earlier)
BATCH_SIZE      = 32     # Number of samples per gradient update (smaller = noisier but more stable)
LABEL_SMOOTHING = 0.1   # Prevents overconfidence by softening one-hot targets (0.9/0.1 instead of 1/0)
L2_REG          = 0.001  # L2 regularization penalty to reduce overfitting by penalizing large weights


# ============================================================
#  CUSTOM CALLBACK 
# ============================================================
# ============================================================
#  CUSTOM KERAS CALLBACK — Live Training Progress Bar
# ============================================================
class TrainingProgressBar(Callback):
    """
    A custom Keras Callback that displays a real-time progress bar in the terminal.
    Shows epoch progress, training/validation accuracy, loss, and estimated time remaining (ETA).

    This replaces Keras's default verbose output with a single-line progress bar
    that updates in-place (using \r carriage return), making it much easier to monitor.

    Triggered events:
      - on_train_begin : Runs once at the very start of training
      - on_epoch_end   : Runs after every epoch completes
      - on_train_end   : Runs once when training finishes or early stops
    """
    def on_train_begin(self, logs=None):
        """Initialize tracking variables and print a training start banner."""
        self.total_epochs = self.params['epochs']   # Total configured epoch count
        self.start_time   = time.time()             # Record training start timestamp
        self.best_val_acc = 0                       # Track best validation accuracy seen so far
        print("\n" + "=" * 64)
        print("  TRAINING IN PROGRESS — DO NOT CLOSE THIS WINDOW")
        print("=" * 64)

    def on_epoch_end(self, epoch, logs=None):
        """Update and redraw the progress bar after each epoch."""
        current = epoch + 1                         # Convert 0-indexed epoch to 1-indexed
        percent = (current / self.total_epochs) * 100  # Percentage of training completed
        elapsed = time.time() - self.start_time     # Time elapsed since training began

        # Calculate estimated time remaining (ETA)
        avg_time_per_epoch = elapsed / current                  # Average seconds per epoch
        remaining_epochs   = self.total_epochs - current        # Epochs still to go
        eta_seconds        = avg_time_per_epoch * remaining_epochs  # Estimated seconds left

        # Format ETA string (e.g., '2m 30s' or '45s')
        if eta_seconds >= 60:
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
        else:
            eta_str = f"{int(eta_seconds)}s"

        # Format elapsed time string
        if elapsed >= 60:
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        else:
            elapsed_str = f"{int(elapsed)}s"

        # Build the ASCII progress bar: '████░░░░░░' style
        bar_length = 30                             # Total number of characters in the bar
        filled     = int(bar_length * current / self.total_epochs)  # Filled portion
        bar        = "█" * filled + "░" * (bar_length - filled)    # Combine filled and empty

        # Extract metric values from the logs dictionary for this epoch
        acc      = logs.get('accuracy', 0) * 100      # Training accuracy (%)
        val_acc  = logs.get('val_accuracy', 0) * 100  # Validation accuracy (%)
        loss     = logs.get('loss', 0)                # Training loss
        val_loss = logs.get('val_loss', 0)            # Validation loss

        # Track the best validation accuracy seen so far
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

        # Write the progress bar on the same line (\r returns cursor to line start)
        sys.stdout.write(
            f"\r  [{bar}] {percent:5.1f}% | "
            f"Epoch {current}/{self.total_epochs} | "
            f"Acc: {acc:.1f}% | Val: {val_acc:.1f}% | "
            f"Loss: {loss:.4f} | ETA: {eta_str}  "
        )
        sys.stdout.flush()  # Ensure the progress bar is immediately displayed

        # Every 25 epochs (and epoch 1), print a full snapshot line for reference
        if current % 25 == 0 or current == 1:
            lr = float(self.model.optimizer.learning_rate)  # Get current learning rate
            print(
                f"\n  ├─ Epoch {current:>4}/{self.total_epochs} │ "
                f"Train Acc: {acc:6.2f}% │ Val Acc: {val_acc:6.2f}% │ "
                f"Loss: {loss:.4f} / {val_loss:.4f} │ "
                f"LR: {lr:.6f} │ Elapsed: {elapsed_str}"
            )

    def on_train_end(self, logs=None):
        """Print a training completion summary with total time and best validation accuracy."""
        total_time = time.time() - self.start_time
        if total_time >= 60:
            time_str = f"{int(total_time // 60)}m {int(total_time % 60)}s"
        else:
            time_str = f"{total_time:.1f}s"
        print(f"\n  └─ Training complete in {time_str} | Best Val Acc: {self.best_val_acc:.2f}%")
        print("=" * 64 + "\n")


# ============================================================
#  DNN FUNCTIONS
# ============================================================

def build_model(input_dim, num_classes, l2_reg=L2_REG,
                label_smoothing=LABEL_SMOOTHING):
    """
    Build and compile a regularized Deep Neural Network (DNN) for intent classification.

    Architecture: Input → Dense(256) → BN → Dropout → Dense(128) → BN → Dropout
                       → Dense(64)  → BN → Dropout → Output(num_classes, softmax)

    Design decisions:
      - ReLU activation  : Fast to train, avoids vanishing gradient in hidden layers
      - BatchNormalization: Normalizes layer inputs to reduce internal covariate shift
      - Dropout          : Randomly disables neurons to prevent overfitting
      - Decreasing units : 256 → 128 → 64 creates a compression bottleneck that forces
                           the model to learn abstract, generalizable intent features
      - L2 regularization: Penalizes large weights to discourage memorization
      - Softmax output   : Converts raw scores to probability distribution (sums to 1)
      - Label smoothing  : Softens targets (e.g., 0.9 instead of 1.0) to reduce overconfidence
      - Adam optimizer   : Adaptive learning rate algorithm, well-suited for NLP tasks

    Parameters:
        input_dim       (int)   : Number of TF-IDF features (vocabulary size)
        num_classes     (int)   : Number of unique intent classes to predict
        l2_reg          (float) : L2 regularization strength (default = 0.001)
        label_smoothing (float) : Label smoothing factor (default = 0.1)

    Returns:
        model (Sequential): Compiled Keras model ready for training
    """
    model = Sequential([
        Input(shape=(input_dim,)),   # Explicitly define input shape for the model

        # Hidden Layer 1: 256 neurons with L2 regularization
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        BatchNormalization(),         # Normalize activations to stabilize training
        Dropout(0.3),                 # Drop 30% of neurons randomly during training

        # Hidden Layer 2: 128 neurons with L2 regularization
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        BatchNormalization(),
        Dropout(0.2),                 # Drop 20% of neurons (less aggressive as we go deeper)

        # Hidden Layer 3: 64 neurons with L2 regularization
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        BatchNormalization(),
        Dropout(0.1),                 # Drop only 10% in the final hidden layer

        # Output Layer: one neuron per intent class, softmax gives probability distribution
        Dense(num_classes, activation='softmax'),
    ])

    # Compile the model with categorical cross-entropy loss and Adam optimizer
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing  # Smooth one-hot labels to reduce overconfidence
        ),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Adaptive learning rate
        metrics=['accuracy'],  # Track accuracy during training and validation
    )
    return model


def train_model(model, X_train, train_labels, X_test, test_labels, num_classes,
                epochs=EPOCHS, batch_size=BATCH_SIZE, class_weights=None):
    """
    Train the DNN model with EarlyStopping, ReduceLROnPlateau, and a live progress bar.

    Callbacks used:
      1. EarlyStopping: Monitors 'val_accuracy'. Stops training if there's no improvement
         for 40 consecutive epochs. Restores the best weights found during training.
         This prevents wasting time on epochs after the model has already peaked.

      2. ReduceLROnPlateau: Monitors 'val_loss'. Halves the learning rate (factor=0.5)
         when val_loss doesn't improve for 15 epochs. Min LR = 1e-7.
         This helps the model escape shallow plateaus and fine-tune more precisely.

      3. TrainingProgressBar: Custom callback that renders a single-line progress bar
         with live accuracy, loss, and ETA.

    Parameters:
        model        : Compiled Keras model from build_model()
        X_train      : TF-IDF feature matrix for training (n_train x vocab_size)
        train_labels : Integer-encoded training labels (1D array)
        X_test       : TF-IDF feature matrix for testing (n_test x vocab_size)
        test_labels  : Integer-encoded test labels (1D array)
        num_classes  : Number of unique intent classes
        epochs       : Max number of training iterations (default = 300)
        batch_size   : Samples per mini-batch (default = 32)
        class_weights: Dict of {class: weight} for handling class imbalance

    Returns:
        history: Keras History object containing per-epoch loss and accuracy values
    """
    # Convert integer labels to one-hot encoded format (required by CategoricalCrossentropy)
    # Example: label '2' with 5 classes → [0, 0, 1, 0, 0]
    y_train = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
    y_test  = tf.keras.utils.to_categorical(test_labels,  num_classes=num_classes)

    # Configure training callbacks
    callbacks = [
        # Stop training if validation accuracy doesn't improve for 40 epochs
        EarlyStopping(
            monitor='val_accuracy',
            patience=40,                  # Wait 40 epochs before stopping
            restore_best_weights=True,    # Revert to the epoch with the best val_accuracy
            verbose=0                     # Suppress default Keras output (we use our own bar)
        ),
        # Reduce learning rate when val_loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,                   # Multiply LR by 0.5 (halve it)
            patience=15,                  # Wait 15 epochs before reducing
            min_lr=1e-7,                  # Never go below this learning rate
            verbose=0
        ),
        TrainingProgressBar(),            # Custom live progress bar
    ]

    # Run model training
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),  # Evaluate on test set at the end of each epoch
        callbacks=callbacks,
        class_weight=class_weights,        # Apply class weights for imbalanced intents
        verbose=0,                         # Suppress default per-epoch output
    )
    return history


def evaluate_model(model, X_test, test_labels, encoder, num_classes):
    """
    Evaluate the trained DNN on the held-out test set and print a full metrics report.

    Metrics computed:
      - Accuracy          : Overall test accuracy
      - Classification report: Per-intent Precision, Recall, F1 (via sklearn)
      - Weighted F1, Precision, Recall averages
      - AUC (Macro, One-vs-Rest): Area Under ROC Curve for multi-class
      - Specificity (Macro):      True Negative Rate, derived from confusion matrix

    Parameters:
        model       : Trained Keras model
        X_test      : TF-IDF feature matrix for test data
        test_labels : True integer-encoded labels for test set
        encoder     : Fitted LabelEncoder for converting integer → intent tag name
        num_classes : Total number of intent classes

    Returns:
        report          (dict)         : Full classification report including AUC and Specificity
        y_true_classes  (np.ndarray)   : True class indices for all test samples
        y_pred_classes  (np.ndarray)   : Predicted class indices for all test samples
        target_names    (list[str])    : Intent tag names (for chart axis labels)
    """
    # Convert integer test labels to one-hot format for model evaluation
    y_test = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

    # Evaluate the model's loss and accuracy on the full test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n{'=' * 64}")
    print(f"  ★  FINAL TEST ACCURACY: {accuracy * 100:.2f}%")
    print(f"  ★  FINAL TEST LOSS:     {loss:.4f}")
    print(f"{'=' * 64}")

    # Get per-sample predictions from the DNN (shape: n_samples x num_classes probabilities)
    y_pred         = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)            # Convert probabilities to class index
    y_true_classes = np.argmax(y_test, axis=1)            # Convert one-hot back to class index

    # Decode integer class indices back to intent tag names (e.g., 0 → 'library_hours')
    target_names = encoder.inverse_transform(sorted(set(y_true_classes)))

    # Print the full sklearn classification report (Precision/Recall/F1 per class)
    print("\n" + "=" * 64)
    print("  CLASSIFICATION REPORT (F1 / Precision / Recall)")
    print("=" * 64)
    print(classification_report(
        y_true_classes, y_pred_classes,
        target_names=target_names, zero_division=0  # Avoid errors for classes with no predictions
    ))

    # Get report as a dictionary for programmatic access and charting
    report = classification_report(
        y_true_classes, y_pred_classes,
        target_names=target_names, output_dict=True, zero_division=0
    )
    print(f"  OVERALL WEIGHTED F1 SCORE:  {report['weighted avg']['f1-score'] * 100:.2f}%")
    print(f"  OVERALL WEIGHTED PRECISION: {report['weighted avg']['precision'] * 100:.2f}%")
    print(f"  OVERALL WEIGHTED RECALL:    {report['weighted avg']['recall'] * 100:.2f}%")

    # --- AUC Calculation (One-vs-Rest, Macro Average) ---
    # Each class is treated as binary (class i vs all other classes)
    # Macro average gives equal weight to all classes regardless of sample count
    auc_score = roc_auc_score(y_test, y_pred, multi_class='ovr', average='macro')
    report['auc'] = auc_score  # Store AUC in report dict for use in visualization
    print(f"  OVERALL MACRO AUC SCORE:    {auc_score * 100:.2f}%")

    # --- Specificity Calculation (True Negative Rate, Macro Average) ---
    # Specificity = TN / (TN + FP) for each class, then averaged across all classes
    # This measures how well the model avoids incorrectly predicting a class
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    specificities = []
    for i in range(len(cm)):
        tp = cm[i, i]                        # True Positives: correctly predicted as class i
        fp = cm[:, i].sum() - tp             # False Positives: others wrongly classified as class i
        fn = cm[i, :].sum() - tp             # False Negatives: class i samples classified as other
        tn = cm.sum() - (tp + fp + fn)       # True Negatives: all other correct non-class-i predictions
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0  # Guard against division by zero
        specificities.append(spec)

    macro_spec = np.mean(specificities)       # Calculate macro-average specificity
    report['specificity'] = macro_spec        # Store in report dict
    print(f"  OVERALL MACRO SPECIFICITY:  {macro_spec * 100:.2f}%")
    print(f"{'=' * 64}\n")

    return report, y_true_classes, y_pred_classes, target_names


def save_assets(model, vectorizer, encoder,
                model_path='tarumt_chatbot_model.keras',
                vectorizer_path='tokenizer.pickle',
                encoder_path='label_encoder.pickle'):
    """
    Persist the three trained artifacts that are required to run the chatbot.

    Files saved:
      - tarumt_chatbot_model.keras : The trained DNN model weights and architecture
      - tokenizer.pickle           : The fitted TF-IDF vectorizer (must match training)
      - label_encoder.pickle       : The label encoder for converting integer → intent name

    These three files are loaded by tarumtAiBot.py at runtime to make predictions.
    If any file is missing, the chatbot will refuse to start.

    Parameters:
        model          : Trained Keras Sequential model
        vectorizer     : Fitted TfidfVectorizer from tfidf_algorithm.py
        encoder        : Fitted sklearn LabelEncoder from tfidf_algorithm.py
        model_path     : Output path for the Keras model file
        vectorizer_path: Output path for the vectorizer pickle file
        encoder_path   : Output path for the label encoder pickle file
    """
    # Save the Keras model in .keras format (recommended for TF 2.x)
    model.save(model_path)

    # Serialize and save the TF-IDF vectorizer using Python's pickle module
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    # Serialize and save the label encoder using Python's pickle module
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)

    print(f"  ✓ Model saved:       {model_path}")
    print(f"  ✓ Vectorizer saved:  {vectorizer_path}")
    print(f"  ✓ Encoder saved:     {encoder_path}")


# ============================================================
#  MAIN TRAINING PIPELINE
# ============================================================
if __name__ == "__main__":

    # 1. LOAD DATA (TF-IDF module)
    print("=" * 64)
    print("  TAR UMT AI CHATBOT — Facility Info Training Pipeline v2.0")
    print("=" * 64)
    print("\n[Step 1/11] Loading TAR UMT facility dataset...")
    sentences, labels, label_list, responses = load_intents('intents.json')
    print(f"  ✓ Loaded {len(sentences)} patterns across {len(label_list)} facility intents.")

    # 2. ENCODE LABELS (TF-IDF module)
    print("\n[Step 2/11] Encoding labels...")
    encoder, encoded_labels = encode_labels(labels)
    print(f"  ✓ Encoded {len(label_list)} unique intent classes.")

    # 3. SPLIT DATA (TF-IDF module)
    print("\n[Step 3/11] Splitting data (80% train / 20% test, stratified)...")
    train_sentences, test_sentences, train_labels, test_labels = split_data(sentences, encoded_labels)
    print(f"  ✓ Train set: {len(train_sentences)} original patterns")
    print(f"  ✓ Test set:  {len(test_sentences)} original patterns (clean, untouched)")

    # 4. DATA AUGMENTATION (TF-IDF module)
    print("\n[Step 4/11] Augmenting training data (3 rounds × 5 techniques)...")
    aug_sentences, aug_labels = augment_data(train_sentences, train_labels, rounds=3)
    all_train_sentences = train_sentences + aug_sentences
    all_train_labels = np.concatenate([train_labels, np.array(aug_labels)])
    print(f"  ✓ Train set after augmentation: {len(all_train_sentences)} total samples")
    print(f"    (Original: {len(train_sentences)} + Augmented: {len(aug_sentences)})")

    # 5. TF-IDF VECTORIZATION (TF-IDF module)
    print(f"\n[Step 5/11] Vectorizing with TF-IDF (vocab={DEFAULT_MAX_WORDS}, ngrams={DEFAULT_NGRAM_RANGE})...")
    vectorizer, X_train = create_tfidf_vectorizer(all_train_sentences)
    X_test = vectorizer.transform(test_sentences).toarray()

    num_classes = len(label_list)
    actual_features = X_train.shape[1]
    print(f"  ✓ Feature matrix: {X_train.shape[0]} x {actual_features}")
    print(f"  ✓ Classes: {num_classes}")
    print(f"  ✓ Bigrams enabled: word pairs like 'sports complex', 'library booking' captured")

    # 5b. COMPUTE CLASS WEIGHTS (TF-IDF module)
    print("\n  Computing class weights for imbalanced facility intents...")
    class_weights = compute_class_weights(all_train_labels)
    print(f"  ✓ Class weights computed")

    # 6. BUILD DNN MODEL
    print(f"\n[Step 6/11] Building DNN (Deep Neural Network)...")
    model = build_model(actual_features, num_classes)
    print(f"  ✓ Architecture: Input({actual_features}) → 256 → 128 → 64 → {num_classes}")
    print(f"  ✓ Total parameters: {model.count_params():,}")
    print(f"  ✓ Regularization: L2={L2_REG}, Dropout=0.3/0.2/0.1")
    print(f"  ✓ Label smoothing: {LABEL_SMOOTHING}")

    # 7. CALLBACKS
    print("\n[Step 7/11] Configuring callbacks...")
    print(f"  ✓ EarlyStopping: patience=40, monitor=val_accuracy")
    print(f"  ✓ ReduceLROnPlateau: patience=15, factor=0.5, min_lr=1e-7")
    print(f"  ✓ Custom Progress Bar: enabled")

    # 8. TRAIN DNN
    print("\n[Step 8/11] Training DNN model...")
    history = train_model(
        model, X_train, all_train_labels, X_test, test_labels,
        num_classes=num_classes,
        class_weights=class_weights,
    )

    # 9. EVALUATE DNN
    print("[Step 9/11] Evaluating DNN on clean test set...")
    report, y_true_classes, y_pred_classes, target_names = evaluate_model(
        model, X_test, test_labels, encoder, num_classes
    )

    # 10. GENERATE VISUAL REPORTS (graph_generator module)
    print("[Step 10/11] Generating visual reports...")
    print("  Generating training history chart...")
    plot_training_history(history)

    print("  Generating confusion matrix heatmap...")
    plot_confusion_matrix(y_true_classes, y_pred_classes, target_names)

    print("  Generating F1 score comparison chart...")
    plot_f1_scores(report, target_names)

    print("  Generating overall metrics dashboard...")
    plot_overall_metrics(report)

    print("\n  ✓ All charts saved successfully!")
    print("    • training_history.png     — Accuracy & Loss curves")
    print("    • confusion_matrix.png     — Per-class prediction heatmap")
    print("    • f1_scores_chart.png      — F1 score bar chart (all intents)")
    print("    • overall_metrics.png      — Dashboard + Pie chart")

    # 11. SAVE ASSETS (TF-IDF vectorizer + DNN model + encoder)
    print("\n[Step 11/11] Saving TF-IDF + DNN assets...")
    save_assets(model, vectorizer, encoder)

    print("\n" + "=" * 64)
    print("  TRAINING COMPLETE — All assets saved successfully!")
    print("  To chat with the bot, run: python tarumtAiBot.py")
    print("=" * 64)
