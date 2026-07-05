"""
TAR UMT AI Chatbot -- Cross-Validation Hyperparameter Tuning
=============================================================
Uses StratifiedKFold CV to evaluate different DNN hyperparameter
combinations and identify the best configuration.

Tuning parameters searched:
  - batch_size        : Training batch size
  - l2_reg            : L2 regularization strength
  - label_smoothing   : Label smoothing factor
  - learning_rate     : Adam optimizer learning rate
  - dropout_rates     : Dropout rates per hidden layer
  - architecture      : Hidden layer sizes

Outputs:
  - cv_tuning_results.csv   : Full results table
  - cv_tuning_report.txt    : Summary report with best parameters

Usage:
  python cv_tuning.py
"""

import csv
import itertools
import numpy as np
import os
import sys
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

from tfidf_algorithm import (
    load_intents,
    encode_labels,
    augment_data,
    create_tfidf_vectorizer,
    compute_class_weights,
    DEFAULT_MAX_WORDS,
    DEFAULT_NGRAM_RANGE,
)


# ============================================================
#  CROSS-VALIDATION SETTINGS
# ============================================================
N_FOLDS = 5                 # Number of CV folds
CV_EPOCHS = 150             # Max epochs per fold
CV_PATIENCE = 20            # EarlyStopping patience

# ============================================================
#  HYPERPARAMETER SEARCH SPACE
# ============================================================
PARAM_GRID = {
    "batch_size":       [32, 64, 128],
    "l2_reg":           [0.0005, 0.001, 0.005],
    "label_smoothing":  [0.05, 0.1, 0.15],
    "learning_rate":    [0.0005, 0.001, 0.002],
    "dropout_rates":    [
        (0.5, 0.4, 0.3),       # Current default
        (0.4, 0.3, 0.2),       # Lower dropout
        (0.6, 0.5, 0.4),       # Higher dropout
    ],
    "architecture":     [
        (512, 256, 128),        # Current default
        (256, 128, 64),         # Smaller
        (512, 512, 256),        # Wider
    ],
}

# To avoid combinatorial explosion (3^6 = 729 combos x 5 folds),
# we use a staged approach: first tune groups, then fine-tune the best.
# Set USE_FULL_GRID = True to run all combinations (very slow).
USE_FULL_GRID = False


# ============================================================
#  HELPER: Build model with given hyperparameters
# ============================================================
def build_model_cv(input_dim, num_classes, architecture, dropout_rates,
                   l2_reg, label_smoothing, learning_rate):
    """Build and compile a DNN with the given hyperparameter configuration."""
    layers = [Input(shape=(input_dim,))]

    for i, (units, drop) in enumerate(zip(architecture, dropout_rates)):
        layers.append(
            Dense(units, activation='relu',
                  kernel_regularizer=regularizers.l2(l2_reg))
        )
        layers.append(BatchNormalization())
        layers.append(Dropout(drop))

    layers.append(Dense(num_classes, activation='softmax'))

    model = Sequential(layers)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy'],
    )
    return model


# ============================================================
#  HELPER: Format time duration
# ============================================================
def format_time(seconds):
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {int(seconds % 60)}s"
    elif seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{seconds:.1f}s"


# ============================================================
#  STAGED TUNING: Tune one group at a time
# ============================================================
def generate_staged_configs():
    """
    Staged tuning strategy to reduce total experiments:
      Stage 1: Tune batch_size + learning_rate (3x3 = 9 combos)
      Stage 2: Tune l2_reg + label_smoothing   (3x3 = 9 combos, using Stage 1 best)
      Stage 3: Tune dropout + architecture      (3x3 = 9 combos, using Stage 1+2 best)
    Total: ~27 combos x 5 folds = 135 fits (manageable)
    """
    # Defaults to start with
    defaults = {
        "batch_size": 64,
        "l2_reg": 0.001,
        "label_smoothing": 0.1,
        "learning_rate": 0.001,
        "dropout_rates": (0.5, 0.4, 0.3),
        "architecture": (512, 256, 128),
    }

    stages = [
        # Stage 1: batch_size x learning_rate
        {
            "name": "Stage 1: batch_size x learning_rate",
            "vary": ["batch_size", "learning_rate"],
        },
        # Stage 2: l2_reg x label_smoothing
        {
            "name": "Stage 2: l2_reg x label_smoothing",
            "vary": ["l2_reg", "label_smoothing"],
        },
        # Stage 3: dropout_rates x architecture
        {
            "name": "Stage 3: dropout_rates x architecture",
            "vary": ["dropout_rates", "architecture"],
        },
    ]

    return defaults, stages


def generate_full_grid():
    """Generate all combinations from PARAM_GRID (can be very large!)."""
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


# ============================================================
#  CORE: Run K-Fold CV for a single configuration
# ============================================================
def run_cv_for_config(config, X_all, y_all, num_classes, class_weights,
                      fold_n=N_FOLDS, config_idx=0, total_configs=1):
    """
    Run StratifiedKFold CV for one hyperparameter configuration.
    Returns dict with mean/std of accuracy, F1, precision, recall.
    """
    skf = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_f1_scores = []
    fold_precisions = []
    fold_recalls = []

    config_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        X_train_fold = X_all[train_idx]
        X_val_fold = X_all[val_idx]
        y_train_fold = y_all[train_idx]
        y_val_fold = y_all[val_idx]

        # One-hot encode
        y_train_cat = tf.keras.utils.to_categorical(y_train_fold, num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val_fold, num_classes)

        # Build model
        model = build_model_cv(
            input_dim=X_train_fold.shape[1],
            num_classes=num_classes,
            architecture=config["architecture"],
            dropout_rates=config["dropout_rates"],
            l2_reg=config["l2_reg"],
            label_smoothing=config["label_smoothing"],
            learning_rate=config["learning_rate"],
        )

        # Train with early stopping
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=CV_PATIENCE,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=10, min_lr=1e-7, verbose=0),
        ]

        model.fit(
            X_train_fold, y_train_cat,
            epochs=CV_EPOCHS,
            batch_size=config["batch_size"],
            validation_data=(X_val_fold, y_val_cat),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0,
        )

        # Evaluate
        loss, accuracy = model.evaluate(X_val_fold, y_val_cat, verbose=0)
        y_pred = np.argmax(model.predict(X_val_fold, verbose=0), axis=1)

        f1 = f1_score(y_val_fold, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_val_fold, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_val_fold, y_pred, average='weighted', zero_division=0)

        fold_accuracies.append(accuracy)
        fold_f1_scores.append(f1)
        fold_precisions.append(prec)
        fold_recalls.append(rec)

        # Clear session to free memory
        tf.keras.backend.clear_session()

        elapsed = time.time() - config_start
        sys.stdout.write(
            f"\r    Config {config_idx + 1}/{total_configs} | "
            f"Fold {fold_idx + 1}/{fold_n} | "
            f"Acc: {accuracy * 100:.1f}% | F1: {f1 * 100:.1f}% | "
            f"Elapsed: {format_time(elapsed)}  "
        )
        sys.stdout.flush()

    config_time = time.time() - config_start
    print()  # newline after progress

    return {
        "mean_accuracy":   np.mean(fold_accuracies),
        "std_accuracy":    np.std(fold_accuracies),
        "mean_f1":         np.mean(fold_f1_scores),
        "std_f1":          np.std(fold_f1_scores),
        "mean_precision":  np.mean(fold_precisions),
        "std_precision":   np.std(fold_precisions),
        "mean_recall":     np.mean(fold_recalls),
        "std_recall":      np.std(fold_recalls),
        "fold_accuracies": fold_accuracies,
        "fold_f1_scores":  fold_f1_scores,
        "time_seconds":    config_time,
    }


# ============================================================
#  MAIN TUNING PIPELINE
# ============================================================
if __name__ == "__main__":
    total_start = time.time()

    print("=" * 70)
    print("  TAR UMT AI CHATBOT -- Cross-Validation Hyperparameter Tuning")
    print("=" * 70)
    print(f"  Started at:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CV Folds:     {N_FOLDS}")
    print(f"  Max Epochs:   {CV_EPOCHS} (with EarlyStopping patience={CV_PATIENCE})")
    print(f"  Strategy:     {'Full Grid Search' if USE_FULL_GRID else 'Staged Tuning (3 stages)'}")
    print("=" * 70)

    # -- Step 1: Load & preprocess data --
    print("\n[Step 1/4] Loading and preprocessing data...")
    sentences, labels, label_list, responses = load_intents('intents.json')
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    num_classes = len(label_list)
    print(f"  [OK] Loaded {len(sentences)} patterns across {num_classes} facility intents.")

    # -- Step 2: Augment + Vectorize --
    print("\n[Step 2/4] Augmenting and vectorizing (Reduced rounds for tuning)...")
    # For hyperparameter tuning, running large data augmentations is a bottleneck.
    # Set rounds=0 to use original data and dramatically speed up tuning. 
    aug_sentences, aug_labels = augment_data(sentences, encoded_labels, rounds=0)
    all_sentences = sentences + aug_sentences
    all_labels = np.concatenate([encoded_labels, np.array(aug_labels)])

    vectorizer, X_all = create_tfidf_vectorizer(all_sentences)
    class_weights = compute_class_weights(all_labels)

    print(f"  [OK] Total samples: {len(all_sentences)} ({len(sentences)} original + {len(aug_sentences)} augmented)")
    print(f"  [OK] Feature matrix: {X_all.shape[0]} x {X_all.shape[1]}")
    print(f"  [OK] Classes: {num_classes}")

    # -- Step 3: Run CV tuning --
    print("\n[Step 3/4] Running Cross-Validation tuning...")

    all_results = []   # List of (config_dict, metrics_dict)

    if USE_FULL_GRID:
        # -- FULL GRID SEARCH --
        configs = generate_full_grid()
        print(f"  Full grid: {len(configs)} configurations x {N_FOLDS} folds = {len(configs) * N_FOLDS} fits\n")

        for i, config in enumerate(configs):
            arch_str = "->".join(str(u) for u in config["architecture"])
            drop_str = "/".join(f"{d}" for d in config["dropout_rates"])
            print(f"  +-- Config {i + 1}/{len(configs)}: "
                  f"bs={config['batch_size']}, lr={config['learning_rate']}, "
                  f"l2={config['l2_reg']}, ls={config['label_smoothing']}, "
                  f"arch=[{arch_str}], drop=[{drop_str}]")

            metrics = run_cv_for_config(
                config, X_all, all_labels, num_classes, class_weights,
                config_idx=i, total_configs=len(configs),
            )
            all_results.append((config, metrics))

            print(f"  +-- Mean Acc: {metrics['mean_accuracy'] * 100:.2f}% +/- "
                  f"{metrics['std_accuracy'] * 100:.2f}% | "
                  f"Mean F1: {metrics['mean_f1'] * 100:.2f}% | "
                  f"Time: {format_time(metrics['time_seconds'])}\n")

    else:
        # -- STAGED TUNING --
        defaults, stages = generate_staged_configs()
        best_params = dict(defaults)

        for stage in stages:
            stage_name = stage["name"]
            vary_keys = stage["vary"]

            # Generate configs for this stage
            vary_values = [PARAM_GRID[k] for k in vary_keys]
            stage_configs = []
            for combo in itertools.product(*vary_values):
                config = dict(best_params)         # start from current best
                for k, v in zip(vary_keys, combo):
                    config[k] = v
                stage_configs.append(config)

            print(f"\n  {'-' * 60}")
            print(f"  {stage_name}")
            print(f"  Varying: {', '.join(vary_keys)}")
            print(f"  {len(stage_configs)} configs x {N_FOLDS} folds = {len(stage_configs) * N_FOLDS} fits")
            print(f"  {'-' * 60}\n")

            stage_results = []

            for i, config in enumerate(stage_configs):
                vary_str = ", ".join(f"{k}={config[k]}" for k in vary_keys)
                print(f"  +-- [{i + 1}/{len(stage_configs)}] {vary_str}")

                metrics = run_cv_for_config(
                    config, X_all, all_labels, num_classes, class_weights,
                    config_idx=i, total_configs=len(stage_configs),
                )
                stage_results.append((config, metrics))
                all_results.append((config, metrics))

                print(f"  +-- Mean Acc: {metrics['mean_accuracy'] * 100:.2f}% +/- "
                      f"{metrics['std_accuracy'] * 100:.2f}% | "
                      f"Mean F1: {metrics['mean_f1'] * 100:.2f}% | "
                      f"Time: {format_time(metrics['time_seconds'])}\n")

            # Find best config in this stage
            best_stage = max(stage_results, key=lambda x: x[1]["mean_accuracy"])
            best_config = best_stage[0]

            # Update best_params with this stage's winners
            for k in vary_keys:
                best_params[k] = best_config[k]

            vary_str = ", ".join(f"{k}={best_params[k]}" for k in vary_keys)
            print(f"  * Stage best: {vary_str} -> "
                  f"Acc={best_stage[1]['mean_accuracy'] * 100:.2f}%\n")

    # -- Step 4: Report results --
    print("\n" + "=" * 70)
    print("  [Step 4/4] CROSS-VALIDATION TUNING RESULTS")
    print("=" * 70)

    # Sort all results by mean accuracy
    all_results.sort(key=lambda x: x[1]["mean_accuracy"], reverse=True)

    # Overall best
    best_config, best_metrics = all_results[0]

    print(f"\n  * BEST CONFIGURATION (by Mean CV Accuracy):")
    print(f"  {'-' * 50}")
    print(f"  batch_size       = {best_config['batch_size']}")
    print(f"  learning_rate    = {best_config['learning_rate']}")
    print(f"  l2_reg           = {best_config['l2_reg']}")
    print(f"  label_smoothing  = {best_config['label_smoothing']}")
    arch_str = " -> ".join(str(u) for u in best_config['architecture'])
    print(f"  architecture     = [{arch_str}]")
    drop_str = " / ".join(str(d) for d in best_config['dropout_rates'])
    print(f"  dropout_rates    = [{drop_str}]")
    print(f"  {'-' * 50}")
    print(f"  Mean Accuracy:   {best_metrics['mean_accuracy'] * 100:.2f}% +/- {best_metrics['std_accuracy'] * 100:.2f}%")
    print(f"  Mean F1 Score:   {best_metrics['mean_f1'] * 100:.2f}% +/- {best_metrics['std_f1'] * 100:.2f}%")
    print(f"  Mean Precision:  {best_metrics['mean_precision'] * 100:.2f}% +/- {best_metrics['std_precision'] * 100:.2f}%")
    print(f"  Mean Recall:     {best_metrics['mean_recall'] * 100:.2f}% +/- {best_metrics['std_recall'] * 100:.2f}%")
    print(f"  Fold Accuracies: {['%.2f%%' % (a * 100) for a in best_metrics['fold_accuracies']]}")
    print(f"  CV Time:         {format_time(best_metrics['time_seconds'])}")

    # -- Top 5 table --
    print(f"\n\n  TOP {min(5, len(all_results))} CONFIGURATIONS:")
    print(f"  {'-' * 90}")
    print(f"  {'Rank':<5} {'Acc (mean+/-std)':>18} {'F1 (mean)':>10} {'BS':>4} {'LR':>8} {'L2':>8} {'LS':>6} {'Architecture':<18}")
    print(f"  {'-' * 90}")

    for rank, (cfg, met) in enumerate(all_results[:5], 1):
        arch = "->".join(str(u) for u in cfg['architecture'])
        print(f"  {rank:<5} "
              f"{met['mean_accuracy']*100:6.2f}% +/- {met['std_accuracy']*100:.2f}% "
              f"{met['mean_f1']*100:8.2f}% "
              f"{cfg['batch_size']:>4} "
              f"{cfg['learning_rate']:>8} "
              f"{cfg['l2_reg']:>8} "
              f"{cfg['label_smoothing']:>6} "
              f"{arch:<18}")

    print(f"  {'-' * 90}")

    # -- Save CSV --
    csv_path = "cv_tuning_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rank", "Mean_Accuracy", "Std_Accuracy",
            "Mean_F1", "Std_F1", "Mean_Precision", "Mean_Recall",
            "Batch_Size", "Learning_Rate", "L2_Reg", "Label_Smoothing",
            "Architecture", "Dropout_Rates", "Time_Seconds",
        ])
        for rank, (cfg, met) in enumerate(all_results, 1):
            writer.writerow([
                rank,
                f"{met['mean_accuracy']:.6f}", f"{met['std_accuracy']:.6f}",
                f"{met['mean_f1']:.6f}", f"{met['std_f1']:.6f}",
                f"{met['mean_precision']:.6f}", f"{met['mean_recall']:.6f}",
                cfg['batch_size'], cfg['learning_rate'],
                cfg['l2_reg'], cfg['label_smoothing'],
                str(cfg['architecture']), str(cfg['dropout_rates']),
                f"{met['time_seconds']:.1f}",
            ])
    print(f"\n  [OK] Full results saved to: {csv_path}")

    # -- Save text report --
    report_path = "cv_tuning_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  TAR UMT AI CHATBOT -- CV Hyperparameter Tuning Report\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("BEST CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  batch_size       = {best_config['batch_size']}\n")
        f.write(f"  learning_rate    = {best_config['learning_rate']}\n")
        f.write(f"  l2_reg           = {best_config['l2_reg']}\n")
        f.write(f"  label_smoothing  = {best_config['label_smoothing']}\n")
        f.write(f"  architecture     = [{arch_str}]\n")
        f.write(f"  dropout_rates    = [{drop_str}]\n\n")

        f.write("PERFORMANCE METRICS ({}-Fold CV)\n".format(N_FOLDS))
        f.write("-" * 40 + "\n")
        f.write(f"  Mean Accuracy:   {best_metrics['mean_accuracy'] * 100:.2f}% +/- {best_metrics['std_accuracy'] * 100:.2f}%\n")
        f.write(f"  Mean F1 Score:   {best_metrics['mean_f1'] * 100:.2f}% +/- {best_metrics['std_f1'] * 100:.2f}%\n")
        f.write(f"  Mean Precision:  {best_metrics['mean_precision'] * 100:.2f}% +/- {best_metrics['std_precision'] * 100:.2f}%\n")
        f.write(f"  Mean Recall:     {best_metrics['mean_recall'] * 100:.2f}% +/- {best_metrics['std_recall'] * 100:.2f}%\n")
        f.write(f"  Fold Accuracies: {['%.2f%%' % (a * 100) for a in best_metrics['fold_accuracies']]}\n\n")

        f.write(f"\nRECOMMENDED DNN_model_train.py SETTINGS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  EPOCHS          = 300\n")
        f.write(f"  BATCH_SIZE      = {best_config['batch_size']}\n")
        f.write(f"  LABEL_SMOOTHING = {best_config['label_smoothing']}\n")
        f.write(f"  L2_REG          = {best_config['l2_reg']}\n")
        f.write(f"  Learning Rate   = {best_config['learning_rate']}\n")
        arch_layers = best_config['architecture']
        drop_layers = best_config['dropout_rates']
        f.write(f"  Architecture    = {arch_layers}\n")
        f.write(f"  Dropout Rates   = {drop_layers}\n\n")

        f.write("  Paste into build_model():\n")
        f.write("  -------------------------\n")
        for i, (units, drop) in enumerate(zip(arch_layers, drop_layers)):
            f.write(f"    Dense({units}, activation='relu', kernel_regularizer=regularizers.l2({best_config['l2_reg']})),\n")
            f.write(f"    BatchNormalization(),\n")
            f.write(f"    Dropout({drop}),\n")
        f.write(f"    Dense(num_classes, activation='softmax'),\n")

    print(f"  [OK] Report saved to:       {report_path}")

    total_time = time.time() - total_start
    print(f"\n  Total tuning time: {format_time(total_time)}")

    print("\n" + "=" * 70)
    print("  TUNING COMPLETE!")
    print(f"  Best Mean CV Accuracy: {best_metrics['mean_accuracy'] * 100:.2f}%")
    print(f"  Update DNN_model_train.py with the recommended settings above.")
    print("=" * 70)
