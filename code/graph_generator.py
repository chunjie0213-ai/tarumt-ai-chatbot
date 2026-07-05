"""
TAR UMT AI Chatbot — Graph Generator Module
=============================================
Contains all visualization functions for model evaluation:
  - Training history (accuracy & loss curves)
  - Confusion matrix heatmap
  - Per-class F1 score bar chart
  - Overall metrics dashboard

Usage:
    from graph_generator import (
        plot_training_history,
        plot_confusion_matrix,
        plot_f1_scores,
        plot_overall_metrics,
    )
"""

import numpy as np                          # For numerical operations (argmax, argsort, etc.)
import matplotlib.pyplot as plt             # Core plotting library for creating all charts
import seaborn as sns                       # For styling the confusion matrix heatmap
from sklearn.metrics import confusion_matrix  # Computes the confusion matrix from true/predicted labels


# ============================================================
#  VISUALIZATION FUNCTIONS — Charts and Graphs
# ============================================================
# All functions save their charts to PNG files (150 DPI).
# They do NOT call plt.show() to avoid blocking execution.

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss curves side-by-side.

    This chart helps diagnose:
      - Underfitting : Both train and val accuracy are low
      - Overfitting  : Train accuracy is high but val accuracy is much lower
      - Good fit     : Both curves converge close together at a high accuracy

    A green dashed vertical line marks the epoch where the best validation
    accuracy or lowest validation loss was achieved.

    Parameters:
        history: Keras History object returned by model.fit()
                 Contains per-epoch 'accuracy', 'val_accuracy', 'loss', 'val_loss'

    Saves:
        training_history.png — Two side-by-side charts (accuracy + loss)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('TAR UMT Chatbot — Training History', fontsize=16, fontweight='bold', y=1.02)

    epochs = range(1, len(history.history['accuracy']) + 1)

    # --- Accuracy Plot ---
    ax1.plot(epochs, [a * 100 for a in history.history['accuracy']],
             'b-', linewidth=2, label='Train Accuracy', alpha=0.8)
    ax1.plot(epochs, [a * 100 for a in history.history['val_accuracy']],
             'r-', linewidth=2, label='Validation Accuracy', alpha=0.8)
    ax1.fill_between(epochs,
                      [a * 100 for a in history.history['accuracy']],
                      [a * 100 for a in history.history['val_accuracy']],
                      alpha=0.1, color='purple')
    ax1.set_title('Model Accuracy Over Epochs', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Mark best validation accuracy
    best_val_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = max(history.history['val_accuracy']) * 100
    ax1.axvline(x=best_val_epoch, color='green', linestyle='--', alpha=0.5)
    ax1.annotate(f'Best: {best_val_acc:.1f}% (Epoch {best_val_epoch})',
                 xy=(best_val_epoch, best_val_acc),
                 xytext=(best_val_epoch + len(epochs)*0.05, best_val_acc - 8),
                 fontsize=9, color='green', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))

    # --- Loss Plot ---
    ax2.plot(epochs, history.history['loss'],
             'b-', linewidth=2, label='Train Loss', alpha=0.8)
    ax2.plot(epochs, history.history['val_loss'],
             'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    ax2.fill_between(epochs,
                      history.history['loss'],
                      history.history['val_loss'],
                      alpha=0.1, color='purple')
    ax2.set_title('Model Loss Over Epochs', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Mark best validation loss
    best_loss_epoch = np.argmin(history.history['val_loss']) + 1
    best_loss = min(history.history['val_loss'])
    ax2.axvline(x=best_loss_epoch, color='green', linestyle='--', alpha=0.5)
    ax2.annotate(f'Best: {best_loss:.4f} (Epoch {best_loss_epoch})',
                 xy=(best_loss_epoch, best_loss),
                 xytext=(best_loss_epoch + len(epochs)*0.05, best_loss + 0.3),
                 fontsize=9, color='green', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    # plt.show()
    print("  [✓] Saved: training_history.png")


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Generate a heatmap visualization of the confusion matrix.

    A confusion matrix shows how often each actual intent class was predicted as
    each other class. The diagonal represents correct predictions; off-diagonal
    entries represent misclassifications.

    The heatmap uses a blue color scale (darker = higher count), with white grid
    lines separating each cell for readability.

    Parameters:
        y_true       (array-like): True intent class indices (from evaluation)
        y_pred       (array-like): Predicted intent class indices (from model.predict)
        class_names  (list[str]) : Intent tag names for axis labels

    Saves:
        confusion_matrix.png — 22x18 inch heatmap (large to accommodate all intent labels)
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Prediction Count'},
                ax=ax)
    ax.set_title('TAR UMT Chatbot — Confusion Matrix\n(Rows = Actual, Columns = Predicted)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Intent', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Intent', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    # plt.show()
    print("  [✓] Saved: confusion_matrix.png")


def plot_f1_scores(report_dict, class_names):
    """
    Generate a horizontal bar chart showing the F1 score for each individual intent class.

    Bars are color-coded by performance tier:
      - Green  (≥80%) : Good performance — model reliably handles this intent
      - Orange (≥50%) : Moderate performance — may need more training data
      - Red    (<50%) : Poor performance — this intent likely needs attention

    Each bar also displays its Precision and Recall values inline (e.g., 'P:85% R:90%')
    so the reader can identify whether the issue is precision or recall.

    Bars are sorted in ascending F1 order so the lowest-performing intents
    appear at the top for immediate attention.

    Parameters:
        report_dict  (dict)     : Output of sklearn classification_report(output_dict=True)
        class_names  (list[str]): Intent tag names to display on the Y-axis

    Saves:
        f1_scores_chart.png — Horizontal bar chart (height scales with number of classes)
    """
    f1_scores = []
    precision_scores = []
    recall_scores = []
    valid_names = []

    for name in class_names:
        if name in report_dict:
            f1_scores.append(report_dict[name]['f1-score'] * 100)
            precision_scores.append(report_dict[name]['precision'] * 100)
            recall_scores.append(report_dict[name]['recall'] * 100)
            valid_names.append(name)

    # Sort by F1 score
    sorted_indices = np.argsort(f1_scores)
    f1_sorted = [f1_scores[i] for i in sorted_indices]
    prec_sorted = [precision_scores[i] for i in sorted_indices]
    rec_sorted = [recall_scores[i] for i in sorted_indices]
    names_sorted = [valid_names[i] for i in sorted_indices]

    # Color coding: green (>80), orange (50-80), red (<50)
    colors = []
    for f1 in f1_sorted:
        if f1 >= 80:
            colors.append('#2ecc71')   # green
        elif f1 >= 50:
            colors.append('#f39c12')   # orange
        else:
            colors.append('#e74c3c')   # red

    fig, ax = plt.subplots(figsize=(14, max(12, len(valid_names) * 0.35)))

    y_pos = np.arange(len(names_sorted))
    bars = ax.barh(y_pos, f1_sorted, height=0.6, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for i, (bar, f1, prec, rec) in enumerate(zip(bars, f1_sorted, prec_sorted, rec_sorted)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{f1:.0f}%  (P:{prec:.0f}% R:{rec:.0f}%)',
                va='center', fontsize=7, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('TAR UMT Chatbot — Per-Class F1 Score\n(Green ≥80% | Orange ≥50% | Red <50%)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim([0, 115])
    ax.grid(axis='x', alpha=0.3)

    # Add vertical reference lines
    ax.axvline(x=80, color='green', linestyle='--', alpha=0.3, label='80% threshold')
    ax.axvline(x=50, color='orange', linestyle='--', alpha=0.3, label='50% threshold')
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig('f1_scores_chart.png', dpi=150, bbox_inches='tight')
    # plt.show()
    print("  [✓] Saved: f1_scores_chart.png")


def plot_overall_metrics(report_dict):
    """
    Generate a two-panel performance dashboard chart.

    Left Panel (Bar Chart): Weighted average scores for all 6 metrics:
      Accuracy, Precision, Recall, F1, AUC, Specificity
      Each bar is a distinct color. A grey dashed line marks the 80% benchmark.

    Right Panel (Pie Chart): Overall split between correct and incorrect predictions.
      The 'Correct' slice is green; 'Incorrect' is red.
      The correct slice is slightly exploded for visual emphasis.

    This dashboard is intended for high-level summary presentation in the report.

    Parameters:
        report_dict (dict): Extended classification report dict containing:
                            - Standard sklearn keys (accuracy, weighted avg, etc.)
                            - 'auc'         : AUC score (added by evaluate_model)
                            - 'specificity' : Macro specificity (added by evaluate_model)

    Saves:
        overall_metrics.png — 14x6 inch two-panel dashboard chart
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('TAR UMT Chatbot — Overall Performance Dashboard',
                 fontsize=16, fontweight='bold', y=1.02)

    # --- Metric Bar Chart ---
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Spec.']
    values = [
        report_dict.get('accuracy', 0) * 100,
        report_dict.get('weighted avg', {}).get('precision', 0) * 100,
        report_dict.get('weighted avg', {}).get('recall', 0) * 100,
        report_dict.get('weighted avg', {}).get('f1-score', 0) * 100,
        report_dict.get('auc', 0) * 100,
        report_dict.get('specificity', 0) * 100
    ]
    bar_colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#f1c40f', '#1abc9c']

    bars = ax1.bar(metrics, values, color=bar_colors, edgecolor='white', width=0.6)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylim([0, 110])
    ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Weighted Average Metrics', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=80, color='gray', linestyle='--', alpha=0.4)

    # --- Pie Chart: Correct vs Incorrect ---
    correct = report_dict['accuracy'] * 100
    incorrect = 100 - correct
    pie_colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)

    wedges, texts, autotexts = ax2.pie(
        [correct, incorrect], explode=explode,
        labels=['Correct', 'Incorrect'], colors=pie_colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12},
        shadow=True
    )
    autotexts[0].set_fontweight('bold')
    autotexts[0].set_fontsize(14)
    ax2.set_title('Overall Prediction Accuracy', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('overall_metrics.png', dpi=150, bbox_inches='tight')
    # plt.show()
    print("  [✓] Saved: overall_metrics.png")
