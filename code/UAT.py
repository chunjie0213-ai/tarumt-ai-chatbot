"""
UAT.py — User Acceptance Testing (UAT) Results Visualizer
===========================================================
This script generates a bar chart that visualizes the results of the
User Acceptance Testing (UAT) conducted on the TAR UMT AI Chatbot.

UAT was performed by surveying 20 students from TAR UMT. Each student
rated the chatbot across 4 evaluation criteria on a scale of 1 to 5:
  - Usability      : How easy and intuitive the chatbot interface is
  - Response Accuracy : Whether the chatbot gave correct and relevant answers
  - System Speed   : How fast the chatbot responded to user queries
  - Helpfulness    : Whether the chatbot was useful in answering facility info

Output:
  - uat_results.png : Bar chart saved as a PNG image (150 DPI)

Usage:
  python UAT.py
"""

import matplotlib.pyplot as plt  # Import matplotlib for creating charts and graphs

# ============================================================
#  UAT SURVEY DATA
# ============================================================

# The 4 evaluation criteria rated by students during the UAT session
categories = ['Usability', 'Response Accuracy', 'System Speed', 'Helpfulness']

# Average scores collected from 20 student respondents (maximum score = 5)
# Usability     : 4.5 — Students found the chatbot easy to use
# Response Accuracy: 4.2 — Chatbot answers were mostly accurate
# System Speed  : 4.8 — Students rated the chatbot as very fast
# Helpfulness   : 4.0 — Students found it helpful for facility info queries
scores = [4.5, 4.2, 4.8, 4.0]  # Maximum score = 5 marks per category

# ============================================================
#  CHART CONFIGURATION
# ============================================================

# Create a figure with a width of 9 inches and height of 6 inches
plt.figure(figsize=(9, 6))

# Plot a vertical bar chart with:
#   - Green color (#27ae60) to represent positive/satisfactory results
#   - Black edge color for visual clarity between bars
#   - Transparency (alpha=0.8) for a cleaner look
plt.bar(categories, scores, color='#27ae60', edgecolor='black', alpha=0.8)

# Set the Y-axis range from 0 to 5.5 to provide space above bars for score labels
plt.ylim(0, 5.5)

# Label the Y-axis to indicate the scoring range
plt.ylabel('Score (Out of 5)', fontweight='bold')

# Set the chart title with font size and bold style
# The title indicates this is a UAT study with 20 student participants
plt.title(
    'User Acceptance Testing (UAT) Ratings - 20 Students Surveyed',
    fontsize=12, fontweight='bold', pad=20
)

# ============================================================
#  SCORE LABELS ON BARS
# ============================================================

# Loop through each bar and place the score value on top of each bar
# - 'i' is the bar index (x-position), 'v' is the score value
# - The text is placed slightly above the bar (v + 0.1) for readability
# - 'ha=center' centers the label horizontally on the bar
for i, v in enumerate(scores):
    plt.text(i, v + 0.1, str(v), ha='center', fontweight='bold', fontsize=11)

# ============================================================
#  STYLING & EXPORT
# ============================================================

# Add a horizontal dashed grid on the Y-axis for easier score reading
# alpha=0.3 makes the grid lines subtle and non-distracting
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Automatically adjust subplot parameters to prevent label clipping
plt.tight_layout()

# Save the chart as a PNG file at 150 DPI resolution
# This file will be used in the final project report (Figure: UAT Results)
plt.savefig('uat_results.png', dpi=150)