"""
main.py  –  Entry point for the Naive Bayes Spam Classifier.

Flow:
  1. Load the pipe-delimited dataset from data/dataset.txt.
  2. Shuffle randomly so the split is not order-dependent.
  3. Split 80 % training / 20 % test.
  4. Train a NaiveBayes model on the training split.
  5. Evaluate on the test split and print Accuracy + confusion matrix values.
  6. (Optional) Enter an interactive loop so a user can test custom messages.
"""

import os                                    # for building OS-independent file paths
import random                                # for shuffling the dataset

from SRC.train   import NaiveBayes, load_data   # model class + data loader (fixed import casing)
from SRC.predict import predict                  # inference function


# ── 1. Resolve absolute path to the dataset ──────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))   # directory of this script
data_path = os.path.join(BASE_DIR, "data", "dataset.txt")

# ── 2. Load all (label, text) pairs from the dataset ─────────────────────────
data = load_data(data_path)

if len(data) == 0:
    print("ERROR: dataset is empty or not found at", data_path)
    raise SystemExit(1)

# ── 3. Shuffle to prevent order bias ─────────────────────────────────────────
random.shuffle(data)

# ── 4. 80 / 20 train-test split ──────────────────────────────────────────────
split      = int(0.8 * len(data))
train_data = data[:split]
test_data  = data[split:]

# ── 5. Train the model on the training split ──────────────────────────────────
model = NaiveBayes()
model.train_from_list(train_data)   # method is now correctly inside the class (bug fix)

# ── 6. Evaluate on the held-out test split ────────────────────────────────────
correct = 0
tp = fp = tn = fn = 0   # confusion-matrix counters

for label, text in test_data:
    pred = predict(model, text)    # classify the test message

    if pred == label:
        correct += 1              # correctly classified

    # Build confusion matrix entries
    if pred == "spam" and label == "spam":
        tp += 1                   # True Positive
    elif pred == "spam" and label == "ham":
        fp += 1                   # False Positive (spam misclassified)
    elif pred == "ham" and label == "ham":
        tn += 1                   # True Negative
    elif pred == "ham" and label == "spam":
        fn += 1                   # False Negative (missed spam)

# ── 7. Print results ──────────────────────────────────────────────────────────
if len(test_data) == 0:
    print("Not enough data for testing — add more entries to dataset.txt")
else:
    accuracy = (correct / len(test_data)) * 100
    print("=" * 45)
    print(f"  Naive Bayes Spam Classifier – Results")
    print("=" * 45)
    print(f"  Accuracy : {accuracy:.2f}%")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print("=" * 45)

# ── 8. Interactive prediction loop ────────────────────────────────────────────
print("\nEnter a message to classify (Ctrl+C to quit):")
try:
    while True:
        msg = input("  >> ")
        if msg.strip():
            result = predict(model, msg)
            print(f"  Prediction: {result.upper()}\n")
except KeyboardInterrupt:
    print("\nExiting.")
# END OF FILE
# main.py  –  orchestrates: load → shuffle → split → train → evaluate → interactive loop
# import casing fixed: SRC (uppercase) matches the actual folder name on disk