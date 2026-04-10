from src.train import NaiveBayes, load_data
from src.predict import predict
import random
import os

# Fix path properly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "dataset.txt")

# Load data
data = load_data(data_path)

# Shuffle
random.shuffle(data)

# Split (80-20)
split = int(0.8 * len(data))
train_data = data[:split]
test_data = data[split:]

# Train model
model = NaiveBayes()
model.train_from_list(train_data)

# Evaluate
correct = 0
tp = fp = tn = fn = 0

for label, text in test_data:
    pred = predict(model, text)

    if pred == label:
        correct += 1

    if pred == "spam" and label == "spam":
        tp += 1
    elif pred == "spam" and label == "ham":
        fp += 1
    elif pred == "ham" and label == "ham":
        tn += 1
    elif pred == "ham" and label == "spam":
        fn += 1

# Avoid division error
if len(test_data) == 0:
    print("Not enough data for testing")
else:
    accuracy = (correct / len(test_data)) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

# Optional: interactive testing
while True:
    msg = input("Enter message: ")
    print("Prediction:", predict(model, msg))