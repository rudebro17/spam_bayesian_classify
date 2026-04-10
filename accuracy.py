from SRC.predict import predict   # predict() – classify a single message


def test_model(model, filepath):
    """
    Evaluate the trained NaiveBayes model against a labelled dataset file.

    Reads each line from filepath (format: label|message), runs predict()
    on the message, compares to the true label, and prints accuracy.

    Bug fix: the original function had an empty loop body so 'correct' and
    'total' were never incremented, causing ZeroDivisionError at the end.
    This version now correctly counts predictions inside the loop.

    Args:
        model    : trained NaiveBayes instance
        filepath : path to the pipe-delimited dataset file
    """
    correct = 0    # running count of correct predictions
    total   = 0    # running count of all evaluated examples

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue                          # skip blank lines

            parts = line.split('|')

            if len(parts) != 2:
                print("Skipping malformed line:", line)
                continue

            # ── BUG FIX: loop body was completely empty in the original ──
            label, text = parts                  # unpack label and message
            pred = predict(model, text)          # classify the message

            total += 1                           # count this example
            if pred == label.strip():
                correct += 1                     # count correct predictions

    # Guard against an empty file to avoid ZeroDivisionError
    if total == 0:
        print("No valid data found in", filepath)
        return

    accuracy = (correct / total) * 100
    print(f"Accuracy: {accuracy:.2f}%  ({correct}/{total} correct)")
# END OF FILE
# test_model()  –  evaluates model accuracy on a file, bug-fixed (loop body was empty)
