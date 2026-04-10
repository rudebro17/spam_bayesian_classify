import random                          # standard-library import at top (was misplaced mid-file)
from collections import defaultdict    # for word-frequency dictionaries
from SRC.preprocess import preprocess  # local preprocessor to tokenize raw text


class NaiveBayes:
    """
    Naive Bayes classifier that learns from labelled text (spam / ham).
    Uses Laplace (add-1) smoothing so unseen words never produce log(0).
    """

    def __init__(self):
        # word-frequency dictionaries for each class
        self.spam_words = defaultdict(int)
        self.ham_words  = defaultdict(int)

        # document-level counts per class
        self.spam_count = 0
        self.ham_count  = 0

        # full vocabulary seen during training
        self.vocab = set()

        # total documents – initialised to 0 so predict() can guard safely
        self.total_messages = 0

    # ------------------------------------------------------------------
    # train()  –  reads a pipe-delimited file  (label|message per line)
    # ------------------------------------------------------------------
    def train(self, filepath):
        """
        Train directly from a dataset file.
        Each line must be in the format:  label|message text
        e.g.   spam|Win a free iPhone now!
        """
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue                       # skip blank lines

                parts = line.split('|')
                if len(parts) != 2:
                    print("Skipping malformed line:", line)
                    continue

                label, text = parts
                words = preprocess(text)

                if label == 'spam':
                    self.spam_count += 1
                    for word in words:
                        self.spam_words[word] += 1
                        self.vocab.add(word)
                else:
                    self.ham_count += 1
                    for word in words:
                        self.ham_words[word] += 1
                        self.vocab.add(word)

        # recompute total after reading the whole file
        self.total_messages = self.spam_count + self.ham_count

    # ------------------------------------------------------------------
    # train_from_list()  –  train from an in-memory list of (label, text)
    # ------------------------------------------------------------------
    def train_from_list(self, data):
        """
        Train from a Python list of (label, text) tuples.
        Used by main.py after the dataset has been shuffled + split
        so that we evaluate on a held-out test set.
        """
        for label, text in data:
            words = preprocess(text)

            if label == 'spam':
                self.spam_count += 1
                for word in words:
                    self.spam_words[word] += 1
                    self.vocab.add(word)
            else:
                self.ham_count += 1
                for word in words:
                    self.ham_words[word] += 1
                    self.vocab.add(word)

        # update total after processing all training samples
        self.total_messages = self.spam_count + self.ham_count


# ----------------------------------------------------------------------
# load_data()  –  standalone helper to read the dataset into a list
# ----------------------------------------------------------------------
def load_data(filepath):
    """
    Reads the pipe-delimited dataset file and returns a list of
    (label, text) tuples that can be shuffled and split externally.
    Lines that do not match the expected format are silently skipped.
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('|')
            if len(parts) != 2:
                continue                          # skip malformed lines

            label, text = parts
            data.append((label.strip(), text.strip()))

    return data
# END OF FILE
# load_data   – parses dataset.txt into (label, text) tuples
# NaiveBayes  – core classifier: train() for file-based, train_from_list() for in-memory
# train_from_list is a METHOD of NaiveBayes (bug fix: was erroneously a standalone function)