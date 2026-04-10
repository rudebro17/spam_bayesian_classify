from collections import defaultdict
from src.preprocess import preprocess

class NaiveBayes:
    def __init__(self):
        self.spam_words = defaultdict(int)
        self.ham_words = defaultdict(int)
        self.spam_count = 0
        self.ham_count = 0
        self.vocab = set()

    def train(self, filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # skip empty lines

                parts = line.split('|')

                if len(parts) != 2:
                    print("Skipping bad line:", line)
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

        self.total_messages = self.spam_count + self.ham_count
        
import random

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split('|')
            if len(parts) != 2:
                continue

            label, text = parts
            data.append((label, text))

    return data

def train_from_list(self, data):
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

    self.total_messages = self.spam_count + self.ham_count