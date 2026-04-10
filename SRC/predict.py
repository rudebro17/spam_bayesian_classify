import math
from src.preprocess import preprocess

def predict(model, text):
    words = preprocess(text)

    p_spam = math.log(model.spam_count / model.total_messages)
    p_ham = math.log(model.ham_count / model.total_messages)

    for word in words:
        spam_freq = model.spam_words[word] + 1
        ham_freq = model.ham_words[word] + 1

        p_spam += math.log(
            spam_freq / (sum(model.spam_words.values()) + len(model.vocab))
        )
        p_ham += math.log(
            ham_freq / (sum(model.ham_words.values()) + len(model.vocab))
        )

    return "spam" if p_spam > p_ham else "ham"