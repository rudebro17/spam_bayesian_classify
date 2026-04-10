import math                            # for logarithm calculations (log-probability)
from SRC.preprocess import preprocess  # tokenizer – converts raw text to word list


def predict(model, text):
    """
    Classify a single message as 'spam' or 'ham' using the trained
    Naive Bayes model.

    Steps:
      1. Tokenise the input text using preprocess().
      2. Start log-probabilities from class priors log P(spam) and log P(ham).
      3. For each word apply Laplace-smoothed likelihoods.
      4. Return the class with the higher log-probability.

    Guard: if the model has never been trained (total_messages == 0) we
    return 'unknown' instead of crashing with a ZeroDivisionError.
    """

    # --- Safety guard: model must have at least one training example ---
    if model.total_messages == 0:
        return "unknown"                          # bug fix: prevents ZeroDivisionError

    words = preprocess(text)

    # --- Log-prior probabilities ---
    p_spam = math.log(model.spam_count / model.total_messages)
    p_ham  = math.log(model.ham_count  / model.total_messages)

    # --- Cache denominators outside the loop (performance fix) ---
    # Recomputing sum() inside a loop over every word is O(V) per word –
    # caching reduces it to O(1) per word.
    spam_total = sum(model.spam_words.values()) + len(model.vocab)
    ham_total  = sum(model.ham_words.values())  + len(model.vocab)

    # --- Log-likelihood for each word (Laplace smoothing: +1) ---
    for word in words:
        spam_freq = model.spam_words.get(word, 0) + 1   # add-1 smoothing
        ham_freq  = model.ham_words.get(word,  0) + 1   # add-1 smoothing

        p_spam += math.log(spam_freq / spam_total)
        p_ham  += math.log(ham_freq  / ham_total)

    # --- Return class with higher log-probability ---
    return "spam" if p_spam > p_ham else "ham"
# END OF FILE
# predict()  –  performs log-probability Naive Bayes classification
#   fixes: ZeroDivisionError guard added, denominator cached outside the word loop