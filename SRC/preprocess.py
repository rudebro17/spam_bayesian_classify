import re   # regular-expression library for text cleaning


def preprocess(text):
    """
    Convert raw message text into a clean list of lowercase words.

    Pipeline:
      1. Lowercase the entire string so 'FREE' and 'free' are the same token.
      2. Strip all characters that are not lowercase letters or whitespace
         (removes punctuation, digits, special symbols).
      3. Split on whitespace to produce the final token list.

    Returns:
        list[str]  –  e.g.  "Win Money NOW!!!" → ['win', 'money', 'now']
    """

    # Step 1 – lowercase
    text = text.lower()

    # Step 2 – remove everything except letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 3 – tokenise on whitespace
    words = text.split()

    return words
# END OF FILE
# preprocess()  –  lowercase → strip non-alpha → split into word tokens