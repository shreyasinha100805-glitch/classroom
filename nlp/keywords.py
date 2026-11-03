import re
from collections import Counter

# ==========================================
# OCR TEXT CLEANING
# ==========================================

def clean_text(text):
    """
    Clean noisy OCR text.
    """

    if not text:
        return ""

    # Remove unusual symbols
    text = re.sub(
        r"[^a-zA-Z0-9\s.,:;()\-+/]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================
# STOP WORDS
# ==========================================

STOP_WORDS = {
    "the",
    "is",
    "are",
    "was",
    "were",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "a",
    "an",
    "as",
    "by",
    "with",
    "from",
    "this",
    "that",
    "these",
    "those",
    "it",
    "be",
    "at",
    "we",
    "you",
    "he",
    "she",
    "they",
    "i",
    "this",
    "using",
    "used"
}


# ==========================================
# KEYWORD EXTRACTION
# ==========================================

def extract_keywords(
    text,
    max_keywords=10
):
    """
    Extract frequently occurring useful words.
    """

    text = clean_text(text)

    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        text.lower()
    )

    useful_words = []

    for word in words:

        if word in STOP_WORDS:
            continue

        useful_words.append(
            word
        )

    frequency = Counter(
        useful_words
    )

    keywords = [
        word
        for word, count
        in frequency.most_common(
            max_keywords
        )
    ]

    return keywords


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    sample_text = """
    Neural networks learn patterns from data.
    Deep learning uses multiple layers.
    Machine learning models learn from data.
    """

    print()
    print("==============================================")
    print("       KEYWORD EXTRACTION TEST")
    print("==============================================")

    cleaned = clean_text(
        sample_text
    )

    print()
    print("Cleaned text:")
    print(cleaned)

    keywords = extract_keywords(
        sample_text
    )

    print()
    print("Keywords:")

    for keyword in keywords:
        print("-", keyword)

    print()
    print("==============================================")