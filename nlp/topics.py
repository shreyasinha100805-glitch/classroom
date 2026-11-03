# ==========================================
# CLASSROOM BOARD TOPIC DETECTION
# ==========================================

TOPIC_CATEGORIES = {

    "Artificial Intelligence": [
        "artificial",
        "intelligence",
        "ai",
        "machine",
        "learning",
        "neural",
        "network",
        "networks",
        "deep",
        "model",
        "models",
        "training",
        "classification",
        "prediction"
    ],

    "Computer Science": [
        "algorithm",
        "algorithms",
        "program",
        "programming",
        "computer",
        "software",
        "data",
        "system",
        "systems",
        "database",
        "code",
        "coding",
        "application"
    ],

    "Mathematics": [
        "equation",
        "equations",
        "formula",
        "formulas",
        "matrix",
        "matrices",
        "vector",
        "vectors",
        "function",
        "functions",
        "derivative",
        "derivatives",
        "integral",
        "integrals",
        "number",
        "numbers",
        "calculation"
    ],

    "Physics": [
        "force",
        "forces",
        "energy",
        "velocity",
        "acceleration",
        "motion",
        "mass",
        "electric",
        "electricity",
        "magnetic",
        "magnetism",
        "wave",
        "waves",
        "frequency",
        "pressure"
    ],

    "Electronics": [
        "circuit",
        "circuits",
        "voltage",
        "current",
        "resistor",
        "resistors",
        "capacitor",
        "capacitors",
        "transistor",
        "transistors",
        "diode",
        "diodes",
        "signal",
        "signals",
        "frequency"
    ],

    "Computer Networks": [
        "network",
        "networks",
        "internet",
        "protocol",
        "protocols",
        "router",
        "switch",
        "server",
        "client",
        "tcp",
        "ip",
        "http",
        "ethernet"
    ],

    "Database": [
        "database",
        "databases",
        "sql",
        "table",
        "tables",
        "query",
        "queries",
        "postgresql",
        "mysql",
        "record",
        "records"
    ]
}


# ==========================================
# TOPIC DETECTION
# ==========================================

def detect_topics(text):
    """
    Detect classroom topics from OCR text.
    """

    if not text:
        return []

    text_lower = text.lower()

    detected_topics = []

    for topic, keywords in TOPIC_CATEGORIES.items():

        matches = []

        for keyword in keywords:

            if keyword in text_lower:

                matches.append(
                    keyword
                )

        if matches:

            detected_topics.append({
                "topic": topic,
                "matches": len(matches),
                "keywords": matches
            })

    # Highest number of matching keywords first
    detected_topics.sort(
        key=lambda item: item["matches"],
        reverse=True
    )

    return detected_topics


# ==========================================
# SIMPLE TOPIC NAME FUNCTION
# ==========================================

def get_topic_names(text):

    topics = detect_topics(
        text
    )

    return [
        item["topic"]
        for item in topics
    ]


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    sample_text = """
    Neural networks and deep learning
    are used to train machine learning models.
    The model learns patterns from data.
    """

    print()
    print("==============================================")
    print("       TOPIC DETECTION TEST")
    print("==============================================")

    topics = detect_topics(
        sample_text
    )

    if not topics:

        print()
        print("No topics detected.")

    else:

        print()

        for item in topics:

            print(
                "Topic:",
                item["topic"]
            )

            print(
                "Matches:",
                item["matches"]
            )

            print(
                "Keywords:",
                ", ".join(
                    item["keywords"]
                )
            )

            print()

    print("==============================================")