"""Conservative selection of OCR evidence for NLP and generated notes.

Raw OCR strings stay in ``ocr_items``.  This module selects only exact, known
academic/common terms, clear identifiers, numbers, and symbols for analysis; it
does not autocorrect or complete uncertain text.
"""

import re

from nlp.topics import TOPIC_CATEGORIES


MIN_REGION_CONFIDENCE = 0.55
MIN_NUMBER_CONFIDENCE = 0.75
MIN_ACRONYM_CONFIDENCE = 0.75

COMMON_TERMS = {
    "a", "about", "age", "all", "and", "are", "as", "at", "be", "build",
    "by", "can", "class", "column", "columns", "create", "data", "date",
    "department", "dept", "display", "do", "dob", "for", "from", "github",
    "have", "id", "in", "is", "it", "key", "name", "no", "of", "on", "or",
    "page", "pages", "primary", "record", "records", "roll", "row", "rows",
    "select", "student", "students", "table", "tables", "that", "the", "to",
    "total", "unique", "use", "using", "value", "values", "where", "with",
    "year",
}

ACADEMIC_TERMS = {
    "ai", "algorithm", "algorithms", "artificial", "attribute", "attributes",
    "average", "database", "databases", "deep", "derivative", "equation",
    "entity", "foreign", "function", "functions", "http", "int", "integer",
    "internet", "ip", "key", "machine", "max", "maximum", "min", "minimum",
    "model", "models", "motion", "mysql", "network", "networks", "neural",
    "normalization", "null", "postgresql", "prediction", "protocol", "query",
    "relation", "relational", "schema", "server", "software", "sql", "system",
    "systems", "tcp", "training", "tuple", "varchar", "vector", "voltage",
    "wave", "energy", "force", "mass", "velocity", "circuit", "current",
    "voltage", "formula", "matrix", "number", "numbers", "calculation",
    "biology", "chemistry", "computer", "mathematics", "physics", "science",
}
ACADEMIC_TERMS.update(
    term.casefold()
    for topic_terms in TOPIC_CATEGORIES.values()
    for term in topic_terms
)

KNOWN_TERMS = COMMON_TERMS | ACADEMIC_TERMS
MEANINGFUL_TERMS = ACADEMIC_TERMS | {
    "age", "attribute", "column", "database", "date", "department", "dept",
    "dob", "foreign", "id", "key", "name", "primary", "record", "relational",
    "roll", "row", "schema", "student", "table", "total", "tuple", "value",
}
TOKEN_RE = re.compile(r"->|→|↔|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[<>=!+*/^%-]+")
OPERATOR_RE = re.compile(r"^(?:<=|>=|!=|==|=|<|>|\+|-|\*|/|\^|%|->|→|↔)$")


def _recognized_tokens(text, confidence):
    tokens = TOKEN_RE.findall(text)
    accepted = []
    for token in tokens:
        normalized = token.lower()
        if normalized in KNOWN_TERMS:
            accepted.append(token)
        elif token.isdigit() or (token.replace(".", "", 1).replace(",", "", 1).isdigit()):
            if confidence >= MIN_NUMBER_CONFIDENCE:
                accepted.append(token)
        elif token.isupper() and 2 <= len(token) <= 6 and confidence >= MIN_ACRONYM_CONFIDENCE:
            accepted.append(token)
        elif OPERATOR_RE.fullmatch(token) and confidence >= MIN_NUMBER_CONFIDENCE:
            accepted.append(token)
    return accepted, tokens


def assess_ocr_quality(ocr_items):
    """Return safe analysis evidence plus a reasoned reliability assessment."""
    valid_items = [
        item for item in ocr_items
        if isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and item.get("text", "").strip()
        and isinstance(item.get("confidence"), (int, float))
    ]
    mean_confidence = (
        sum(float(item["confidence"]) for item in valid_items) / len(valid_items)
        if valid_items else 0.0
    )
    accepted_lines = []
    accepted_tokens = []
    meaningful_terms = set()
    accepted_regions = 0
    analysis_items = []

    for item in valid_items:
        text = item["text"].strip()
        confidence = float(item["confidence"])
        if confidence < MIN_REGION_CONFIDENCE:
            continue
        tokens, all_tokens = _recognized_tokens(text, confidence)
        if not tokens:
            continue
        accepted_tokens.extend(tokens)
        meaningful_terms.update(
            token.lower() for token in tokens if token.lower() in MEANINGFUL_TERMS
        )
        # Retain a complete line only if every content token is supported.
        # For mixed lines, pass only exact supported tokens to NLP; the complete
        # recognizer text remains available in the raw OCR panel/API field.
        if len(tokens) == len(all_tokens):
            accepted_lines.append(text)
        else:
            accepted_lines.append(" ".join(tokens))
        accepted_regions += 1
        analysis_item = {
            "text": accepted_lines[-1],
            "confidence": round(confidence, 4),
        }
        if isinstance(item.get("bbox"), (list, tuple)):
            analysis_item["bbox"] = item["bbox"]
        analysis_items.append(analysis_item)

    analysis_text = "\n".join(accepted_lines)
    quality_reliable = (
        mean_confidence >= 0.72
        and accepted_regions >= 2
        and len(accepted_tokens) >= 5
        and len(meaningful_terms) >= 3
        and len(analysis_text) >= 24
    )
    warning = None if quality_reliable else (
        "OCR confidence is low or too little text could be verified. "
        "Please retake the image closer to the board, in focus, with all writing visible. "
        "No study notes were generated from uncertain text."
    )
    return {
        "analysis_text": analysis_text,
        "analysis_items": analysis_items,
        "reliable": quality_reliable,
        "warning": warning,
        "average_confidence": round(mean_confidence, 4),
        "valid_region_count": len(valid_items),
        "analysis_region_count": accepted_regions,
        "supported_token_count": len(accepted_tokens),
        "meaningful_term_count": len(meaningful_terms),
        "meaningful_terms": sorted(meaningful_terms),
    }


def low_confidence_notes(warning):
    """A notes-shaped warning response with no unsupported classroom content."""
    title = "Text Capture Needs Review"
    message = warning or "The OCR output could not be verified. Please retake the image."
    markdown = f"# {title}\n\n> {message}\n"
    plain = f"{title}\n{'=' * len(title)}\n\n{message}\n"
    return {
        "title": title,
        "date": "",
        "board_type": "Unverified",
        "confidence": 0,
        "primary_topic": "Unverified",
        "topics": [],
        "keywords": [],
        "key_points": [],
        "full_text": "",
        "warning": message,
        "markdown": markdown,
        "plain_text": plain,
    }
