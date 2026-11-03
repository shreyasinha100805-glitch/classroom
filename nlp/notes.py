"""
Step 6.0: Structured Notes Generation
Combines the extracted information (classification, text, keywords, topics)
and generates organized, structured notes for the user.
"""

from datetime import datetime
import re


def generate_structured_notes(classification, extracted_text, keywords, topics, ocr_items=None):
    """
    Generate organized classroom notes from multi-modal analysis.

    Parameters:
    - classification (dict): {"class": str, "confidence": float}
    - extracted_text (str): Combined text extracted by OCR
    - keywords (list): List of extracted important keywords
    - topics (list): List of detected topic dicts [{"topic": ..., "keywords": ...}]
    - ocr_items (list, optional): Individual detected OCR text snippets with confidence

    Returns:
    - dict: Structured notes containing title, date, classification_info,
            key_points, keywords, topics, full_transcript, and formatted markdown/text.
    """
    date_str = datetime.now().strftime("%B %d, %Y - %I:%M %p")

    # 1. Determine Title & Subject
    primary_topic = "General Academic"
    if topics and len(topics) > 0:
        primary_topic = topics[0].get("topic", "General Academic")
    elif keywords and len(keywords) > 0:
        primary_topic = keywords[0].capitalize()

    title = f"{primary_topic} - Classroom Board Notes"

    # 2. Format Classification Details
    board_type = (classification.get("class") or "Board").capitalize()
    confidence = classification.get("confidence", 0.0)

    # 3. Clean and split extracted text into organized bullet points
    raw_text = extracted_text.strip() if extracted_text else ""
    key_points = []

    if raw_text:
        # Split on common delimiters or sentence boundaries
        sentences = re.split(r"(?<=[.?!;])\s+|\n+", raw_text)
        for sentence in sentences:
            cleaned = sentence.strip()
            if len(cleaned) > 3:
                # Capitalize first letter
                cleaned = cleaned[0].upper() + cleaned[1:]
                key_points.append(cleaned)

    # Fallback to OCR items if sentences were few
    if len(key_points) <= 1 and ocr_items:
        items = [item.get("text", "").strip() for item in ocr_items if len(item.get("text", "").strip()) > 3]
        if items:
            key_points = items

    if not key_points and raw_text:
        key_points = [raw_text]

    # 4. Generate Markdown Formatted Notes
    md_lines = [
        f"# {title}",
        f"**Date:** {date_str}  ",
        f"**Source Classification:** {board_type} (Confidence: {confidence:.1f}%)  ",
        "",
        "## 1. Topic Taxonomy",
    ]

    if topics:
        for t in topics:
            matched_kws = ", ".join(t.get("keywords", []))
            md_lines.append(f"- **{t.get('topic')}** (Matched terms: `{matched_kws}`)")
    else:
        md_lines.append("- *No specific subject taxonomy matched.*")

    md_lines.append("")
    md_lines.append("## 2. Key Vocabulary & Concepts")
    if keywords:
        md_lines.append(", ".join([f"`{k}`" for k in keywords]))
    else:
        md_lines.append("*No prominent keywords identified.*")

    md_lines.append("")
    md_lines.append("## 3. Structured Board Points")
    if key_points:
        for pt in key_points:
            md_lines.append(f"- {pt}")
    else:
        md_lines.append("*No readable board points detected.*")

    md_lines.append("")
    md_lines.append("## 4. Full Extracted Board Text")
    if raw_text:
        md_lines.append(f"> {raw_text}")
    else:
        md_lines.append("> *No text detected on the board.*")

    markdown_notes = "\n".join(md_lines)

    # 5. Generate Clean Plain Text Representation
    txt_lines = [
        f"{title}",
        f"=" * len(title),
        f"Date: {date_str}",
        f"Classification: {board_type} ({confidence:.1f}% confidence)",
        "",
        "TOPICS:",
    ]
    if topics:
        for t in topics:
            txt_lines.append(f"  * {t.get('topic')} ({', '.join(t.get('keywords', []))})")
    else:
        txt_lines.append("  * None detected")

    txt_lines.extend(["", "KEYWORDS:"])
    if keywords:
        txt_lines.append(f"  {', '.join(keywords)}")
    else:
        txt_lines.append("  None")

    txt_lines.extend(["", "KEY POINTS:"])
    if key_points:
        for pt in key_points:
            txt_lines.append(f"  - {pt}")
    else:
        txt_lines.append("  - None")

    txt_lines.extend(["", "FULL EXTRACTED TEXT:"])
    txt_lines.append(raw_text if raw_text else "None")

    plain_text_notes = "\n".join(txt_lines)

    return {
        "title": title,
        "date": date_str,
        "board_type": board_type,
        "confidence": confidence,
        "primary_topic": primary_topic,
        "topics": topics,
        "keywords": keywords,
        "key_points": key_points,
        "full_text": raw_text,
        "markdown": markdown_notes,
        "plain_text": plain_text_notes
    }
