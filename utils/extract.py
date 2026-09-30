"""
Rule-based field extraction from OCR text.

This step is ordinary pattern matching (regular expressions), NOT AI.
The AI part of LocalLens is the OCR model that reads the text from the image.
"""

import re
from typing import Dict, List

PATTERNS = {
    "Emails": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "Phone numbers": r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,5}\)?[\s-]?)?\d{3,5}[\s-]?\d{4,5}(?!\d)",
    "Dates": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s\d{4}\b",
    "Amounts": r"(?:Rs\.?|INR|USD|EUR|[$€£₹])\s?\d[\d,]*(?:\.\d{1,2})?",
    "Links": r"https?://[^\s]+|www\.[^\s]+",
}


def extract_fields(text: str) -> Dict[str, List[str]]:
    """Return {label: [unique matches in order of appearance]} for labels that matched."""
    found: Dict[str, List[str]] = {}
    for label, pattern in PATTERNS.items():
        matches = []
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            cleaned = match.strip().rstrip(".,;")
            # Phone pattern is loose; require at least 8 digits to count as a phone number.
            if label == "Phone numbers" and sum(c.isdigit() for c in cleaned) < 8:
                continue
            if cleaned and cleaned not in matches:
                matches.append(cleaned)
        if matches:
            found[label] = matches
    # A date like 14/10/2026 must not also be listed as a phone number.
    if "Phone numbers" in found and "Dates" in found:
        found["Phone numbers"] = [p for p in found["Phone numbers"] if p not in found["Dates"]]
        if not found["Phone numbers"]:
            del found["Phone numbers"]
    return found
