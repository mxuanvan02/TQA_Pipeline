from __future__ import annotations

import re
from typing import Any


DOMAIN_MAP: dict[str, str] = {
    "4": "History of Law",
    "10": "Legislative Drafting",
    "11": "Legal Theory",
    "12": "Legal Theory",
    "13": "Civil Law I",
    "14": "Civil Law II",
    "15": "Commercial Law I",
    "16": "Commercial Law II",
    "17": "Constitutional Law VN",
    "18": "Constitutional Law VN",
    "19": "Comparative Constitutional",
    "20": "Comparative Law",
    "21": "Administrative Law",
    "22": "Admin. Procedure",
    "23": "Environmental Law",
    "24": "Civil Procedure",
    "25": "Family Law",
    "26": "Criminal Law (General)",
    "27": "Criminal Law (Specific)",
    "28": "Criminal Procedure",
    "29": "Tax / Budget Law",
    "30": "Banking Law",
    "31": "International Law",
    "32": "International Law",
    "33": "Private International Law",
    "34": "Intl. Commercial Law",
    "36": "Intl. Economic Law",
    "37": "Land Law",
    "38": "Labor Law",
    "39": "IP Law",
    "40": "Forensic Psychology",
    "41": "Competition Law",
    "42": "Civil Registration",
    "43": "Legal Practice",
    "46": "Criminology",
    "47": "Criminal Qualification",
    "48": "Social Security Law",
    "49": "Notarization Law",
    "50": "Inheritance Law",
    "51": "Real Estate Law",
    "52": "Land Dispute Law",
    "53": "Contract Skills",
}

_PLACEHOLDER_DOMAIN_TAGS = {"", "civil_law", "unknown", "Unknown"}


def infer_domain_tag_from_identifier(identifier: str) -> str:
    match = re.match(r"^(\d+)", str(identifier).strip())
    if not match:
        return "unknown"
    return DOMAIN_MAP.get(match.group(1), "Other")


def infer_domain_tag_from_record(record: dict[str, Any]) -> str:
    domain_tag = str(record.get("domain_tag", "")).strip()
    if domain_tag not in _PLACEHOLDER_DOMAIN_TAGS:
        return domain_tag

    for key in ("doc_id", "chunk_id", "qa_id"):
        inferred = infer_domain_tag_from_identifier(str(record.get(key, "")))
        if inferred != "unknown":
            return inferred

    return "unknown"
