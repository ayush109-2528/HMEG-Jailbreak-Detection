"""Offline metadata candidates; extracted PDF contents are never instructions."""
import re
import unicodedata


def normalize_title(value):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", value).casefold())


def metadata_candidates(pages):
    candidates = []
    for page, text in enumerate(pages, 1):
        head = text[:6000]
        abstract = bool(re.search(r"\babstract\b|A B S T R A C T", head, re.I))
        if page == 1 or abstract:
            candidates.append({"page": page, "opening_lines": head.splitlines()[:12],
                "doi_candidates": sorted(set(re.findall(r"10\.\d{4,9}/[^\s<>]+", head))),
                "arxiv_candidates": sorted(set(re.findall(r"arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", head, re.I))),
                "abstract_heading_found": abstract, "requires_human_review": True})
    return candidates
