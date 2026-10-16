"""Preserve formatting evidence; lexical experts normalize their own text."""
import re
import unicodedata


def preprocess(prompt: str) -> str:
    return prompt.replace('\x00', '').strip()


def words(prompt: str) -> str:
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', preprocess(prompt))).lower()
