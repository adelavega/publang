import pytest
from publang.extract.base import extract_from_text, extract_from_multiple, search_extract, extract_on_match
from publang.extract import templates

def test_extract_from_text():
    text = ""  # fill in test data
    expected = ""  # fill in expected result
    result = extract_from_text(text)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_extract_from_multiple():
    texts = ["", ""]  # fill in test data
    expected = ["", ""]  # fill in expected results
    result = extract_from_multiple(texts)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_search_extract():
    text = ""  # fill in test data
    expected = ""  # fill in expected result
    result = search_extract(text)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_extract_on_match():
    text = ""  # fill in test data
    expected = ""  # fill in expected result
    result = extract_on_match(text)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_templates():
    # fill in test data and expected results
    # depending on what templates is (function, class, etc.)
    pass