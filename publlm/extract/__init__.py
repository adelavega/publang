""" Semantic embedding and search functions. """
from publlm.extract.base import extract_from_text, extract_from_multiple, search_extract, extract_on_match
from publlm.extract import templates

__all__ = [
    'extract_from_text',
    'extract_from_multiple',
    'search_extract',
    'extract_on_match',
    'templates'
]