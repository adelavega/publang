import pytest
from publang.extract.extract import (
    extract_from_text, extract_from_multiple,
    extract_on_match, search_extract
)
from publang.extract.templates import ZERO_SHOT_MULTIGROUP


# Test extract_from_text function
def test_extract_from_text(test_docs):
    pmid, text = test_docs[0]
    result = extract_from_text(text, model_name='gpt-3.5-turbo', **ZERO_SHOT_MULTIGROUP)

    assert isinstance(result, dict)


# Test extract_from_multiple function
# def test_extract_from_multiple():
#     texts = ["Text 1", "Text 2", "Text 3"]
#     messages = "Extract information from multiple texts."
#     output_schema = {"param1": "value1", "param2": "value2"}
#     model_name = "gpt-3.5-turbo"
#     num_workers = 2
    
#     result = extract_from_multiple(texts, messages, output_schema, model_name, num_workers)
    
#     # Add assertions to validate the result


# # Test extract_on_match function
# def test_extract_on_match():
#     embeddings_df = ...
#     annotations_df = ...
#     messages = "Extract annotations based on matching criteria."
#     output_schema = {"param1": "value1", "param2": "value2"}
#     model_name = "gpt-3.5-turbo"
#     num_workers = 1
    
#     result = extract_on_match(embeddings_df, annotations_df, messages, output_schema, model_name, num_workers)
    
#     # Add assertions to validate the result


# # Test search_extract function
# def test_search_extract():
#     embeddings_df = ...
#     query = "search query"
#     messages = "Search for query and extract annotations."
#     output_schema = {"param1": "value1", "param2": "value2"}
#     model_name = "gpt-3.5-turbo"
#     output_path = "output.csv"
#     num_workers = 2
    
#     result = search_extract(embeddings_df, query, messages, output_schema, model_name, output_path, num_workers)
    
#     # Add assertions to validate the result
