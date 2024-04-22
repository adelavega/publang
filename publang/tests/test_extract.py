import pytest
from publang.extract import extract_from_text


TEST_TEMPLATE = {
    "messages": [
        {
            "content": """You will be provided with a text sample/ The sample is delimited with triple backticks. 
            Your task is to identify groups of participants that participated in the study, and underwent MRI. 
            If there is no mention of any participant groups, return a null array.
            For each group identify the number of participants in each grouCall the extractData function to save the output.
            Text sample: ```{text}```""",
            "role": "user"
        }
    ],
    "output_schema": {
        "properties": {
            "groups": {
                "items": {
                    "properties": {
                        "count": {
                            "description": "Number of participants in this group",
                            "type": "integer"
                        }
                    },
                    "required": [
                        "count"
                    ],
                    "type": "object"
                },
                "type": "array"
            }
        },
        "type": "object"
    }
}

# Test extract_from_text function
@pytest.mark.vcr()
def test_extract_from_text(test_docs_body):
    text = test_docs_body[0]
    result = extract_from_text(
        text, model="gpt-4-0125-preview", **TEST_TEMPLATE)

    expected_result = {'groups': [{'count': 28}, {'count': 20}, {'count': 30}]}

    assert isinstance(result, list)
    res = result[0]
    assert 'groups' in res
    assert res == expected_result

@pytest.mark.vcr()
def test_extract_from_multiple(test_docs):
    texts = [doc['text'] for doc in test_docs]
    result = extract_from_text(
        texts, model="gpt-4-0125-preview", **TEST_TEMPLATE)

    assert isinstance(result, list)
    assert len(result) == 5


# # Test search_extract function
# def test_search_extract():
#     embeddings_df = ...
#     query = "search query"
#     messages = "Search for query and extract annotations."
#     output_schema = {"param1": "value1", "param2": "value2"}
#     model_name = "gpt-3.5-turbo"
#     output_path = "output.csv"
#     num_workers = 2

#     result = search_extract(embeddings_df, query, messages, output_schema, model, output_path, num_workers)

#     # Add assertions to validate the result
