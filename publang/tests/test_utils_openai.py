import pytest
import openai
from publang.utils.oai import (
    get_openai_embedding,
    get_openai_chatcompletion
)

@pytest.mark.vcr()
def test_get_openai_chatcompletion():

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Wherea was the world series in 2020?"}
    ]

    response = get_openai_chatcompletion(
        messages=messages,
        model_name="gpt-3.5-turbo",
        temperature=0,
    )

    assert isinstance(response, str)
    assert len(response) > 0
    assert "Globe Life Field" in response


@pytest.mark.vcr()
def test_get_openai_chatcompletion_function_calling():
    messages = [
        {"role": "user", "content": "How many fingers does a human hand have?"}
    ]

    output_schema = {
                'type': 'object',
                'properties': {
                    'fingers': {
                        'type': 'integer',
                        'description': 'Number of fingers in a human hand'
                    }
                }
            }

    response = get_openai_chatcompletion(
        messages=messages,
        output_schema=output_schema,
        model_name="gpt-3.5-turbo",
        temperature=0,
    )

    assert isinstance(response, dict)
    assert 'fingers' in response
    assert response['fingers'] == 5


@pytest.mark.vcr()
def test_get_openai_embedding():
    input_text = "Hello, world!"
    model = 'text-embedding-ada-002'

    embedding = get_openai_embedding(
        input_text, model_name=model)

    assert len(embedding) == 1536
    assert isinstance(embedding[0], float)
    assert embedding[0] != 0.0
