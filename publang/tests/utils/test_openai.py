import pytest
import openai
from publang.utils import openai_utils


@pytest.mark.vcr()
def test_get_openai_chatcompletion_response():

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]

    response = openai_utils.get_openai_chatcompletion_response(
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

    response = openai_utils.get_openai_chatcompletion_response(
        messages=messages,
        output_schema=output_schema,
        model_name="gpt-3.5-turbo",
        temperature=0,
    )

    assert isinstance(response, dict)
    assert 'fingers' in response
    assert response['fingers'] == 5


@pytest.mark.vcr()
def test_get_openai_embedding_response():
    input_text = "Hello, world!"
    model = 'text-embedding-ada-002'

    response = openai_utils.get_openai_embedding_response(
        input=input_text, model=model)

    assert isinstance(
        response,
        openai.types.create_embedding_response.CreateEmbeddingResponse
        )
    
    assert response.model == "text-embedding-ada-002"
    assert response.object == "list"

    assert len(response.data) > 0
    assert len(response.data[0].embedding) == 1536
    assert isinstance(response.data[0].embedding[0], float)
    assert response.data[0].embedding[0] != 0.0
