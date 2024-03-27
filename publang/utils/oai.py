""" Methods related to interacting with OpenAI API"""

import openai
import json
from typing import List, Dict

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

client = openai.OpenAI()

retry_openai = retry(
    retry=retry_if_exception_type(
        (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.Timeout,
        )
    ),
    wait=wait_random_exponential(multiplier=1, max=100),
    stop=stop_after_attempt(50),
)


@retry_openai
def _chat_completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def _format_function(output_schema):
    """Format function for OpenAI function calling from parameters"""
    functions = [{"name": "extractData", "parameters": output_schema}]

    return functions, {"name": "extractData"}


def get_openai_chatcompletion(
    messages: List[Dict[str, str]],
    output_schema: Dict[str, object] = None,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0,
    timeout: int = 30,
) -> str:

    kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
    }
    if output_schema is not None:
        functions, function_call = _format_function(output_schema)
        kwargs["functions"] = functions
        kwargs["function_call"] = function_call

    completion = _chat_completion_with_backoff(**kwargs)

    message = completion.choices[0].message

    # If parameters were given, extraction json
    if output_schema is not None:
        response = json.loads(message.function_call.arguments)
    else:
        response = message.content

    return response


@retry_openai
def get_openai_embedding(
    input: str, model_name: str = "text-embedding-ada-002"
) -> List[float]:
    resp = client.embeddings.create(input=input, model=model_name)

    embedding = resp.data[0].embedding

    return embedding
