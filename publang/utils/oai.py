""" Methods related to interacting with OpenAI API"""

import openai
import json
from typing import List, Dict
import os

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)


def reexecutor(func, *args, **kwargs):
    return func(*args, **kwargs)


retry_attempts = int(os.getenv("PUBLANG_RETRY_ATTEMPTS", 50))

if retry_attempts > 1:
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
        stop=stop_after_attempt(retry_attempts),
    )

    reexecutor = retry_openai(reexecutor)


def _format_function(output_schema):
    """Format function for OpenAI function calling from parameters"""
    functions = [{"name": "extractData", "parameters": output_schema}]

    return functions, {"name": "extractData"}


client = openai.OpenAI()


def get_openai_chatcompletion(
    messages: List[Dict[str, str]],
    output_schema: Dict[str, object] = None,
    model_name: str = "gpt-4-0125-preview",
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

    completion = reexecutor(client.chat.completions.create, **kwargs)

    message = completion.choices[0].message

    # If parameters were given, extraction json
    if output_schema is not None:
        response = json.loads(message.function_call.arguments)
    else:
        response = message.content

    return response


def get_openai_embedding(input: str, 
                         model_name: str = "text-embedding-ada-002") -> List[float]:
    resp = reexecutor(client.embeddings.create, input=input, model=model_name)

    embedding = resp.data[0].embedding

    return embedding