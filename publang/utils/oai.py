""" Methods related to interacting with OpenAI API"""

import openai
import json
from typing import List, Dict
import os
import logging
import untruncate_json

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)


def reexecutor(func, *args, **kwargs):
    return func(*args, **kwargs)


def on_error(retry_state):
    if os.getenv("PL_RAISE_EXCEPTIONS", 'False').lower() in ('true', '1', 't'):
        raise retry_state.retry_object.retry_error_cls(retry_state.outcome)

    # Log exception that was going to be raised
    logging.error(
        f"Max retried attempts reached. Exception: {retry_state.outcome.exception()}"
        )

    return False


retry_attempts = int(os.getenv("PL_RETRY_ATTEMPTS", 10))


def test_error(*args, **kwargs):
    print("trying...")
    raise ValueError("Test error")


if retry_attempts > 1:
    retry_openai = retry(
        retry=retry_if_exception_type(
            (
                openai.APIError,
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.Timeout,
                ValueError,
            )
        ),
        wait=wait_random_exponential(multiplier=1, max=100),
        stop=stop_after_attempt(retry_attempts),
        retry_error_callback=on_error
    )

    reexecutor = retry_openai(reexecutor)


def _format_function(output_schema):
    """Format function for OpenAI function calling from parameters"""
    return [
        {
            "type": "function",
            "function": {
                "name": "extractData",
                "description": "Extract data from scientific text",
                "parameters": output_schema
            }
        }
    ]


def get_openai_chatcompletion(
    messages: List[Dict[str, str]],
    client: openai.OpenAI = None,
    output_schema: Dict[str, object] = None,
    model: str = "gpt-4-0125-preview",
    temperature: float = 0,
    timeout: int = 60,
    response_format: str = None,
    **kwargs
) -> str:
    """Get a chat completion from OpenAI API

    Args:
        messages: A list of dictionaries containing the messages for the LLM.
        client: An OpenAI client object.
        output_schema: A dictionary containing the template for the prompt and the expected keys in the completion.
        model: A string containing the name of the LLM to be used for the extraction.
        temperature: A float containing the temperature for the LLM.
        timeout: An integer containing the timeout for the LLM.
        response_format: A string containing the type of response expected from the LLM (e.g. "json" or "text")
        kwargs: Additional keyword arguments to be passed to the OpenAI API.
    """
    if response_format is not None and response_format.get("type") == "json_object":
        mode = "json"
    elif output_schema is not None:
        mode = "function"
    else:
        mode = "text"

    # If response format is not given, and output schema is given, assume function call
    if mode == "function":
        kwargs["tools"] = _format_function(output_schema)

    if client is None:
        client = openai.OpenAI()

    kwargs.update(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout,
            "response_format": response_format,
        }
    )

    completion = reexecutor(client.chat.completions.create, **kwargs)

    if completion is False:
        return False

    choice = completion.choices[0]

    # If parameters were given, validate & extract json
    response = False
    if mode == "function":
        if choice.message.tool_calls is None:
            logging.error(
                f"No tool calls found. Message: {choice.message.content}"
            )
        else:
            arguments = choice.message.tool_calls[0].function.arguments

            if choice.finish_reason == "length":
                logging.warning("Ran out of tokens. Untruncating...")
                response = json.loads(
                    untruncate_json.complete(arguments)
                )

            else:
                response = json.loads(arguments)

    elif mode == "json":
        # TODO: Improve json validation
        response = json.loads(choice.message.content)
    else:
        response = choice.message.content

    return response


def get_openai_embedding(
    input: str,
    model: str = "text-embedding-ada-002",
    client=None
) -> List[float]:
    """Get the embedding for a given input string"""

    if client is None:
        client = openai.OpenAI()

    resp = reexecutor(client.embeddings.create, input=input, model=model)

    if resp is False:
        return False

    embedding = resp.data[0].embedding

    return embedding
