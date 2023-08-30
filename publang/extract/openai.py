""" Methods related to interacting with OpenAI Chat Completion API"""

import openai
import json
import re 
from typing import List, Dict

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)


@retry(
    retry=retry_if_exception_type((
        openai.error.APIError, 
        openai.error.APIConnectionError, 
        openai.error.RateLimitError, 
        openai.error.ServiceUnavailableError, 
        openai.error.Timeout)), 
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def format_function(parameters):
    """ Format function for OpenAI function calling from parameters"""
    functions = [
        {
            'name': 'extractData',
            'parameters': parameters
        }

    ]

    function_call = {"name": "extractData"}

    return functions, function_call

def get_openai_json_response(
        messages: List[Dict[str, str]],
        parameters: Dict[str, object],
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0, 
        request_timeout: int = 30) -> str:
    # Check that model_name is a valid model name before using it in the openai.ChatCompletion.create() call.
    valid_models = ["gpt-3.5-turbo", "gpt-4"]
    if model_name not in valid_models:
        raise ValueError(f"{model_name} is not a valid OpenAI model name.")
    
    functions, function_call = format_function(parameters)

    completion = chat_completion_with_backoff(
        model=model_name,
        messages=messages,
        functions=functions,
        function_call=function_call,
        temperature=temperature,
        request_timeout=request_timeout
    )
    
    c = completion['choices'][0]['message']
    res = c['function_call']['arguments']
    data = json.loads(res)

    return data


def format_string_with_variables(string: str, **kwargs: str) -> str:
    # Find all possible variables in the string
    possible_variables = set(re.findall(r"{(\w+)}", string))

    # Find all provided variables in the kwargs dictionary
    provided_variables = set(kwargs.keys())

    # Check that all provided variables are in the possible variables
    if not provided_variables.issubset(possible_variables):
        raise ValueError(f"Provided variables {provided_variables} are not in the possible variables {possible_variables}.")

    # Format the string with the provided variables
    return string.format(**kwargs)