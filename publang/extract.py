""" Provides a high-level API for LLMs for the purpose of infomation retrieval from documents and evaluation of the results."""

from copy import deepcopy
from typing import Dict
from string import Template
from publang.utils.oai import get_openai_chatcompletion
from publang.utils.parallelize import parallelize_inputs


@parallelize_inputs
def extract_from_text(
    text: str,
    messages: str,
    model: str,
    output_schema: Dict[str, object],
    response_format: str = None,
    client=None,
    **kwargs
) -> Dict[str, str]:
    """Extracts information from a text sample using an OpenAI LLM.

    Args:
        text: A string containing the text sample.
        messages: A list of dictionaries containing the messages for the LLM.
        output_schema: A dictionary containing the template for the prompt and the expected keys in the completion.
        response_type: A string containing the type of response expected from the LLM (e.g. "json" or "text")
        model: A string containing the name of the LLM to be used for the extraction.
        num_workers: An integer containing the number of workers to be used for the extraction.
        client: An OpenAI client object.
        **kwargs: Additional keyword arguments to be passed to the OpenAI API.
    """
    # Encode text to ascii
    text = text.encode("ascii", "ignore").decode()

    messages = deepcopy(messages)
    # Format the message with the text
    for message in messages:
        message["content"] = Template(message["content"]).substitute(text=text)

    return get_openai_chatcompletion(
        messages,
        output_schema=output_schema,
        model=model,
        response_format=response_format,
        client=client,
        **kwargs
    )
