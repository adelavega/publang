""" Provides a high-level API for LLMs for the purpose of infomation retrieval from documents and evaluation of the results."""

import tqdm
from copy import deepcopy
from typing import Dict
import concurrent.futures
from string import Template
from publang.utils.oai import get_openai_chatcompletion


def parallelize_extract(func):
    """Decorator to parallelize the extraction process over texts."""

    def wrapper(text, *args, **kwargs):
        num_workers = kwargs.get("num_workers", 1)

        # If only one text is provided, run the function directly
        if isinstance(text, str):
            return func(text, *args, **kwargs)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as exc:
            futures = [exc.submit(func, t, *args, **kwargs) for t in text]
        
        results = []
        for future in tqdm.tqdm(futures, total=len(text)):
            results.append(future.result())
        return results

    return wrapper


@parallelize_extract
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
