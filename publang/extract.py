""" Provides a high-level API for LLMs for the purpose of infomation retrieval from documents and evaluation of the results."""

import tqdm
from copy import deepcopy
from typing import List, Dict, Union
import concurrent.futures

from publang.utils.oai import get_openai_chatcompletion
from publang.utils.string import format_string_with_variables


def extract_from_text(
    texts: Union[str, List[str]],
    messages: str,
    output_schema: Dict[str, object],
    model_name: str = "gpt-3.5-turbo",
    num_workers: int = 1,
) -> Dict[str, str]:
    """Extracts information from a text sample using an OpenAI LLM.

    Args:
        texts (Union[str, List[str]]): The text to extract information from.
        messages (str): The message to use for the extraction.
        output_schema (Dict[str, object]): The schema for the output.
        model_name (str, optional): The name of the OpenAI model to use.
        num_workers (int, optional): The number of worker threads to use.

    Returns:
        Dict[str, str]: The extracted information.
    """

    if isinstance(texts, str):
        texts = [texts]

    def _extract(text, messages, output_schema, model_name):
        # Encode text to ascii
        text = text.encode("ascii", "ignore").decode()

        messages = deepcopy(messages)

        # Format the message with the text
        for message in messages:
            message["content"] = format_string_with_variables(message["content"], text=text)

        data = get_openai_chatcompletion(
            messages, output_schema=output_schema, model_name=model_name
        )

        return data

    # If only one text, return single result
    if len(texts) == 1:
        return _extract(texts[0], messages, output_schema, model_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _extract, text, messages, output_schema, model_name
            )
            for text in texts
        ]

    results = []
    for future in tqdm.tqdm(futures, total=len(texts)):
        results.append(future.result())

    return results
