""" Provides a high-level API for LLMs for the purpose of infomation retrieval from documents and evaluation of the results."""

import pandas as pd
import tqdm
from copy import deepcopy
from typing import List, Dict, Union
import concurrent.futures

from publang.search.embed import get_chunk_query_distance
from publang.utils.oai import get_openai_chatcompletion
from publang.utils.string import format_string_with_variables


def parallelize_extract(func):
    """Decorator to parallelize the extraction process over texts."""
    def wrapper(texts, *args, **kwargs):
        num_workers = kwargs.get("num_workers", 1)
        if isinstance(texts, str):
            texts = [texts]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    func, text, *args, **kwargs
                )
                for text in texts
            ]
        results = []
        for future in tqdm.tqdm(futures, total=len(texts)):
            results.append(future.result())
        return results
    return wrapper


@parallelize_extract
def extract_from_text(
    text: str,
    messages: str,
    output_schema: Dict[str, object] = None,
    response_format: str = None,
    model: str = "gpt-3.5-turbo",
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
        message["content"] = format_string_with_variables(
            message["content"], text=text)

    return get_openai_chatcompletion(
        messages, output_schema=output_schema, model=model,
        response_format=response_format, client=client, **kwargs
    )


def _extract_iteratively(sub_df, **kwargs):
    """Iteratively attempt to extract annotations from chunks in ranks_df 
    until one succeeds."""
    for _, row in sub_df.iterrows():
        res = extract_from_text(row["content"], **kwargs)
        if res["groups"]:
            result = [
                {**r,
                 **row[["rank", "start_char", "end_char", "pmcid"]].to_dict()}
                for r in res["groups"]
            ]
            return result

    return []


def search_extract(
    embeddings_df: pd.DataFrame,
    query: str,
    messages: List[Dict[str, str]],
    output_schema: Dict[str, object] = None,
    response_format: str = None,
    chat_model: str = "gpt-3.5-turbo",
    chat_client=None,
    embed_client=None,
    embed_model=None,
    num_workers: int = 1,
    **kwargs
):
    """Search for query in embeddings_df and extract annotations from nearest
    chunks using heuristic to narrow down search space if specified.

    Args:
        embeddings_df: A DataFrame containing the embeddings of the document chunks..
        query: A string containing the query to find relevant chunks.
        messages: A list of dictionaries containing the messages for the LLM.
        output_schema: A dictionary containing the template for the prompt and the expected keys in the completion.
        response_format: A string containing the type of response expected from the LLM (e.g. "json" or "text")
        model: A string containing the name of the LLM to be used for the extraction.
        num_workers: An integer containing the number of workers to be used for the extraction.
        completion_client: An OpenAI client object for the completion API.
        embed_client: An OpenAI client object for the embedding API.
        **kwargs: Additional keyword arguments to be passed to the OpenAI API.
    """

    # Search for query in chunks
    ranks_df = get_chunk_query_distance(
        embeddings_df, query, client=embed_client, model=embed_model)

    # For every document, iteratively try to extract annotations by distance
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as exc:
        futures = [
            exc.submit(
                _extract_iteratively,
                sub_df, messages, output_schema, response_format,
                chat_model, client=chat_client, **kwargs
            )
            for _, sub_df in ranks_df.groupby("pmcid", sort=False)
        ]

        results = []
        for future in tqdm.tqdm(futures, total=len(ranks_df.pmcid.unique())):
            results.extend(future.result())

    results = pd.DataFrame(results)

    return results
