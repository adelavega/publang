""" Provides a high-level API for LLMs for the purpose of infomation retrieval from documents and evaluation of the results."""

import pandas as pd
import tqdm
from copy import deepcopy
from typing import List, Dict, Union
import concurrent.futures

from publang.search.embed import get_chunk_query_distance
from publang.utils.oai import get_openai_chatcompletion
from publang.utils.string import format_string_with_variables
from publang.search import get_relevant_chunks


def extract_from_text(
    texts: Union[str, List[str]],
    messages: str,
    output_schema: Dict[str, object] = None,
    response_format: str = None,
    model: str = "gpt-3.5-turbo",
    num_workers: int = 1,
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
        **kwargs: Additional keyword arguments to be passed to the OpenAI API.
    """

    if isinstance(texts, str):
        texts = [texts]

    def _extract(text, messages, output_schema, model, **kwargs):
        # Encode text to ascii
        text = text.encode("ascii", "ignore").decode()

        messages = deepcopy(messages)

        # Format the message with the text
        for message in messages:
            message["content"] = format_string_with_variables(message["content"], text=text)

        data = get_openai_chatcompletion(
            messages, output_schema=output_schema, model=model,
            response_format=response_format, **kwargs
        )

        return data

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _extract, text, messages, output_schema, model, **kwargs
            )
            for text in texts
        ]

    results = []
    for future in tqdm.tqdm(futures, total=len(texts)):
        results.append(future.result())

    return results


def _extract_iteratively(sub_df, **kwargs):
    """Iteratively attempt to extract annotations from chunks in ranks_df 
    until one succeeds."""
    for _, row in sub_df.iterrows():
        res = extract_from_text(row["content"], **kwargs)
        if res["groups"]:
            result = [
                {**r, **row[["rank", "start_char", "end_char", "pmcid"]].to_dict()}
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
    model: str = "gpt-3.5-turbo",
    num_workers: int = 1,
    **kwargs
):
    """Search for query in embeddings_df and extract annotations from nearest chunks,
    using heuristic to narrow down search space if specified.
    """

    # Search for query in chunks
    ranks_df = get_chunk_query_distance(embeddings_df, query)
    ranks_df.sort_values("distance", inplace=True)

    # For every document, iteratively try to extract annotations by distance
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _extract_iteratively,
                sub_df, messages, output_schema, response_format, model, **kwargs
            )
            for _, sub_df in ranks_df.groupby("pmcid", sort=False)
        ]

        results = []
        for future in tqdm.tqdm(futures, total=len(ranks_df.pmcid.unique())):
            results.extend(future.result())

    results = pd.DataFrame(results)

    return results


def extract_on_match(
    embeddings_df,
    annotations_df,
    messages,
    output_schema,
    model="gpt-3.5-turbo",
    num_workers=1,
):
    """Extract anntotations on chunk with relevant information (based on annotation meta data)"""

    embeddings_df = embeddings_df[embeddings_df.section_0 == "Body"]

    sections = get_relevant_chunks(embeddings_df, annotations_df)

    res = extract_from_text(
        sections.content.to_list(),
        messages,
        output_schema,
        model=model,
        num_workers=num_workers,
    )

    # Combine results into single df and add pmcid
    pred_groups_df = []
    for ix, r in enumerate(res):
        rows = r["groups"]
        pmcid = sections.iloc[ix]["pmcid"]
        for row in rows:
            row["pmcid"] = pmcid
            pred_groups_df.append(row)
    pred_groups_df = pd.DataFrame(pred_groups_df)

    return sections, pred_groups_df
