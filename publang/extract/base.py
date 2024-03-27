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
        text: str, 
        messages: str,
        output_schema: Dict[str, object],
        model_name: str = "gpt-3.5-turbo") -> Dict[str, str]:
    """Extracts information from a text sample using an OpenAI LLM.

    Args:
        text: A string containing the text sample.
        template: A dictionary containing the template for the prompt and the expected keys in the completion.
        model_name: A string containing the name of the LLM to be used for the extraction.
    """

    # Encode text to ascii
    text = text.encode("ascii", "ignore").decode()

    messages = deepcopy(messages)

    # Format the message with the text
    for message in messages:
        message['content'] = format_string_with_variables(message['content'], text=text)

    data = get_openai_chatcompletion(
        messages,
        output_schema=output_schema,
        model_name=model_name
    )

    return data


def extract_from_multiple(
        texts: List[str], 
        messages: str,
        output_schema: Dict[str, object],
        model_name: str = "gpt-3.5-turbo", 
        num_workers: int = 1) -> Union[List[Dict[str, str]], pd.DataFrame]:
    """Extracts information from multiple text samples using an OpenAI LLM.

    Args:
        texts: A list of strings containing the text samples.
        template: A dictionary containing the template for the prompt and the expected keys in the completion.
        model_name: A string containing the name of the LLM to be used for the extraction.
        return_type: A string specifying the type of the returned object. Can be either "pandas" or "list".
        num_workers: An integer specifying the number of workers to use for parallel processing.
    """

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                extract_from_text, text, messages, output_schema, model_name) 
            for text in texts
            ]

        results = []
        for future in tqdm.tqdm(futures, total=len(texts)):
            results.append(future.result())

    return results


def extract_on_match(
        embeddings_df, annotations_df, messages, output_schema, model_name="gpt-3.5-turbo", 
        num_workers=1):
    """ Extract anntotations on chunk with relevant information (based on annotation meta data) """

    embeddings_df = embeddings_df[embeddings_df.section_0 == 'Body']

    sections = get_relevant_chunks(embeddings_df, annotations_df)

    res = extract_from_multiple(
        sections.content.to_list(), messages, output_schema,
        model_name=model_name, num_workers=num_workers)

    # Combine results into single df and add pmcid
    pred_groups_df = []
    for ix, r in enumerate(res):
        rows = r['groups']
        pmcid = sections.iloc[ix]['pmcid']
        for row in rows:
            row['pmcid'] = pmcid
            pred_groups_df.append(row)
    pred_groups_df = pd.DataFrame(pred_groups_df)

    return sections, pred_groups_df


def _extract_iteratively(
        sub_df, messages, output_schema, model_name="gpt-3.5-turbo"):
    """ Iteratively attempt to extract annotations from chunks in ranks_df until one succeeds. """
    for _, row in sub_df.iterrows():
        res = extract_from_text(row['content'], messages, output_schema, model_name)
        if res['groups']:
            result = [
                {**r, **row[['rank', 'start_char', 'end_char', 'pmcid']].to_dict()} for r in res['groups']
                ]
            return result
        
    return []
    

def search_extract(
        embeddings_df, query, messages, output_schema, model_name="gpt-3.5-turbo", 
        num_workers=1):
    """ Search for query in embeddings_df and extract annotations from nearest chunks,
    using heuristic to narrow down search space if specified.
    """

    # Search for query in chunks
    ranks_df = get_chunk_query_distance(embeddings_df, query)
    ranks_df.sort_values('distance', inplace=True)
    
    # For every document, try to extract annotations by distance until one succeeds
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _extract_iteratively, sub_df, messages, output_schema, model_name) 
            for _, sub_df in ranks_df.groupby('pmcid', sort=False)
            ]

        results = []
        for future in tqdm.tqdm(futures, total=len(ranks_df.pmcid.unique())):
            results.extend(future.result())

    results = pd.DataFrame(results)

    return results