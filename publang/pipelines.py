from typing import List, Dict
import pandas as pd
import os
import json
from publang.search.embed import embed_pmc_articles, get_chunk_query_distance
from publang.extract import extract_from_text
from publang.search import get_relevant_chunks
import tqdm
import concurrent.futures


def _extract_iteratively(sub_df, messages, output_schema, model_name):
    """Iteratively attempt to extract annotations from chunks in ranks_df.

    Args:
        sub_df (pd.DataFrame): A dataframe containing the document chunks.
        messages (List[Dict[str, str]]): A list of messages to use.
        output_schema (Dict[str, object]): Schema for the output.
        model_name (str): Name of the OpenAI model to use for the extraction.
    Returns:
        List[Dict[str, any]]: A list of dictionaries containing values.
    """
    for _, row in sub_df.iterrows():
        res = extract_from_text(
            row["content"], messages, output_schema, model_name)
        if res:
            result = {**res, **row[["rank", "start_char", "end_char", "pmcid"]].to_dict()}
            return result

    return []


def search_extract(
    search_query: str,
    messages: List[Dict[str, str]],
    output_schema: Dict[str, object],
    articles: List[Dict],
    output_path: str = None,
    embeddings: pd.DataFrame = None,
    embedding_model_name: str = "text-embedding-ada-002",
    embeddings_path: str = None,
    min_chars: int = 30,
    max_chars: int = 4000,
    extraction_model_name: str = "gpt-3.5-turbo",
    subset_section: str = "Body",
    num_workers: int = 1,
) -> pd.DataFrame:
    """Extract participant demographics from a list of articles using OpenAI's API.

    Args:
        search_query (str): Query to use for semantic search.
        messages (list): List of messages to use for the extraction.
        output_schema (dict): Schema for the output.
        articles (list): List of articles. Each article is a dictionary with keys 'pmcid' and 'text'.
        output_path (str): Path to JSON file to save the predictions to. If file exists, the predictions will
            be loaded from the file, and extraction will start from the last article in the file.
        embedding_model_name (str): Name of the OpenAI embedding model to use.
        embeddings_path (str): Path to parquet file to save the embeddings to.
        min_chars (int): Minimum number of chars per chunk.
        max_chars (int): Maximum number of chars per chunk.
        extraction_model_name (str): Name of the OpenAI model.
        subset_section (str): Optional header to use for subset extraction.
        num_workers (int): Number of workers to use for parallel processing.
    Returns:
        pd.DataFrame: Dataframe containing the extracted values.
        pd.DataFrame: Dataframe containing the chunked embeddings.
    """
    embeddings = None
    if embeddings_path is not None and os.path.exists(embeddings_path):
        # Load embeddings from file, but only for articles in input
        unique_pmcids = set([a["pmcid"] for a in articles])
        embeddings = pd.read_parquet(
            embeddings_path, filters=[("pmcid", "in", unique_pmcids)]
        )

    if embeddings is None:
        if articles is None:
            raise ValueError("Either articles or embeddings must be provided.")
        print("Embedding articles...")
        embeddings = embed_pmc_articles(
            articles,
            embedding_model_name,
            min_chars,
            max_chars,
            num_workers=num_workers,
        )
        embeddings = pd.DataFrame(embeddings)

        if embeddings_path is not None:
            embeddings.to_parquet(embeddings_path, index=False)

    if subset_section is not None:
        embeddings = embeddings[embeddings.section_0 == subset_section]

    # Search for query in chunks
    ranks_df = get_chunk_query_distance(embeddings, search_query)
    ranks_df.sort_values("distance", inplace=True)

    # For every document, extract annotations by distance iteratively
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _extract_iteratively, sub_df, messages, output_schema,
                extraction_model_name
            )
            for _, sub_df in ranks_df.groupby("pmcid", sort=False)
        ]

        results = []
        for future in tqdm.tqdm(futures, total=len(futures)):
            results += future.result()
            # Save every 10 results
            if output_path is not None and len(results) % 10 == 0:
                json.dump(results, open(output_path, "w"))

    if output_path is not None:
        json.dump(results, open(output_path, "w"))

    return results


def extract_on_match(
    embeddings_df,
    annotations_df,
    messages,
    output_schema,
    model_name="gpt-3.5-turbo",
    num_workers=1,
):
    """Extract anntotations on chunk with relevant information
    (based on annotation meta data)
    """

    embeddings_df = embeddings_df[embeddings_df.section_0 == "Body"]

    sections = get_relevant_chunks(embeddings_df, annotations_df)

    res = extract_from_text(
        sections.content.to_list(),
        messages,
        output_schema,
        model_name=model_name,
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
