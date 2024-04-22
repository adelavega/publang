from typing import List, Dict
import pandas as pd
import os
import json
from publang.search.embed import embed_pmc_articles, get_chunk_query_distance
from publang.extract import extract_from_text
from publang.search import get_relevant_chunks
import tqdm
import concurrent.futures


def _extract_iteratively(
    sub_df, messages, output_schema, model, retry_attempts=2, **kwargs
):
    """Iteratively attempt to extract annotations from chunks in ranks_df.

    Args:
        sub_df (pd.DataFrame): A dataframe containing the document chunks.
        messages (List[Dict[str, str]]): A list of messages to use.
        output_schema (Dict[str, object]): Schema for the output.
        model (str): Name of the OpenAI model to use for the extraction.
        retry_attempts (int): Number of retry attempts to make.
    Returns:
        List[Dict[str, any]]: A list of dictionaries containing values.
    """
    output_keys = output_schema["properties"].keys()
    for _, row in sub_df.iterrows():
        _retries = retry_attempts
        while _retries > 0:
            res = extract_from_text(
                row["content"], messages, output_schema, model, **kwargs
            )
            # Check that main key contains values
            if res and all([res[key] for key in output_keys]):
                return {
                    **res,
                    **row[["rank", "start_char", "end_char", "pmcid"]].to_dict(),
                }
            _retries -= 1
    return []


def search_extract(
    search_query: str,
    messages: List[Dict[str, str]],
    output_schema: Dict[str, object],
    articles: List[Dict],
    extraction_model: str,
    extraction_client=None,
    output_path: str = None,
    embeds_path: str = None,
    embed_model: str = "text-embedding-ada-002",
    embed_client=None,
    min_chars: int = 30,
    max_chars: int = 4000,
    section: str = "Body",
    num_workers: int = 1,
    **kwargs
) -> pd.DataFrame:
    """Extract participant demographics from a list of articles using OpenAI's API.

    Args:
        [Required]
        search_query (str): Query to use for semantic search.
        messages (list): List of messages to use for the extraction.
        output_schema (dict): Schema for the output.
        articles (list): List of articles. Each article is a dictionary with keys 'pmcid' and 'text'.
        extraction_model (str): Name of the chat completion model to use for the extraction.

        [Optional]
        extraction_client: OpenAI client object to use for the extraction.
        output_path (str): Path to JSON prediction. If file exists, the
            extraction will start from previous article in file.
        embeds_path (str): Path to parquet file to save the embeddings to.
            If file exists, the embeddings will be loaded from the file.
        embed_model: Model to use for the embedding.
        embed_client: OpenAI client object to use for the embedding.
        min_chars (int): Minimum chars per chunk.
        max_chars (int): Maximum chars per chunk.
        section (str): Markdown header used to subset articles.
        num_workers (int): Number of workers to use for parallel processing.
        **kwargs: Additional keyword arguments to pass to the OpenAI API.
    Returns:
        pd.DataFrame: Dataframe containing the extracted values.
        pd.DataFrame: Dataframe containing the chunked embeddings.
    """
    embeddings = None
    if embeds_path is not None and os.path.exists(embeds_path):
        # Load embeddings from file, but only for articles in input
        unique_pmcids = set([a["pmcid"] for a in articles])
        embeddings = pd.read_parquet(
            embeds_path, filters=[("pmcid", "in", unique_pmcids)]
        )

    if embeddings is None:
        if articles is None:
            raise ValueError("Either articles or embeddings must be provided.")
        print("Embedding articles...")
        embeddings = embed_pmc_articles(
            articles,
            embed_model,
            min_chars,
            max_chars,
            num_workers=num_workers,
            client=embed_client,
        )
        embeddings = pd.DataFrame(embeddings)

        if embeds_path is not None:
            embeddings.to_parquet(embeds_path, index=False)

    if section is not None:
        embeddings = embeddings[embeddings.section_0 == section]

    # Search for query in chunks
    print("Searching for query in chunks...")
    ranks_df = get_chunk_query_distance(
        embeddings, search_query, client=embed_client, model=embed_model
    )
    ranks_df.sort_values("distance", inplace=True)

    # For every document, extract annotations by distance iteratively
    print("Extracting annotations...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as exc:
        futures = [
            exc.submit(
                _extract_iteratively,
                sub_df,
                messages,
                output_schema,
                extraction_model,
                client=extraction_client,
                **kwargs
            )
            for _, sub_df in ranks_df.groupby("pmcid", sort=False)
        ]

        results = []
        for future in tqdm.tqdm(futures, total=len(futures)):
            results.append(future.result())
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
