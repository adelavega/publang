""" Wrappers around OpenAI to make help embedding chunked documents """

import tqdm
import numpy as np
from publang.utils.split import split_pmc_document
from typing import Dict, List, Tuple, Union
import concurrent.futures
from sklearn.metrics.pairwise import euclidean_distances
from publang.utils.oai import get_openai_embedding


def embed_pmc_articles(
    articles: Union[str, List[str]],
    model_name: str = "text-embedding-ada-002",
    min_chars: int = 30,
    max_chars: int = 4000,
    num_workers: int = 1,
    **kwargs
) -> List[Dict[str, any]]:
    """Embed a PMC article using OpenAI's API.
    Split the article into chunks of min_chars to max_chars,
    and embed each chunk.
    """

    if isinstance(articles, str):
        articles = [articles]

    def _split_embed(article, model_name, min_chars, max_chars, **kwargs):
        split_doc = split_pmc_document(
            article, min_chars=min_chars, max_chars=max_chars
        )

        if split_doc:
            # Embed each chunk
            for chunk in split_doc:
                res = get_openai_embedding(chunk["content"], model_name)
                chunk["embedding"] = res
                for key, value in kwargs.items():
                    chunk[key] = value
            return split_doc
        else:
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _split_embed,
                article,
                model_name,
                min_chars,
                max_chars,
                **kwargs,
            )
            for article in articles
        ]

        results = []
        for future in tqdm.tqdm(futures, total=len(articles)):
            results += future.result()

    return results


def _rank_numbers(numbers: List[float]) -> List[Tuple[float, int]]:
    """Rank a list of numbers in descending order relative to their original index.

    Args:
        numbers (List[float]): The list of numbers to rank.

    Returns:
        List[Tuple[float, int]]: A list of tuples containing the number and its rank relative to its original index.
    """
    ranked_numbers = sorted([(num, i) for i, num in enumerate(numbers)])
    ranks = [0] * len(numbers)
    for rank, (num, index) in enumerate(ranked_numbers):
        ranks[index] = rank
    return ranks


def query_embeddings(
    embeddings: List[List], query: str, compute_ranks=True
) -> Tuple[List[float], List[int]]:
    """Query a list of embeddings with a query string. Returns the distances and ranks of the embeddings."""

    embeddings = np.array(embeddings)

    query_embedding = get_openai_embedding(query)
    distances = euclidean_distances(
        embeddings, np.array(query_embedding).reshape(1, -1), squared=True
    )

    return distances, _rank_numbers(distances)


def get_chunk_query_distance(embeddings_df, query):
    # For every document, get distance and rank between query and embeddings
    distances, ranks = zip(
        *[
            query_embeddings(sub_df["embedding"].tolist(), query)
            for pmcid, sub_df in embeddings_df.groupby("pmcid", sort=False)
        ]
    )

    # Combine with meta-data into a df
    ranks_df = embeddings_df[["pmcid", "content", "start_char", "end_char"]].copy()
    ranks_df["distance"] = np.concatenate(distances)
    ranks_df["rank"] = np.concatenate(ranks)

    return ranks_df
