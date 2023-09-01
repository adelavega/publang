""" Wrappers around OpenAI to make help embedding chunked documents """

import openai
import tqdm
import numpy as np
from publang.search.split import split_pmc_document
from typing import Dict, List, Tuple
import concurrent.futures
from sklearn.metrics.pairwise import euclidean_distances 

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
    wait=wait_random_exponential(multiplier=1, max=60)
)
def openaiembedding_with_backoff(input, model):
    return openai.Embedding.create(
        input=model,
        model=model
    )


def embed_text(text: str, model_name: str = 'text-embedding-ada-002') -> List[float]:
    """ Embed a document using OpenAI's API """
    # Return the embedding
    response = openaiembedding_with_backoff(text, model_name)
    embeddings = response['data'][0]['embedding']

    return embeddings


def embed_pmc_article(
        document: str, 
        model_name: str = 'text-embedding-ada-002', 
        min_tokens: int = 30, 
        max_tokens: int = 4000, 
        **kwargs) -> List[Dict[str, any]]:
    """ Embed a PMC article using OpenAI's API. 
    Split the article into chunks of min_tokens to max_tokens,
    and embed each chunk.
    """

    split_doc = split_pmc_document(document, min_tokens=min_tokens, max_tokens=max_tokens)

    if split_doc:
        # Embed each chunk
        for chunk in split_doc:
            res = embed_text(chunk['content'], model_name)
            chunk['embedding'] = res
            for key, value in kwargs.items():
                chunk[key] = value
        return split_doc
    else:
        return []

def embed_pmc_articles(
        articles: List[Dict],  # List of dicts with keys 'pmcid' and 'text'
        model_name: str = 'text-embedding-ada-002', 
        min_tokens: int = 30, 
        max_tokens: int = 4000,
        num_workers: int = 1
        ) -> List[Dict[str, any]]:
    """ Embed a list of PMC articles using OpenAI's API. 
    Split the article into chunks of min_tokens to max_tokens,
    and embed each chunk.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                embed_pmc_article, art['text'], model_name, min_tokens, max_tokens, 
                pmcid=art['pmcid']) 
            for art in articles
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

def query_embeddings(embeddings: List[List], query: str, compute_ranks=True) -> Tuple[List[float], List[int]]:
    """Query a list of embeddings with a query string. Returns the distances and ranks of the embeddings. """

    embeddings = np.array(embeddings)

    query_embedding = embed_text(query)
    distances = euclidean_distances(embeddings, np.array(query_embedding).reshape(1, -1), squared=True)

    return distances, _rank_numbers(distances)

def get_chunk_query_distance(embeddings_df, query):
    # For every document, get distance and rank between query and embeddings
    distances, ranks = zip(*[
        query_embeddings(sub_df['embedding'].tolist(), query) 
        for pmcid, sub_df in embeddings_df.groupby('pmcid', sort=False)
    ])

    # Combine with meta-data into a df
    ranks_df = embeddings_df[['pmcid', 'content', 'start_char', 'end_char']].copy()
    ranks_df['distance'] = np.concatenate(distances)
    ranks_df['rank'] = np.concatenate(ranks)

    return ranks_df