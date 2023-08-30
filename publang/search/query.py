import numpy as np
from typing import List, Tuple
from publang.embed.base import embed_text
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

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
        embeddings: List[List], 
        query: str,
        distance_metric: str = 'euclidean'
        ) -> Tuple[List[float], List[int]]:
    """Query a list of embeddings with a query string. Returns the distances and ranks of the embeddings. """

    embeddings = np.array(embeddings)

    query_embedding = embed_text(query)

    if distance_metric == 'euclidean':
        distances = euclidean_distances(embeddings, np.array(query_embedding).reshape(1, -1), squared=True)
    elif distance_metric == 'cosine':
        distances = cosine_distances(embeddings, np.array(query_embedding).reshape(1, -1))

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