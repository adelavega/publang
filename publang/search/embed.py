""" Wrappers around OpenAI to make help embedding chunked documents """

import openai
import tqdm
from publang.search.split import split_pmc_document
from typing import Dict, List

def embed_text(text: str, model_name: str = 'text-embedding-ada-002') -> List[float]:
    """ Embed a document using OpenAI's API """
    # Return the embedding
    response = openai.Embedding.create(
        input=text,
        model=model_name
    )
    embeddings = response['data'][0]['embedding']

    return embeddings

def embed_pmc_articles(
        articles: List[Dict], # List of dicts with keys 'pmcid' and 'text'
        model_name: str = 'text-embedding-ada-002', 
        min_tokens: int = 30, 
        max_tokens: int = 4000) -> List[Dict[str, any]]:
    """ Embed a list of PMC articles using OpenAI's API. 
    Split the article into chunks of min_tokens to max_tokens,
    and embed each chunk.
    """

    results = []
    for art in tqdm.tqdm(articles):
        split_doc = split_pmc_document(art['text'], min_tokens=min_tokens, max_tokens=max_tokens)
        if not split_doc:
            continue
        
        for chunk in split_doc:
            res = embed_text(chunk['content'], model_name)
            chunk['embedding'] = res
            chunk['pmcid'] = art['pmcid']
            results.append(chunk)

    return results