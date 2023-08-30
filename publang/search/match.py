""" Functions for matching embeddings to annotations, skipping semantic searching and instead finding relevant chunks based on annotation meta data. """

def get_relevant_chunks(embeddings_df, annotations_df):
    """ Get relevant chunks from embeddings_df based on annotations_df"""
    # Find first chunk that contains a true annotation
    sections = []
    # For every section, see if it contains any annotations
    for ix, row in embeddings_df.iterrows():
        annotations  = annotations_df[annotations_df.pmcid == row['pmcid']]
        for ix_a, annot in annotations.iterrows():
            contains = [True for s, e in  zip(annot['start_char'], annot['end_char']) if (row.start_char <= s) & (row.end_char >= e)]
            if any(contains):
                sections.append(ix)
                break
    return embeddings_df.loc[sections]

    