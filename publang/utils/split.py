import re
from typing import List, Optional
import warnings


def split_lines(text: str, max_chars: int = 100) -> List[str]:
    """Join strings to form largest possible strings that are less than max_chars
    """

    strings = text.splitlines()
    if text[-1] == "\n":
        strings[-1] = strings[-1] + "\n"

    chunks = []
    current_chunk = ""
    for ix, string in enumerate(strings):
        if ix != 0:
            string = "\n" + string
        if len(current_chunk) + len(string) + 1 <= max_chars:
            current_chunk += string
        else:
            if current_chunk != "":
                chunks.append(current_chunk)
            current_chunk = string
    chunks.append(current_chunk)

    if strings[-1] == "":
        chunks[-1] = chunks[-1] + "\n"

    return chunks


def split_markdown(
    text: str,
    delimiters: List[str],
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> List[str]:
    """Split markdown text into chunks based on delimiters.

    Args:
        text (str): Markdown text to split.
        delimiters (list): List of delimiters to split on.
        top_level (bool): Whether or not the current text is top level.
        max_chars (int): Maximum number of tokens per chunk.

    Returns:
        list: List of chunks.
    """

    if not delimiters:
        # Join lines to form largest possible strings that are less than max_chars
        return [(None, c) for c in split_lines(text, max_chars=max_chars)]

    delim_match = f"\n{delimiters[0]}"

    # Split on first delimiter
    candidate_chunks = re.split(delim_match, text)

    # If there is only one chunk, split on next delimiter
    if len(candidate_chunks) == 1:
        chunks = split_markdown(text, delimiters[1:], min_chars, max_chars)

    # Iterate over chunks
    chunks = []
    prev_chunk = None
    section_name = None
    for ix, chunk in enumerate(candidate_chunks):
        if chunk:
            if not ix == 0:
                section_name, _ = chunk.split("\n", maxsplit=1)
                chunk = delim_match + chunk
            if prev_chunk:
                chunk = prev_chunk + chunk
                prev_chunk = None
            if min_chars and len(chunk) < min_chars:
                prev_chunk = chunk
                continue
            if max_chars and len(chunk) > max_chars:
                chunks.append(
                    (
                        section_name,
                        split_markdown(chunk, delimiters[1:], min_chars, max_chars),
                    )
                )
            else:
                chunks.append((section_name, chunk))

    return chunks


def _flatten_sections(sections, section_depth=0, **kwargs):
    """Flatten list of tuples into list of dictionaries with keys corresponding to section headers."""
    flattened = []
    for section_name, content in sections:
        section_key = f"section_{section_depth}"
        if isinstance(content, tuple):
            content = [content]
        if isinstance(content, list):
            if section_name is not None:
                kwargs[section_key] = section_name
            flattened.extend(
                _flatten_sections(content, section_depth=section_depth + 1, **kwargs)
            )
        else:
            if section_name is not None:
                kwargs[section_key] = section_name
            flattened.append({"content": content, **kwargs})
    return flattened


def split_pmc_document(
    text: str,
    delimiters: List[str] = ["# ", "## ", "### "],
    min_chars: int = 20,
    max_chars: int = None,
) -> List[str]:
    """Split PMC document text into chunks based on delimiters, and split by top level sections.

    Args:
        text (str): Markdown text to split.
        delimiters (list): List of delimiters to split on.
        min_chars (int): Minimum number of tokens per chunk (for headers)
        max_chars (int): Maximum number of tokens per chunk.

    Returns:
        list: List of chunks.
    """

    # If failed to split, markdown is not formatted properly
    # Skip for now
    if len(re.split(f"\n# ", text)) == 1:
        warnings.warn("Skipping document, not in markdown")
        return

    _outputs = split_markdown(text, delimiters, min_chars, max_chars)
    _outputs = _flatten_sections(_outputs)

    # Add start_chars and end_chars
    for ix, chunk in enumerate(_outputs):
        if ix == 0:
            chunk["start_char"] = 0
        else:
            chunk["start_char"] = _outputs[ix - 1]["end_char"]
        chunk["end_char"] = chunk["start_char"] + len(chunk["content"])

    return _outputs
