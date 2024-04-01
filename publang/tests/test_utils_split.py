from publang.utils.split import split_lines, split_markdown, _flatten_sections, split_pmc_document


def test_split_lines():
    text = "This is a test.\nThis is only a test."
    result = split_lines(text, max_tokens=20)
    assert result == ["This is a test.", "\nThis is only a test."]


def test_split_markdown():
    text = "# Header\nThis is a test.\n## Subheader\nThis is only a test."
    delimiters = ["# ", "## "]
    result = split_markdown(text, delimiters, min_tokens=20, max_tokens=50)
    assert result == [(None, [(None, '# Header\nThis is a test.'), ('Subheader', '\n## Subheader\nThis is only a test.')])]


def test__flatten_sections():
    sections = [('Header', '# Header\nThis is a test.'), ('Subheader', '## Subheader\nThis is only a test.')]
    result = _flatten_sections(sections)
    assert result == [{'content': '# Header\nThis is a test.', 'section_0': 'Header'}, {'content': '## Subheader\nThis is only a test.', 'section_0': 'Subheader'}]


def test_split_pmc_document(test_docs):
    text = test_docs[0]['text']
    delimiters = ["# ", "## ", "###"]
    result = split_pmc_document(text, delimiters, min_tokens=20, max_tokens=50)
    assert len(result) == 180

    # First section is author names
    assert result[0]['start_char'] == 0
    assert result[0]['end_char'] == 225

    # Body begins a few sections later
    assert result[10]['section_0'] == 'Body'
    assert result[10]['section_1'] == 'Introduction '
    assert result[10]['start_char'] == 2296
    assert result[10]['end_char'] == 3392
    assert len(result[10]['content']) == 1096
