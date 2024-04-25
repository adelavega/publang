import json
import pytest
from pathlib import Path
import os
from publang.utils.split import split_pmc_document

if not os.environ.get("OPENAI_API_KEY", None):
    # This is a test key and should not be used for production
    os.environ["OPENAI_API_KEY"] = "TEST_OPENAI_API_KEY"


@pytest.fixture(scope="session")
def get_data_folder():
    return Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def test_docs(get_data_folder):
    with open(get_data_folder / "test_docs.json") as f:
        data = json.load(f)
    return data


@pytest.fixture(scope="session")
def test_docs_body(test_docs):
    """Returns first Body section for each tes paper"""
    sections = []
    for doc in test_docs:
        for section in split_pmc_document(doc["text"], max_chars=None):
            if section.get("section_0", "") == "Body":
                sections.append(section["content"])

    return sections


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            "authorization",
            "cookie",
            "user-agent",
            "x-stainless-arch",
            "x-stainless-async",
            "x-stainless-lang",
            "x-stainless-os",
            "x-stainless-package-version",
            "x-stainless-runtime",
            "x-stainless-runtime-version",
        ]
    }
