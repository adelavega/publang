import json
import pytest
from pathlib import Path
import os

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


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ['authorization', 'cookie', 'user-agent'],
    }
