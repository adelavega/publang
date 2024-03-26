import json
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def get_data_folder():
    return Path(__file__).resolve().parent / 'data'


@pytest.fixture(scope="session")
def test_docs(get_data_folder):
    with open(get_data_folder / 'test_docs.json') as f:
        data = json.load(f)
    return data


@pytest.fixture(scope='module')
def vcr_config():
    return {
        "filter_headers": [('authorization', 'DUMMY')],
    }
