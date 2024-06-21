# publang

[![CI](https://github.com/adelavega/publang/actions/workflows/tests.yml/badge.svg)](https://github.com/adelavega/publang/actions/workflows/tests.yml)

publang is a tool for semantic search and information retrieval using large lange models over biomedical documents.
Currently, we are focused on using OpenAI's API, but we have plans to expand to locally available and domain specific LLMs.

This tool facilitates extracting and evaluating structured annotations over corpora of biomedical articles, such as demographic information, or 

## Installation

    git clone https://github.com/adelavega/publang.git 
    cd publang
    pip install -e .


## Authentication

To use LLM APIs, you'll need to configure your API keys.

PubLang uses OpenAI's client library for inference. Set your API keys using the `OPENAI_API_KEY` environment variable.
Note that many third-party LLM inference APIs support using OpenAI's library, and thus are compatible.
To use other API, simply set the `OPENAI_API_BASE` variable.

For example, to use Fireworks AI:

```
export OPENAI_API_BASE="https://api.fireworks.ai/inference/v1"
export OPENAI_API_KEY="<YOUR_FIREWORKS_API_KEY>"
```

Alternatively, you can pass an initialized `client` object to extraction functions.


## Environment keys

The following envirnonment keys control publang behavior:

`PL_RETRY_ATTEMPTS`: (default: 10) Number of attempts to retry API call before failing
`PL_RAISE_EXCEPTIONS`: (default: false) Whether to raise an exemption to continue to next article.