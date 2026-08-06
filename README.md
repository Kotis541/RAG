*This project has been created as part of the 42 curriculum by vokotera.*

## Description
This project implements a Retrieval-Augmented Generation (RAG) system designed to answer questions about a codebase, specifically targeting the vLLM repository. The primary goal is to search the knowledge base for relevant code snippets and documentation, and then generate accurate, source-grounded answers using the Qwen/Qwen3-0.6B language model.

## Instructions

### Installation
```bash
make install

uv run python -m src index --max_chunk_size 2000