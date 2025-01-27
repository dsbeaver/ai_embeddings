# AI Embeddings

AI Embeddings is a Python package designed to provide embedding tools and AI-related functionalities. The package is capable of processing both JSON and plain text files, splitting the content into chunks, and storing the generated embeddings in a [ChromaDB](https://github.com/openai/chromadb) collection.

## Key Features

- Chunking: AI Embeddings splits text into chunks using a specified splitter and returns a `ChunkedData` object containing the chunks and associated metadata.
- Embedding: The package generates embeddings using OpenAI's model for a list of texts.
- Storage: The generated embeddings are stored in a ChromaDB collection.
- MIME-type Validation: The input file's MIME type is validated before processing.
- Multiprocessing: Thanks to multiprocessing, the package can process files concurrently, providing a significant performance boost when dealing with large datasets.
- Daemon Mode: In this mode, the package watches a directory for new files and processes them as they appear.

## Installation

You can install AI Embeddings by cloning the repository and building it using `setuptools` and `wheel`.

```bash
git clone https://github.com/username/ai_embeddings.git
cd ai_embeddings
pip install setuptools wheel
python setup.py bdist_wheel
pip install dist/*.whl
```

## Configuration

Configuration is done through environment variables. Below are the available options:

- `EMBED_FILES_CHUNK_SIZE`: Size of each chunk (default: 400).
- `EMBED_FILES_CHUNK_OVERLAP`: Overlap between chunks (default: 100).
- `EMBED_FILES_CHROMADB_PATH`: Path to ChromaDB storage (default: '/path/to/chromadb').
- `EMBED_FILES_COLLECTION_NAME`: ChromaDB collection name (default: 'test_collection').
- `EMBED_FILES_EMBEDDING_MODEL_NAME`: Embedding model name (default: 'text-embedding-ada-002').
- `EMBED_FILES_WATCH_DIR`: Directory to watch in daemon mode.
- `EMBED_FILES_PROCESSED_DIR`: Directory to move successfully processed files.
- `EMBED_FILES_FAILED_DIR`: Directory to move failed files.
- `EMBED_FILES_CONCURRENCY`: Number of concurrent workers in daemon mode (default: 5).

## Usage Examples

- File Processing:

```bash
python ai_embeddings/cli.py input.txt
```

- Run in Daemon Mode:

```bash
export EMBED_FILES_WATCH_DIR=/path/to/watch
python ai_embeddings/cli.py --daemon
```

## Directory Structure

This project has the following structure:

```
ai_embeddings/
    pyproject.toml
    .gitignore
    env_template
    LICENSE
    ai_embeddings/
        __init__.py
        embed.py
        cli.py
```

- `embed.py`: Contains the core functionality for chunking files, generating embeddings, and storing them in ChromaDB.
- `cli.py`: The command-line interface for the package.

## License

This project is licensed under the terms of the LICENSE file included in this repository.

> This README.md was generated using ChatGPT.
