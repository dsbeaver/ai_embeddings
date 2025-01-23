# AI Embeddings

AI Embeddings is a Python package designed for generating embeddings for text documents using AI models and storing them efficiently. The package provides the ability to split large text files or documents into smaller chunks, generate embeddings for each chunk using OpenAI models, and store the results in ChromaDB for easy retrieval and analysis.

## Key Features

- **Text Chunking:** Splits large text files or documents into manageable chunks.
- **Embedding Generation:** Uses OpenAI models to generate embeddings for each chunk of text.
- **Efficient Storage:** Stores the generated embeddings in a ChromaDB database for easy retrieval and analysis.
- **Metadata Management:** Allows for the inclusion of additional metadata with each chunk of text, which is also stored in ChromaDB.
- **File Type Support:** Supports a variety of file types, including PDF, DOCX, and plain text files.
- **Command-Line Interface:** Provides a CLI for easy usage and integration into larger workflows.

## Installation

Ensure you have Python version 3.8 or later installed. 

To install the package, you can build it from source using Poetry. First, clone the repository:

```
git clone https://github.com/yourusername/ai_embeddings.git
cd ai_embeddings
```

Then, build and install the package:

```
python -m build
pip install dist/ai_embeddings-0.1.2-py3-none-any.whl
```

## Configuration

There is no specific environment variable configuration required for this package.

## Usage

The package can be used both as a library and as a CLI tool. Here is an example of how to use the CLI:

```
ai_embeddings /path/to/file --chunk_size 400 --chunk_overlap 100 --chromadb_path /path/to/chromadb --collection_name test-collection --embedding_model_name text-embedding-3-small
```

This will chunk the file at the specified path into chunks of size 400 with an overlap of 100, generate embeddings for each chunk using the `text-embedding-3-small` model, and store the results in the ChromaDB database at the specified path under the collection `test-collection`.

## Directory Structure

```
ai_embeddings/
    pyproject.toml
    .gitignore
    ai_embeddings/
        __init__.py
        embed.py
        cli.py
```

## Contributing

Contributions are welcomed! Please fork the repository and create a pull request with your changes, or create an issue if you find any bugs or have suggestions for improvements.

## License

This project is licensed under the terms specified in the `LICENSE` file of the repository.

## Contact

For any inquiries, please reach out to Dwight Beaver at dsbeav@gmail.com.  This readme was generated using ChatGPT