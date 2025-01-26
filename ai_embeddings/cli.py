import argparse
from embed import ChunkInput, Chunker, EmbedChromaDB
from langchain.text_splitter import RecursiveCharacterTextSplitter


def main():
    # Set up argparse
    parser = argparse.ArgumentParser(
        description="Chunk a file, print metadata, and store chunks in ChromaDB."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the file to be chunked.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=400,
        help="Size of each chunk (default: 400).",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=100,
        help="Overlap between chunks (default: 100).",
    )
    parser.add_argument(
        "--chromadb_path",
        type=str,
        default="/home/dsbeav/database/chromadb_store",
        help="Path to ChromaDB storage (default: /home/dsbeav/database/chromadb_store).",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="openai-text-collection-test",
        help="ChromaDB collection name (default: openai-text-collection-test).",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="text-embedding-3-small",
        help="Name of the OpenAI embedding model (default: text-embedding-ada-002).",
    )
    args = parser.parse_args()

    # Create a ChunkInput instance
    chunk_input = ChunkInput(
        path=args.file_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        splitter=RecursiveCharacterTextSplitter,
    )

    try:
        # Process the file with Chunker
        chunker = Chunker()
        chunked_data = chunker.chunk(chunk_input)

        # Output summary of chunks and metadata
        print(f"Total Chunks: {len(chunked_data.docs)}")
        print(f"Source: {chunked_data.source}")
        print(f"Indexed On: {chunked_data.index_on}")
        print(f"Metadata: {chunked_data.metadata}")

        # Store embeddings in ChromaDB
        chromadb_store = EmbedChromaDB(
            chromadb_path=args.chromadb_path,
            collection_name=args.collection_name,
            embedding_model_name=args.embedding_model_name,
        )
        result = chromadb_store.store_embeddings(chunked_data)

        # Print operation summary
        print("\nChromaDB Operation Summary:")
        print(result)

    except Exception as e:
        # Print error message if something goes wrong
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
