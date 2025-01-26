import os
import shutil
import json
import argparse
from datetime import datetime
from multiprocessing import Pool, Lock, cpu_count
import time
from embed import ChunkInput, Chunker, EmbedChromaDB
from langchain.text_splitter import RecursiveCharacterTextSplitter

lock = Lock()


def read_data_file(file_path):
    """Reads a JSON file and returns the data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {e}")


def create_chunk_input(data, chunk_size, chunk_overlap):
    """Creates a ChunkInput object from the provided data."""
    return ChunkInput(
        text=data.get("content", ""),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter=RecursiveCharacterTextSplitter,
        metadata={
            "url": data.get("url", ""),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "source": data.get("source", ""),
            "creator": data.get("creator", ""),
            "success": str(data.get("success", "")),
        },
    )


def handle_processed_file(file_path, move_to=None, delete=False):
    """
    Handles the file after processing: either moves or deletes it, and logs the action.
    """
    if move_to:
        destination = os.path.join(move_to, os.path.basename(file_path))
        os.makedirs(move_to, exist_ok=True)
        shutil.move(file_path, destination)
        print(f"[{datetime.now()}] Moved file to {destination}")
    elif delete:
        os.remove(file_path)
        print(f"[{datetime.now()}] Deleted file {file_path}")


def process_file(args):
    """
    Processes a single file: chunks data, embeds it into ChromaDB, and handles the file after processing.
    This function is suitable for use in multiprocessing.
    """
    file_path, chromadb_path, collection_name, embedding_model_name, chunk_size, chunk_overlap, move_to, delete = args

    try:
        with lock:
            print(f"[{datetime.now()}] Starting processing file: {file_path}")

        # Initialize ChromaDB store (each process creates its own instance)
        chromadb_store = EmbedChromaDB(
            chromadb_path=chromadb_path,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
        )

        # Read and parse data file
        data = read_data_file(file_path)

        # Create ChunkInput and chunk the data
        chunker = Chunker()
        chunk_input = create_chunk_input(data, chunk_size, chunk_overlap)
        chunked_data = chunker.chunk(chunk_input)

        # Embed and store the chunks
        result = chromadb_store.store_embeddings(chunked_data)

        with lock:
            print(f"[{datetime.now()}] Finished processing file: {file_path}")
            print(f"[{datetime.now()}] Processing result: {result['status']}")

        # Handle file after processing
        handle_processed_file(file_path, move_to=move_to, delete=delete)

    except Exception as e:
        with lock:
            print(f"[{datetime.now()}] Error processing file: {file_path} - {e}")


def watch_directory(directory, chromadb_path, collection_name, embedding_model_name, chunk_size, chunk_overlap, max_parallel, move_to=None, delete=False):
    """
    Watches a directory for new files and processes them in parallel.
    """
    print(f"[{datetime.now()}] Daemon mode activated. Watching directory: {directory}")
    processed_files = set()

    while True:
        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith('.json') and f not in processed_files
        ]

        if files:
            args = [
                (file_path, chromadb_path, collection_name, embedding_model_name, chunk_size, chunk_overlap, move_to, delete)
                for file_path in files
            ]

            with Pool(processes=max_parallel) as pool:
                pool.map(process_file, args)

            # Update processed files set
            processed_files.update(os.path.basename(f) for f in files)

        time.sleep(2)  # Polling interval


def validate_file_handling_options(args):
    """Validates that only one file handling option is selected."""
    if args.move_to and args.delete:
        raise ValueError("You cannot use both --move_to and --delete options simultaneously.")


def main():
    # Set up argparse
    parser = argparse.ArgumentParser(
        description="Read JSON files, chunk the data, and embed it into ChromaDB. Supports file, directory, and daemon mode."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a single JSON file, a directory, or the directory to watch in daemon mode.",
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
        help="Name of the OpenAI embedding model (default: text-embedding-3-small).",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Enable daemon mode to watch a directory for new files.",
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=cpu_count(),
        help="Maximum number of parallel tasks (default: number of CPU cores).",
    )
    parser.add_argument(
        "--move_to",
        type=str,
        default=None,
        help="Directory to move successfully processed files.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete successfully processed files.",
    )
    args = parser.parse_args()

    # Validate file handling options
    try:
        validate_file_handling_options(args)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if args.daemon:
        if not os.path.isdir(args.input_path):
            print(f"[{datetime.now()}] Error: Daemon mode requires a directory. '{args.input_path}' is not a valid directory.")
            return
        watch_directory(
            args.input_path,
            args.chromadb_path,
            args.collection_name,
            args.embedding_model_name,
            args.chunk_size,
            args.chunk_overlap,
            args.max_parallel,
            move_to=args.move_to,
            delete=args.delete,
        )
    else:
        if os.path.isfile(args.input_path):
            process_file(
                (
                    args.input_path,
                    args.chromadb_path,
                    args.collection_name,
                    args.embedding_model_name,
                    args.chunk_size,
                    args.chunk_overlap,
                    args.move_to,
                    args.delete,
                )
            )
        elif os.path.isdir(args.input_path):
            process_directory(
                args.input_path,
                args.chromadb_path,
                args.collection_name,
                args.embedding_model_name,
                args.chunk_size,
                args.chunk_overlap,
                args.max_parallel,
                args.move_to,
                args.delete,
            )
        else:
            print(f"[{datetime.now()}] Invalid input path: {args.input_path}. Please provide a valid file or directory.")


if __name__ == "__main__":
    main()
