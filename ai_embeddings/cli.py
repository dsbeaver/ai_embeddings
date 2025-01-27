import os
import json
import argparse
import shutil
from datetime import datetime
from ai_embeddings.embed import ChunkInput, Chunker, EmbedChromaDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from multiprocessing import Pool, Lock
import time
import logging
import openai

# Don't want to see each Post to api.openai
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING) 

# Ignore chromadb for testing 
chroma_logger = logging.getLogger("chromadb")
chroma_logger.setLevel(logging.ERROR) 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)

lock = Lock()

def process_file_wrapper(args):
    """Wrapper to process files in a multiprocessing pool."""
    file_path, chunk_size, chunk_overlap, chromadb_path, collection_name, embedding_model_name, processed_dir, failed_dir, delete = args

    try:
        # Reinitialize FileProcessor in the worker process
        chunker = Chunker()
        chromadb_store = EmbedChromaDB(
            chromadb_path=chromadb_path,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
        )
        file_processor = FileProcessor(chunker, chromadb_store, chunk_size, chunk_overlap, processed_dir, failed_dir, delete)
        file_processor.process_file(file_path)
    except Exception as e:
        with lock:
            logging.error(f"Failed to process file {file_path}. Error: {e}")


class FileProcessor:
    """Lightweight, serializable version of FileProcessor for worker processes."""
    def __init__(self, chunker, chromadb_store, chunk_size, chunk_overlap, processed_dir, failed_dir, delete):
        self.chunker = chunker
        self.chromadb_store = chromadb_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_dir = processed_dir
        self.failed_dir = failed_dir
        self.delete = delete

    @staticmethod
    def is_valid_json_schema(data):
        """Validate if the JSON data matches the required schema."""
        required_keys = {"url", "timestamp", "source", "creator", "content", "success"}
        return all(key in data for key in required_keys)

    def process_json(self, file_path):
        """Process a JSON file and embed its content."""
        try:
            logging.info(f"Processing JSON file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not self.is_valid_json_schema(data):
                raise ValueError("JSON schema validation failed. Processing as plain text.")

            chunk_input = ChunkInput(
                text=data["content"],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                splitter=RecursiveCharacterTextSplitter,
                metadata={
                    "url": data["url"],
                    "timestamp": data["timestamp"],
                    "source": data["source"],
                    "creator": data["creator"],
                    "success": str(data["success"]),
                },
            )
            self.store_embeddings(chunk_input, file_path)

        except Exception as e:
            self.handle_failed_file(file_path, error=e)

    def process_text_file(self, file_path):
        """Process a plain text file and embed its content."""
        try:
            logging.info(f"Processing text file: {file_path}")
            chunk_input = ChunkInput(
                path=file_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                splitter=RecursiveCharacterTextSplitter,
            )
            self.store_embeddings(chunk_input, file_path)

        except Exception as e:
            self.handle_failed_file(file_path, error=e)

    def store_embeddings(self, chunk_input, file_path):
        """Process the input chunks and store them into ChromaDB."""
        chunked_data = self.chunker.chunk(chunk_input)
        result = self.chromadb_store.store_embeddings(chunked_data)

        logging.info(f"Successfully processed and stored embeddings for file: {file_path}")
        logging.info(f"Embedding result: {result}")

        if self.processed_dir:
            os.makedirs(self.processed_dir, exist_ok=True)
            dest_path = os.path.join(self.processed_dir, os.path.basename(file_path))
            logging.info(f"Moving file to processed directory: {dest_path}")
            shutil.move(file_path, dest_path)
        elif self.delete:
            logging.info(f"Deleting file: {file_path}")
            os.remove(file_path)

    def handle_failed_file(self, file_path, error=None):
        """Handle failed files by moving them to the failed directory."""
        logging.error(f"Failed to process file: {file_path}. Error: {error}")
        if self.failed_dir:
            os.makedirs(self.failed_dir, exist_ok=True)
            dest_path = os.path.join(self.failed_dir, os.path.basename(file_path))
            logging.info(f"Moving file to failed directory: {dest_path}")
            shutil.move(file_path, dest_path)

    def process_file(self, file_path):
        """Determine file type and process accordingly."""
        logging.info(f"Starting processing for file: {file_path}")
        if file_path.endswith(".json"):
            self.process_json(file_path)
        else:
            self.process_text_file(file_path)


def run_daemon(args):
    """Run the script in daemon mode to watch a directory for new files."""
    logging.info(f"Processing existing files in directory: {args.watch_dir}")
    files = [os.path.join(args.watch_dir, f) for f in os.listdir(args.watch_dir) if os.path.isfile(os.path.join(args.watch_dir, f))]

    # Serialize necessary arguments for workers
    pool_args = [
        (
            file,
            args.chunk_size,
            args.chunk_overlap,
            args.chromadb_path,
            args.collection_name,
            args.embedding_model_name,
            args.processed_dir,
            args.failed_dir,
            args.delete,
        )
        for file in files
    ]

    # Process existing files
    with Pool(args.concurrency) as pool:
        pool.map(process_file_wrapper, pool_args)

    # Set up the observer to watch for new files
    logging.info(f"Watching directory: {args.watch_dir} for new files...")
    with Pool(args.concurrency) as pool:
        event_handler = DirectoryWatcher(args, pool)  # Pass args instead of precomputed pool_args
        observer = Observer()
        observer.schedule(event_handler, args.watch_dir, recursive=False)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()


class DirectoryWatcher(FileSystemEventHandler):
    """Lightweight event handler for multiprocessing pool."""
    def __init__(self, args, pool):
        self.args = args  # Store args to reuse when a new file is detected
        self.pool = pool

    def on_created(self, event):
        if not event.is_directory:
            logging.info(f"New file detected: {event.src_path}")
            pool_args = (
                event.src_path,
                self.args.chunk_size,
                self.args.chunk_overlap,
                self.args.chromadb_path,
                self.args.collection_name,
                self.args.embedding_model_name,
                self.args.processed_dir,
                self.args.failed_dir,
                self.args.delete,
            )
            self.pool.apply_async(process_file_wrapper, args=(pool_args,))


def main():
    parser = argparse.ArgumentParser(description="File processing tool with JSON and text embedding support.")
    parser.add_argument(
        "input_path", 
        type=str, 
        nargs="?",  # Make input_path optional
        help="Path to a file or directory for processing (not required in daemon mode)."
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=int(os.getenv("EMBED_FILES_CHUNK_SIZE", 400)), 
        help="Size of each chunk (default: value from EMBED_FILES_CHUNK_SIZE or 400)."
    )
    parser.add_argument(
        "--chunk_overlap", 
        type=int, 
        default=int(os.getenv("EMBED_FILES_CHUNK_OVERLAP", 100)), 
        help="Overlap between chunks (default: value from EMBED_FILES_CHUNK_OVERLAP or 100)."
    )
    parser.add_argument(
        "--chromadb_path", 
        type=str, 
        default=os.getenv("EMBED_FILES_CHROMADB_PATH", "/path/to/chromadb"), 
        help="Path to ChromaDB storage (default: value from EMBED_FILES_CHROMADB_PATH or '/path/to/chromadb')."
    )
    parser.add_argument(
        "--collection_name", 
        type=str, 
        default=os.getenv("EMBED_FILES_COLLECTION_NAME", "test_collection"), 
        help="ChromaDB collection name (default: value from EMBED_FILES_COLLECTION_NAME or 'test_collection')."
    )
    parser.add_argument(
        "--embedding_model_name", 
        type=str, 
        default=os.getenv("EMBED_FILES_EMBEDDING_MODEL_NAME", "text-embedding-ada-002"), 
        help="Embedding model name (default: value from EMBED_FILES_EMBEDDING_MODEL_NAME or 'text-embedding-ada-002')."
    )
    parser.add_argument(
        "--watch_dir", 
        type=str, 
        default=os.getenv("EMBED_FILES_WATCH_DIR"), 
        help="Directory to watch in daemon mode (default: value from EMBED_FILES_WATCH_DIR)."
    )
    parser.add_argument(
        "--processed_dir", 
        type=str, 
        default=os.getenv("EMBED_FILES_PROCESSED_DIR"), 
        help="Directory to move successfully processed files (default: value from EMBED_FILES_PROCESSED_DIR)."
    )
    parser.add_argument(
        "--failed_dir", 
        type=str, 
        default=os.getenv("EMBED_FILES_FAILED_DIR"), 
        help="Directory to move failed files (default: value from EMBED_FILES_FAILED_DIR)."
    )
    parser.add_argument("--delete", action="store_true", help="Delete successfully processed files.")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode to watch a directory.")
    parser.add_argument(
        "--concurrency", 
        type=int, 
        default=int(os.getenv("EMBED_FILES_CONCURRENCY", 5)), 
        help="Number of concurrent workers in daemon mode (default: value from EMBED_FILES_CONCURRENCY or 5)."
    )
    args = parser.parse_args()

    # Validation: Check for conflicting or missing arguments
    if args.daemon:
        if not args.watch_dir:
            print("Error: --watch_dir must be specified in daemon mode.")
            return
        run_daemon(args)
    elif args.input_path:
        chromadb_store = EmbedChromaDB(
            chromadb_path=args.chromadb_path,
            collection_name=args.collection_name,
            embedding_model_name=args.embedding_model_name,
        )
        chunker = Chunker()
        file_processor = FileProcessor(
            chunker=chunker,
            chromadb_store=chromadb_store,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            processed_dir=args.processed_dir,
            failed_dir=args.failed_dir,
            delete=args.delete,
        )
        if os.path.isfile(args.input_path):
            file_processor.process_file(args.input_path)
        elif os.path.isdir(args.input_path):
            for file_name in os.listdir(args.input_path):
                file_path = os.path.join(args.input_path, file_name)
                file_processor.process_file(file_path)
        else:
            print(f"Invalid input path: {args.input_path}. Please provide a valid file or directory.")
    else:
        print("Error: Please provide an input path for file/directory processing or enable daemon mode.")


if __name__ == "__main__":
    main()
