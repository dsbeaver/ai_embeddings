import os
import json
import argparse
import shutil
import time
import logging
from datetime import datetime
from multiprocessing import Pool, Queue, Manager
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ai_embeddings.embed import ChunkInput, Chunker, EmbedChromaDB
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    handlers=[logging.StreamHandler()],
)

class FileProcessor:
    """Processes files to generate text chunks and embeddings but does not store them."""

    def __init__(self, chunker, chromadb_store, chunk_size, chunk_overlap):
        self.chunker = chunker
        self.chromadb_store = chromadb_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def is_valid_json_schema(data):
        """Validate if the JSON data matches the required schema."""
        required_keys = {"url", "timestamp", "source", "creator", "content", "success"}
        return all(key in data for key in required_keys)

    def process_json(self, file_path):
        """Process a JSON file and return embeddings."""
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

            chunked_data = self.chunker.chunk(chunk_input)
            embeddings = self.chromadb_store.generate_embeddings(chunked_data.docs)

            return {
                "file_path": file_path,
                "texts": chunked_data.docs,
                "embeddings": embeddings["embeddings"],
                "metadata": chunked_data.metadata,
                "source": chunked_data.source,
            }

        except Exception as e:
            logging.error(f"Failed to process JSON file {file_path}: {e}")
            return None

    def process_text_file(self, file_path):
        """Process a plain text file and return embeddings."""
        try:
            logging.info(f"Processing text file: {file_path}")
            chunk_input = ChunkInput(
                path=file_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                splitter=RecursiveCharacterTextSplitter,
            )
            chunked_data = self.chunker.chunk(chunk_input)
            embeddings = self.chromadb_store.generate_embeddings(chunked_data.docs)

            return {
                "file_path": file_path,
                "texts": chunked_data.docs,
                "embeddings": embeddings["embeddings"],
                "metadata": chunked_data.metadata,
                "source": chunked_data.source,
            }

        except Exception as e:
            logging.error(f"Failed to process text file {file_path}: {e}")
            return None

    def process_file(self, file_path):
        """Determine file type and process accordingly."""
        logging.info(f"Starting processing for file: {file_path}")
        if file_path.endswith(".json"):
            return self.process_json(file_path)
        else:
            return self.process_text_file(file_path)


def process_file_wrapper(file_path, chunk_size, chunk_overlap, chromadb_path, collection_name, embedding_model_name):
    """Wrapper function to process a file using FileProcessor."""
    try:
        chunker = Chunker()
        chromadb_store = EmbedChromaDB(
            chromadb_path=chromadb_path,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
        )
        file_processor = FileProcessor(chunker, chromadb_store, chunk_size, chunk_overlap)
        return file_processor.process_file(file_path)

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None


def move_file(file_path, destination_dir):
    """Move a file to the specified directory."""
    try:
        os.makedirs(destination_dir, exist_ok=True)
        dest_path = os.path.join(destination_dir, os.path.basename(file_path))
        shutil.move(file_path, dest_path)
        logging.info(f"Moved {file_path} to {destination_dir}")
    except Exception as e:
        logging.error(f"Failed to move {file_path} to {destination_dir}: {e}")


def store_embeddings(chromadb_store, embedding_queue, args):
    """Stores embeddings sequentially while allowing concurrent processing."""
    while True:
        while not embedding_queue.empty():
            res = embedding_queue.get()
            try:
                logging.info(f"Storing embeddings for source: {res['source']}")
                chromadb_store.store_embeddings(
                    texts=res["texts"],
                    embeddings=res["embeddings"],
                    source=res["source"],
                    metadata=res["metadata"],
                )
                logging.info(f"Completed storing embeddings for source: {res['source']}")

                if args.processed_dir:
                    move_file(res["file_path"], args.processed_dir)
                elif args.delete:
                    os.remove(res["file_path"])
            except Exception as e:
                logging.error(f"Failed to store embeddings for {res['file_path']}: {e}")
                if args.failed_dir:
                    move_file(res["file_path"], args.failed_dir)

        time.sleep(1)  # Prevent busy looping


def run_daemon(args):
    """Watches a directory for new files and processes them concurrently."""
    logging.info(f"Watching directory: {args.watch_dir} for new files...")

    pool = Pool(args.concurrency)
    manager = Manager()
    embedding_queue = manager.Queue()

    chromadb_store = EmbedChromaDB(
        chromadb_path=args.chromadb_path,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model_name,
    )

    observer = Observer()
    observer.schedule(DirectoryWatcher(args, pool, embedding_queue), path=args.watch_dir, recursive=False)
    observer.start()

    try:
        store_embeddings(chromadb_store, embedding_queue, args)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


class DirectoryWatcher(FileSystemEventHandler):
    """Watches a directory and processes new files as they appear."""

    def __init__(self, args, pool, embedding_queue):
        self.args = args
        self.pool = pool
        self.embedding_queue = embedding_queue

    def on_created(self, event):
        """Triggered when a new file appears in the directory."""
        if not event.is_directory:
            file_path = event.src_path
            logging.info(f"New file detected: {file_path}")

            self.pool.apply_async(process_file_wrapper, args=(
                file_path, self.args.chunk_size, self.args.chunk_overlap,
                self.args.chromadb_path, self.args.collection_name, self.args.embedding_model_name
            ), callback=self.embedding_queue.put)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="File processing tool with JSON and text embedding support.")
    parser.add_argument("--watch_dir", type=str, default=os.getenv("EMBED_FILES_WATCH_DIR"))
    parser.add_argument("--processed_dir", type=str, default=os.getenv("EMBED_FILES_PROCESSED_DIR"))
    parser.add_argument("--failed_dir", type=str, default=os.getenv("EMBED_FILES_FAILED_DIR"))
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("EMBED_FILES_CONCURRENCY", 5)))

    args = parser.parse_args()

    if args.watch_dir:
        run_daemon(args)
    else:
        print("Error: Provide --watch_dir to enable daemon mode.")
