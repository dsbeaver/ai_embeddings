import argparse
import os
import mimetypes
import time
import json
from typing import List, Dict
from datetime import datetime

import chromadb
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader


def store_embeddings_in_chromadb(
    documents: List[Dict[str, str]],
    chromadb_path: str,
    collection_name: str,
    embedding_model_name: str
) -> Dict:
    """
    Generates embeddings using OpenAI and stores them in ChromaDB.

    Args:
        documents (List[Dict[str, str]]): A list of dictionaries with 'text' and 'source'.
        chromadb_path (str): Path to the ChromaDB storage directory.
        collection_name (str): Name of the ChromaDB collection.
        embedding_model_name (str): Name of the OpenAI embedding model.

    Returns:
        dict: Summary of the operation.
    """
    try:
        embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        embed_func = lambda doc: embedding_model.embed_query(doc)

        texts = [doc["text"] for doc in documents]
        sources = [doc["source"] for doc in documents]

        start_time = time.time()
        embeddings = [embed_func(text) for text in texts]
        stop_time = time.time()

        client = chromadb.PersistentClient(path=chromadb_path)
        collection = client.get_or_create_collection(name=collection_name)
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = [{"source": source} for source in sources]

        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        return {
            "status": f"Successfully stored {len(documents)} documents.",
            "success": True,
            "start_time": datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            "stop_time": datetime.fromtimestamp(stop_time).strftime('%Y-%m-%d %H:%M:%S'),
            "duration_seconds": stop_time - start_time,
            "number_of_documents": len(documents),
            "total_document_size": sum(len(text) for text in texts)
        }
    except Exception as e:
        return {
            "status": f"An error occurred: {str(e)}",
            "success": False,
            "number_of_documents": len(documents),
            "total_document_size": sum(len(doc["text"]) for doc in documents)
        }


def chunk_text(text: str, file_path: str, chunk_size: int, chunk_overlap: int) -> List[dict]:
    """
    Splits text into chunks with overlap.

    Args:
        text (str): The text to split.
        file_path (str): Path to the source file (used for metadata).
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Characters overlapping between chunks.

    Returns:
        List[dict]: A list of text chunks with source metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return [{"text": chunk, "source": file_path} for chunk in chunks]


def create_chunks_from_file(file_path: str, chunk_size: int, chunk_overlap: int) -> List[dict]:
    """
    Reads a file and creates text chunks.

    Args:
        file_path (str): Path to the file.
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Characters overlapping between chunks.

    Returns:
        List[dict]: A list of text chunks with source metadata.
    """
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type == "application/pdf":
        reader = PdfReader(file_path)
        text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        text = " ".join(doc.page_content for doc in documents)
    elif mime_type == "text/markdown":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")

    return chunk_text(text, file_path, chunk_size, chunk_overlap)


def append_to_json_log(new_entry, file_path: str):
    """
    Appends a JSON entry to a log file.

    Args:
        new_entry (dict): Entry to log.
        file_path (str): Log file path.
    """
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump({"log": []}, f)

    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {"log": []}

    data["log"].append(new_entry)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files and store embeddings in ChromaDB.")
    parser.add_argument("input_path", type=str, help="Path to a file or directory.")
    parser.add_argument("--chromadb_path", type=str, default=os.getenv("CHROMADB_PATH"),
                        help="Path to ChromaDB storage (default: CHROMADB_PATH environment variable).")
    parser.add_argument("--collection_name", type=str, default="openai_text_embeddings",
                        help="Name of ChromaDB collection.")
    parser.add_argument("--chunk_size", type=int, default=400, help="Maximum characters per chunk.")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Characters overlapping between chunks.")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small",
                        help="Name of the OpenAI embedding model (default: text-embedding-3-small).")
    parser.add_argument("--log_file", type=str, default="log/embedding.log", help="Path to log file.")
    args = parser.parse_args()

    # Validate ChromaDB path
    if not args.chromadb_path:
        raise ValueError("ChromaDB path is required. Set the CHROMADB_PATH environment variable or use --chromadb_path.")

    if os.path.isdir(args.input_path):
        for file_name in os.listdir(args.input_path):
            file_path = os.path.join(args.input_path, file_name)
            if os.path.isfile(file_path):
                try:
                    chunks = create_chunks_from_file(file_path, args.chunk_size, args.chunk_overlap)
                    results = store_embeddings_in_chromadb(
                        chunks, args.chromadb_path, args.collection_name, args.embedding_model
                    )
                    append_to_json_log({'file': file_path, 'message': results, 'status': results['status']}, args.log_file)
                    print(f"Processed: {file_name}")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
    elif os.path.isfile(args.input_path):
        chunks = create_chunks_from_file(args.input_path, args.chunk_size, args.chunk_overlap)
        results = store_embeddings_in_chromadb(
            chunks, args.chromadb_path, args.collection_name, args.embedding_model
        )
        append_to_json_log({'file': args.input_path, 'message': results, 'status': results['status']}, args.log_file)
        print(f"Processed: {args.input_path}")
    else:
        print(f"Invalid input path: {args.input_path}")
