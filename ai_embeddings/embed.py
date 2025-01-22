import os
import mimetypes
import time
from datetime import datetime 
from pydantic import BaseModel, Field, field_validator, model_validator
from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Callable, Type, Optional
import chromadb


class ChunkedData(BaseModel):
    docs: List[str] = Field(description="A list of strings should be processed and stored in the vector database")
    source: str = Field(description="The source of the metadata.")
    index_on: datetime = Field(description="The date the source was processed")
    metadata:  Dict[str, str] = Field(
        default_factory=dict, 
        description="Additional metadata fields."
    )

class ChunkInput(BaseModel):
    path: Optional[str] = Field(
        default=None, description="Path to the file to chunk (optional if 'text' is provided)."
    )
    text: Optional[str] = Field(
        default=None, description="Text to be chunked (optional if 'path' is provided)."
    )
    chunk_size: int = Field(description="Size of chunks", default=400)
    chunk_overlap: int = Field(description="Chunk overlap", default=100)
    separators: List[str] = Field(
        description="Separators for the text splitter", default=["\n\n", "\n", " ", ""]
    )
    splitter: Type[TextSplitter] = Field(
        description="A class reference for a text splitter extending TextSplitter. This will be instantiated later."
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional key-value pairs as metadata. Keys and values must be strings.",
    )

    @field_validator("splitter", mode="before")
    @classmethod
    def validate_splitter(cls, value):
        if not isinstance(value, type) or not issubclass(value, TextSplitter):
            raise ValueError("The splitter must be a class inheriting from TextSplitter.")
        return value

    @model_validator(mode="after")
    def validate_path_or_text(cls, self):
        """
        Validate that either 'path' or 'text' is provided, but not both.
        """
        errors = []
        if not self.path and not self.text:
            errors.append("You must provide either 'path' or 'text'.")
        if self.path and self.text:
            errors.append("You cannot provide both 'path' and 'text'.")

        # Validate the path if it is provided
        if self.path and not os.path.exists(self.path):
            errors.append(f"The file path '{self.path}' does not exist.")

        if errors:
            raise ValueError("\n".join(errors))

        return self

class Chunker:
    # MIME type to loader mapping
    mime_type_to_loader: Dict[str, Callable[[str], str]] = {
        "application/pdf": "load_pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "load_docx",
        # Specific mappings can remain here
    }

    def __init__(self, ci: ChunkInput):
        self.ci = ci

    def load(self) -> str:
        """
        Load text from the provided path using the appropriate loader.

        Returns:
            str: The loaded text.

        Raises:
            ValueError: If the MIME type is invalid or unsupported.
        """
        if not self.ci.path:
            raise ValueError("No file path provided in the input.")

        mime_type, _ = mimetypes.guess_type(self.ci.path)

        if not mime_type:
            raise ValueError(f"Could not determine MIME type for file: {self.ci.path}")

        # Use the specific loader if available
        loader_name = self.mime_type_to_loader.get(mime_type)

        # Fallback for all MIME types starting with "text/"
        if not loader_name and mime_type.startswith("text/"):
            loader_name = "load_text_file"

        if not loader_name:
            raise ValueError(f"Unsupported MIME type: {mime_type}")

        # Call the appropriate loader function
        loader = getattr(self, loader_name)
        return loader(self.ci.path)

    def load_pdf(self, path: str) -> str:
        """
        Loads text from a PDF file.

        Args:
            path (str): Path to the PDF file.

        Returns:
            str: Extracted text from the PDF.
        """
        reader = PdfReader(path)
        return " ".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )

    def load_docx(self, path: str) -> str:
        """
        Loads text from a DOCX file.

        Args:
            path (str): Path to the DOCX file.

        Returns:
            str: Extracted text from the DOCX.
        """
        loader = Docx2txtLoader(path)
        documents = loader.load()
        return " ".join(doc.page_content for doc in documents)

    def load_text_file(self, path: str) -> str:
        """
        Loads text from plain text, markdown, or CSV files.

        Args:
            path (str): Path to the text-based file.

        Returns:
            str: Content of the file as a string.
        """
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def chunk(self) -> ChunkedData:
        """
        Splits the loaded text into chunks using the specified splitter and returns a ChunkedData object.

        Returns:
            ChunkedData: The result containing the chunks and associated metadata.
        """
        # Load the text (file or inline text)
        if self.ci.path:
            text = self.load()
        else:
            text = self.ci.text

        # Initialize the text splitter from the ChunkInput
        text_splitter = self.ci.splitter(
            chunk_size=self.ci.chunk_size,
            chunk_overlap=self.ci.chunk_overlap,
            separators=self.ci.separators,
        )

        # Split the text into chunks
        chunks = text_splitter.split_text(text)

        # Return a ChunkedData object
        return ChunkedData(
            docs=chunks,
            source=self.ci.path if self.ci.path else "Inline Text",
            index_on=datetime.now(),
            metadata=self.ci.metadata,
        )

    def validate_mimetype(self, path: str):
        """
        Validates the MIME type of the given file path.

        Args:
            path (str): The path to the file to validate.

        Raises:
            ValueError: If the file's MIME type is not supported.
        """
        mime_type, _ = mimetypes.guess_type(path)

        if not mime_type.startswith("text/") and mime_type not in self.mime_type_to_loader:
            raise ValueError(
                f"The file '{path}' has an invalid MIME type '{mime_type}'. "
                f"Supported MIME types are: {', '.join(self.mime_type_to_loader.keys())} or any 'text/' type."
            )

class ChromaDBEmbeddingStore:
    def __init__(
        self,
        chunked_data: ChunkedData,
        chromadb_path: str,
        collection_name: str,
        embedding_model_name: str,
    ):
        """
        Initializes the ChromaDBEmbeddingStore.

        Args:
            chunked_data (ChunkedData): The input ChunkedData object containing texts and metadata.
            chromadb_path (str): Path to the ChromaDB storage directory.
            collection_name (str): Name of the ChromaDB collection.
            embedding_model_name (str): Name of the OpenAI embedding model.
        """
        self.chunked_data = chunked_data
        self.chromadb_path = chromadb_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

    def store_embeddings(self) -> Dict:
        """
        Generates embeddings using OpenAI and stores them in ChromaDB.

        Returns:
            dict: Summary of the operation.
        """
        try:
            # Initialize the embedding model
            embedding_model = OpenAIEmbeddings(model=self.embedding_model_name)
            embed_func = lambda doc: embedding_model.embed_query(doc)

            # Extract texts and metadata
            texts = self.chunked_data.docs
            sources = [self.chunked_data.source] * len(texts)
            index_on = self.chunked_data.index_on.isoformat()
            additional_metadata = self.chunked_data.metadata

            # Generate embeddings
            start_time = time.time()
            embeddings = [embed_func(text) for text in texts]
            stop_time = time.time()

            # Initialize ChromaDB client and collection
            client = chromadb.PersistentClient(path=self.chromadb_path)
            collection = client.get_or_create_collection(name=self.collection_name)
            ids = [f"doc_{i}" for i in range(len(texts))]
            metadatas = [
                {
                    **additional_metadata,
                    "source": source,
                    "index_on": index_on,
                }
                for source in sources
            ]

            # Add documents to the collection
            collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )

            # Return operation summary
            return {
                "status": f"Successfully stored {len(texts)} documents.",
                "success": True,
                "chromadb_path": self.chromadb_path,
                "collection_name": self.collection_name,
                "embedding_model_name": self.embedding_model_name,
                "start_time": datetime.fromtimestamp(start_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "stop_time": datetime.fromtimestamp(stop_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "duration_seconds": stop_time - start_time,
                "number_of_documents": len(texts),
                "total_document_size": sum(len(text) for text in texts),
            }
        except Exception as e:
            return {
                "status": f"An error occurred: {str(e)}",
                "success": False,
                "chromadb_path": self.chromadb_path,
                "collection_name": self.collection_name,
                "embedding_model_name": self.embedding_model_name,
                "number_of_documents": len(self.chunked_data.docs),
                "total_document_size": sum(len(doc) for doc in self.chunked_data.docs),
            }


