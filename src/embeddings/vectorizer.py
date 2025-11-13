"""Create vector embeddings from chef transcripts."""

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
import os


class ChefVectorizer:
    """Create and manage vector embeddings for chef transcripts."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-large",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_db_path: str = "./data/vector_db"
    ):
        """
        Initialize vectorizer.

        Args:
            embedding_model: OpenAI embedding model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            vector_db_path: Path to store vector database
        """
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_db_path = vector_db_path

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def create_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents with content and metadata

        Returns:
            List of chunks with metadata
        """
        chunks = []

        print(f"Creating chunks from {len(documents)} documents...")

        for doc in tqdm(documents, desc="Chunking documents"):
            content = doc['content']
            metadata = doc['metadata']

            # Split text into chunks
            text_chunks = self.text_splitter.split_text(content)

            # Add metadata to each chunk
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    'content': chunk,
                    'metadata': {
                        **metadata,
                        'chunk_index': i,
                        'total_chunks': len(text_chunks)
                    }
                })

        print(f"Created {len(chunks)} chunks")
        return chunks

    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> Chroma:
        """
        Create vector store from chunks.

        Args:
            chunks: List of text chunks with metadata

        Returns:
            Chroma vector store
        """
        print(f"Creating vector store with {len(chunks)} chunks...")

        # Prepare texts and metadatas
        texts = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]

        # Create vector store
        vector_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=self.vector_db_path
        )

        print(f"Vector store created at {self.vector_db_path}")
        return vector_store

    def load_vector_store(self) -> Chroma:
        """
        Load existing vector store.

        Returns:
            Chroma vector store
        """
        if not os.path.exists(self.vector_db_path):
            raise ValueError(f"Vector store not found at {self.vector_db_path}")

        vector_store = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings
        )

        print(f"Loaded vector store from {self.vector_db_path}")
        return vector_store

    def search(self, vector_store: Chroma, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search vector store for relevant chunks.

        Args:
            vector_store: Chroma vector store
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with scores
        """
        results = vector_store.similarity_search_with_score(query, k=top_k)

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            })

        return formatted_results
