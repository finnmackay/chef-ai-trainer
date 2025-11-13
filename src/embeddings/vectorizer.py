"""Create vector embeddings from chef transcripts using open-source models."""

import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import numpy as np


class ChefVectorizer:
    """Create and manage vector embeddings for chef transcripts."""

    def __init__(
        self,
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_db_path: str = "./data/vector_db"
    ):
        """
        Initialize vectorizer.

        Args:
            embedding_model: Sentence transformer model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            vector_db_path: Path to store vector database
        """
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_db_path = vector_db_path

        print(f"Loading embedding model: {embedding_model}")
        # Initialize sentence transformer
        # Use trust_remote_code for nomic models
        self.embedding_model = SentenceTransformer(
            embedding_model,
            trust_remote_code=True
        )
        print(f"Model loaded! Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")

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

            # Simple text splitter
            text_chunks = self._split_text(content)

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

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def create_vector_store(self, chunks: List[Dict[str, Any]], collection_name: str = "chef_transcripts") -> chromadb.Collection:
        """
        Create vector store from chunks.

        Args:
            chunks: List of text chunks with metadata
            collection_name: Name of ChromaDB collection

        Returns:
            ChromaDB collection
        """
        print(f"Creating vector store with {len(chunks)} chunks...")

        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=self.vector_db_path)

        # Delete collection if it exists
        try:
            client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except:
            pass

        # Create collection
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Prepare data
        texts = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Add to collection in batches
        batch_size = 500
        for i in tqdm(range(0, len(chunks), batch_size), desc="Adding to vector store"):
            batch_end = min(i + batch_size, len(chunks))

            collection.add(
                embeddings=embeddings[i:batch_end].tolist(),
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )

        print(f"Vector store created at {self.vector_db_path}")
        return collection

    def load_vector_store(self, collection_name: str = "chef_transcripts") -> chromadb.Collection:
        """
        Load existing vector store.

        Args:
            collection_name: Name of ChromaDB collection

        Returns:
            ChromaDB collection
        """
        if not os.path.exists(self.vector_db_path):
            raise ValueError(f"Vector store not found at {self.vector_db_path}")

        client = chromadb.PersistentClient(path=self.vector_db_path)
        collection = client.get_collection(name=collection_name)

        print(f"Loaded vector store from {self.vector_db_path}")
        print(f"Collection has {collection.count()} items")
        return collection

    def search(self, collection: chromadb.Collection, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search vector store for relevant chunks.

        Args:
            collection: ChromaDB collection
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        # Search
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i]  # Convert distance to similarity
            })

        return formatted_results
