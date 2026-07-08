import os
import chromadb

class VectorStore:
    def __init__(self, persist_directory="workspace/chroma"):
        """Initialize the persistent ChromaDB vector store."""
        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize the persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create a collection for our document chunks
        # ChromaDB automatically uses all-MiniLM-L6-v2 to embed the documents for free
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"description": "Raw text chunks from mounted documents"}
        )

    def upsert_chunk(self, chunk_id: str, text: str, metadata: dict = None):
        """
        Upserts a text chunk into the vector database.
        ChromaDB automatically generates an embedding for the text.
        """
        if metadata is None or len(metadata) == 0:
            metadata = {"source": "unknown"}
            
        self.collection.upsert(
            documents=[text],
            ids=[chunk_id],
            metadatas=[metadata]
        )
        
    def search_chunks(self, query: str, n_results: int = 5):
        """
        Searches the vector database for chunks semantically similar to the query.
        Returns a dictionary containing distances, metadatas, and documents.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

    def remove_by_file(self, filename: str):
        """Removes all chunks associated with a specific file from the vector store."""
        try:
            self.collection.delete(where={"file": filename})
            print(f"  [VectorStore] Removed all chunks for {filename}")
        except Exception as e:
            print(f"  [VectorStore] Failed to remove chunks for {filename}: {e}")
