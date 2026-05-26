import os
import chromadb
from chromadb.config import Settings

persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
os.makedirs(persist_dir, exist_ok=True)

class MemoryStore:
    def __init__(self, session_id: str, embed_router=None):
        self.session_id = session_id
        self.embed = embed_router

        # Новый API ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=f"session_{session_id}",
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, role: str, text: str):
        self.collection.add(
            documents=[text],
            metadatas=[{"role": role}],
            ids=[f"{self.session_id}_{self.collection.count()}"]
        )

    def get_context(self, query: str, recent_n: int = 5, relevant_n: int = 3):
        if not self.embed:
            return []

        # Recent messages
        recent = self.collection.peek(limit=recent_n) if self.collection.count() > 0 else {"documents": []}
        recent_docs = recent.get("documents", []) or []

        # Relevant by embedding
        try:
            q_embed = self.embed.embed([query])
            relevant = self.collection.query(
                query_embeddings=q_embed,
                n_results=min(relevant_n, self.collection.count()),
                include=["documents"]
            )
            rel_docs = relevant.get("documents", [[]])[0] if relevant else []
        except Exception:
            rel_docs = []

        # Combine and deduplicate
        seen = set()
        context = []
        for doc in list(recent_docs) + list(rel_docs):
            if doc and doc not in seen:
                seen.add(doc)
                context.append({"role": "user", "content": doc})

        return context[-recent_n:] if len(context) > recent_n else context

    def persist(self):
        pass  # PersistentClient сохраняет автоматически