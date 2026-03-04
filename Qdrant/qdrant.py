from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue, PointStruct
from pathlib import Path
import uuid

class QdrantStore:
    def __init__(self, collection_name: str, dim: int, persist_dir: str):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = QdrantClient(
            path=str(self.persist_dir)   # LƯU TRÊN Ổ CỨNG
        )

        self.collection_name = collection_name
        self.dim = dim

        self._init_collection()

    def _init_collection(self):
        collections = [
            c.name for c in self.client.get_collections().collections
        ]

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE
                )
            )

    def to_uuid(self, val: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, val))

    def upsert_chunks(self, chunks: list):
        points = []

        for ch in chunks:
            payload = {
                "text": ch["text"],
                "title": ch.get("title", "")
            }

            if "metadata" in ch:
                payload.update(ch["metadata"])

            uid = self.to_uuid(str(ch["_id"]))   # 🔥 FIX Ở ĐÂY

            points.append(
                PointStruct(
                    id=uid,
                    vector=ch["vector"],
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector, top_k=10):
        # Sử dụng query_points là phương thức mạnh mẽ và ổn định nhất hiện nay
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector, # Đối với search vector thuần, truyền trực tiếp vector vào
            limit=top_k,
            with_payload=True # Đảm bảo lấy được text và metadata
        ).points

        hits = []
        for r in results:
            hits.append({
                "_id": r.id,
                "score": r.score,
                "text": r.payload.get("text", ""),
                "metadata": r.payload, # Trả về full metadata để bên Streamlit dùng
                "title": r.payload.get("title", "")
            })

        return hits

    def drop_collection(self):
        self.client.delete_collection(
            collection_name=self.collection_name
        )

    def delete_by_title(self, title: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="title",
                        match=MatchValue(value=title)
                    )
                ]
            )
        )

    def reset(self):
        self.drop_collection()
        self._init_collection()
