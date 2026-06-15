"""
Tenant-aware search engine.

This PoC uses TF-IDF instead of a paid embedding API so it can run locally.
The security point is the same:
- attach tenant_id to each document
- filter by tenant_id before generating an AI answer
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SearchResult:
    doc_id: str
    tenant_id: str
    title: str
    summary: str
    content: str
    score: float

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "tenant_id": self.tenant_id,
            "title": self.title,
            "summary": self.summary,
            "content": self.content,
            "score": self.score,
        }


class TenantSearchEngine:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.doc_matrix = None

    def build_index(self):
        if not self.documents:
            raise ValueError("documents is empty")

        texts = [
            f"{doc['title']} {doc['summary']} {doc['content']}"
            for doc in self.documents
        ]
        self.doc_matrix = self.vectorizer.fit_transform(texts)
        return self.doc_matrix.shape

    def search_for_tenant(
        self,
        query: str,
        tenant_id: str,
        role: str,
        top_k: int = 3,
        threshold: float = 0.05,
    ) -> list[dict]:
        """
        Safe search.
        Admin can see all documents for PoC comparison.
        Normal users can see only documents with the same tenant_id.
        """
        if role == "admin":
            allowed_indices = list(range(len(self.documents)))
        else:
            allowed_indices = [
                i for i, doc in enumerate(self.documents)
                if doc["tenant_id"] == tenant_id
            ]

        return self._search_indices(query, allowed_indices, top_k, threshold)

    def search_all_tenants(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.05,
    ) -> list[dict]:
        """Unsafe comparison search. Do not pass this directly to the AI."""
        return self._search_indices(
            query,
            list(range(len(self.documents))),
            top_k,
            threshold,
        )

    def _search_indices(
        self,
        query: str,
        indices: list[int],
        top_k: int,
        threshold: float,
    ) -> list[dict]:
        if self.doc_matrix is None:
            raise ValueError("index is empty. Call build_index first.")
        if not indices:
            return []

        query_vec = self.vectorizer.transform([query])
        target_matrix = self.doc_matrix[indices]
        scores = cosine_similarity(query_vec, target_matrix).ravel()

        ranked = sorted(
            zip(indices, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results: list[dict] = []
        for idx, score in ranked[:top_k]:
            if score < threshold:
                continue
            doc = self.documents[idx]
            results.append(
                SearchResult(
                    doc_id=doc["doc_id"],
                    tenant_id=doc["tenant_id"],
                    title=doc["title"],
                    summary=doc["summary"],
                    content=doc["content"],
                    score=float(score),
                ).as_dict()
            )

        return results


def classify_result_type(safe_results: list[dict], unsafe_results: list[dict]) -> str:
    """
    result_type:
    - answered: current user has accessible documents
    - no_permission: other-tenant documents exist, but current user cannot access them
    - no_information: no tenant has relevant documents
    """
    if safe_results:
        return "answered"
    if unsafe_results:
        return "no_permission"
    return "no_information"
