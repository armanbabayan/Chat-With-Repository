from sentence_transformers import CrossEncoder
from typing import Any, Tuple


class Reranker:
    """Class for reranking retrieved documents."""

    def __init__(self, model_name="BAAI/bge-reranker-base", max_length=512, top_k=3):

        self.cross_encoder = CrossEncoder(
            model_name, max_length=max_length, trust_remote_code=True
        )
        self.top_k = top_k

    def rerank(self, query: str, documents: list) -> list:
        """
         Rerank documents based on relevance to the query.
        :param str query: Original user query
        :param list documents:  List of retrieved documents
        :return list:  Reranked documents
        """
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        reranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked[: self.top_k]]

    def format_for_prompt(self, results: dict[str, Any]) -> dict[str, Any]:

        """
        Format retrieval results for prompt after reranking.
        :param dict results: Dictionary with query and retrieved documents
        :return dict: Formatted dictionary with reranked context
        """

        if not results or "context" not in results or "question" not in results:
            return {"context": "", "question": ""}

        query = results["question"]
        documents = results["context"]
        reranked_docs = self.rerank(query, documents)

        context = "\n\n".join([doc.page_content for doc in reranked_docs])

        return {"context": context, "question": query}
