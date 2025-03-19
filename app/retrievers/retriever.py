from langchain_qdrant import Qdrant
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger


class RetrieverFactory:
    """Factory class for creating and managing retrievers."""

    def __init__(self, qdrant_client):
        """
        Initialize the retriever factory.

        :param qdrant_client: Qdrant client instance
        """
        self.qdrant_client = qdrant_client
        self.embeddings_cache = {}
        self.vectorstore_cache = {}
        self.retriever_cache = {}

    def get_embeddings(self, model_name):
        """
        Get HuggingFace embeddings model.

        :param str model_name: Name of the embeddings model
        :return: HuggingFaceEmbeddings: Embeddings model
        """
        if model_name not in self.embeddings_cache:
            logger.info(f"Loading embeddings model: {model_name}")
            self.embeddings_cache[model_name] = HuggingFaceEmbeddings(
                model_name=model_name, cache_folder="retrievers/cache"
            )
            logger.success(f"Embeddings model {model_name} loaded successfully.")
        return self.embeddings_cache[model_name]

    def get_vectorstore(self, collection_name, embedding_model_name):
        """
        Get or create a vector store for a collection.
        :param str collection_name: Name of the Qdrant collection
        :param embedding_model_name: Name of the embeddings model
        :return: Vector store instance
        """
        cache_key = f"{collection_name}_{embedding_model_name}"

        if cache_key not in self.vectorstore_cache:
            logger.info(
                f"Creating vector store for collection: {collection_name} with embedding model: {embedding_model_name}"
            )
            embeddings = self.get_embeddings(embedding_model_name)

            self.vectorstore_cache[cache_key] = Qdrant(
                client=self.qdrant_client,
                collection_name=collection_name,
                embeddings=embeddings,
            )
            logger.success(
                f"Vector store created for collection: {collection_name} with embedding model: {embedding_model_name}"
            )

        return self.vectorstore_cache[cache_key]

    def get_retriever(
        self, collection_name, embedding_model_name, search_type="mmr", k=5, **kwargs
    ):
        """
        Get a retriever for a collection.
        :param str collection_name: Name of the Qdrant collection
        :param str embedding_model_name: Name of the embeddings model
        :param str search_type: Type of search (e.g., "mmr")
        :param int k: Number of documents to retrieve
        :param dict kwargs: Additional search parameters
        :return: Retriever instance
        """
        cache_key = f"{collection_name}_{embedding_model_name}_{search_type}_{k}"

        if cache_key not in self.retriever_cache:
            logger.info(
                f"Creating retriever for collection: {collection_name} with embedding model: {embedding_model_name}, search type: {search_type}, k: {k}"
            )
            vectorstore = self.get_vectorstore(collection_name, embedding_model_name)

            search_kwargs = {"k": k, **kwargs}

            self.retriever_cache[cache_key] = vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
            logger.success(
                f"Retriever created for collection: {collection_name} with embedding model: {embedding_model_name}, search type: {search_type}, k: {k}"
            )

        return self.retriever_cache[cache_key]

    @staticmethod
    def create_ensemble_retriever(retrievers, weights=None):
        """
        Create an ensemble retriever from multiple retrievers.

        :param list retrievers: List of retriever instances
        :param list weights: List of weights for each retriever
        :return EnsembleRetriever: Ensemble retriever instance
        """
        if weights is None:
            weights = [1.0 / len(retrievers)] * len(retrievers)

        return EnsembleRetriever(retrievers=retrievers, weights=weights)
