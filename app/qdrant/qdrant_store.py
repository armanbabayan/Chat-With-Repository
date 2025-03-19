from app.qdrant.config import Config


class QdrantStore:
    """Class for managing Qdrant collections and operations."""

    def __init__(self, client, models, config=None):
        """
        Initialize the Qdrant store.
        :param client: Qdrant client instance
        :param models: Module containing Qdrant models (VectorParams, Distance, etc.)
        :param config: Optional configuration object (defaults to global Config)
        """
        self.client = client
        self.models = models
        self.config = config or Config

    def create_collection(self, collection_name, vectors_config):
        """
        Create a new collection if it doesn't exist.

        :param str collection_name: Name of the collection
        :param vectors_config: VectorParams instance or dict
        :return bool: True if collection was created, False if it already existed
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if collection_name not in collection_names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )
            return True
        return False

    def create_collection_from_encoder(self, collection_name, encoder, distance=None):
        """
        Create a collection with parameters derived from an encoder.

        :param str collection_name: Name of the collection
        :param SentenceTransformer encoder: SentenceTransformer encoder
        :param str distance: Distance metric (e.g., "COSINE", "EUCLID", "DOT")
        :return: bool: True if collection was created, False if it already existed
        """
        # Get distance from config if not provided
        if distance is None:
            distance = self.config.get("vector_params.distance_metric", "COSINE")

        distance_metric = getattr(self.models.Distance, distance)

        vectors_config = self.models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=distance_metric,
        )

        # Create collection
        return self.create_collection(collection_name, vectors_config)

    def create_custom_collection(
        self, collection_name=None, vector_size=None, distance=None
    ):
        """
        Create a collection with custom parameters.

        :param str collection_name: Name of the collection
        :param int vector_size: Size of the vectors
        :param str distance: Distance metric
        :return: bool: True if collection was created, False if it already existed
        """
        # Use config values as defaults
        collection_name = collection_name or self.config.get(
            "collection_defaults.default_collection"
        )
        vector_size = vector_size or self.config.get("vector_params.vector_size")
        distance = distance or self.config.get("vector_params.distance_metric")

        # Get the appropriate distance metric
        distance_metric = getattr(self.models.Distance, distance)

        # Create vector params
        vectors_config = self.models.VectorParams(
            size=vector_size,
            distance=distance_metric,
        )

        # Create collection
        return self.create_collection(collection_name, vectors_config)

    def upload_points(self, collection_name, points, batch_size=None):
        """
        Upload points to the collection.

        :param str collection_name: Name of the collection
        :param list points: List of points to upload
        :param int batch_size: Size of batches for uploading
        :return: Upload operation results
        """
        # Use config value as default for batch_size
        if batch_size is None:
            batch_size = self.config.get("vector_params.batch_size", 100)

        results = []
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            result = self.client.upload_points(
                collection_name=collection_name, points=batch
            )
            results.append(result)
        return results

    def delete_collection(self, collection_name):

        """
        Delete a collection.

        :param str collection_name: Name of the collection to delete
        :return bool: True if deleted successfully
        """
        return self.client.delete_collection(collection_name=collection_name)
