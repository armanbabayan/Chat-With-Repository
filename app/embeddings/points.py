from tqdm import tqdm
from app.encoders.encoder import get_code_encoder, get_text_encoder


class PointsCreator:
    """Class for creating vector database points from documents."""

    def __init__(self, models, code_encoder=None, text_encoder=None):

        """
        Initialize the points creator.
        :param models: Module containing the PointStruct class
        :param code_encoder: Optional pre-initialized code encoder
        :param text_encoder: Optional pre-initialized text encoder
        """

        self.models = models
        self.code_encoder = code_encoder or get_code_encoder()
        self.text_encoder = text_encoder or get_text_encoder()

    def create_code_points(
        self, documents, start_id=0, batch_size=32, show_progress=True
    ):

        """
        Create points for code documents.

        :param list documents: List of documents to encode
        :param int start_id: Starting ID for the points
        :param int batch_size: Batch size for encoding
        :param bool show_progress: Whether to show a progress bar. Default is True.
        :return list: List of point structures ready for database insertion
        """

        return self._create_points(
            documents, self.code_encoder, "code", start_id, batch_size, show_progress
        )

    def create_text_points(
        self, documents, start_id=0, batch_size=32, show_progress=True
    ):
        """
        Create points for text documents.

        :param list documents: List of documents to encode
        :param int start_id: Starting ID for the points
        :param int batch_size: Batch size for encoding
        :param bool show_progress: Whether to show a progress bar. Default is True.
        :return list: List of point structures ready for database insertion
        """

        return self._create_points(
            documents, self.text_encoder, "text", start_id, batch_size, show_progress
        )

    def _create_points(
        self, documents, encoder, doc_type, start_id, batch_size, show_progress
    ):
        """
        Create points for documents.
        :param list documents: List of documents to encode
        :param SentenceTransformer encoder: The encoder to use
        :param str doc_type: Document type identifier
        :param int start_id: Starting ID for the points
        :param int batch_size: Batch size for encoding
        :param bool show_progress: Whether to show a progress bar. Default is True.
        :return list: List of points structures ready for database insertion
        """

        points = []
        iterator = enumerate(documents)

        if show_progress:
            total = len(documents) if hasattr(documents, "__len__") else None
            iterator = tqdm(iterator, total=total)

        # Process documents in batches
        current_batch = []
        current_indices = []

        for idx, doc in iterator:
            current_batch.append(doc.page_content)
            current_indices.append(idx)

            # Process batch when it reaches the desired size
            if len(current_batch) >= batch_size:
                batch_vectors = encoder.encode(current_batch)

                for i, vector in enumerate(batch_vectors):
                    doc_idx = current_indices[i]
                    doc = documents[doc_idx]

                    point = self.models.PointStruct(
                        id=start_id + doc_idx,
                        vector=vector,
                        payload={
                            "metadata": doc.metadata,
                            "type": doc_type,
                            "page_content": doc.page_content,
                        },
                    )
                    points.append(point)

                # Reset batch
                current_batch = []
                current_indices = []

        # Process any remaining documents
        if current_batch:
            batch_vectors = encoder.encode(current_batch)

            for i, vector in enumerate(batch_vectors):
                doc_idx = current_indices[i]
                doc = documents[doc_idx]

                point = self.models.PointStruct(
                    id=start_id + doc_idx,
                    vector=vector,
                    payload={
                        "metadata": doc.metadata,
                        "type": doc_type,
                        "page_content": doc.page_content,
                    },
                )
                points.append(point)

        return points
