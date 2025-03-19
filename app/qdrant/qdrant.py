from qdrant_client import QdrantClient

from app.config.config import Config


def get_client(config: Config):
    qdrant_client = QdrantClient(
        url=config.qdrant_url,
        port=config.qdrant_port,
        grpc_port=config.qdrant_grpc_port,
        api_key=config.qdrant_api_key,
        prefer_grpc=False,
    )

    return qdrant_client
