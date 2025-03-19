from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Config(BaseModel):
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_port: int
    qdrant_grpc_port: int
    openai_api_key: str | None


def get_config() -> Config:

    return Config(
        qdrant_url=str(os.getenv("QDRANT_URL")),
        qdrant_api_key=str(os.getenv("QDRANT_KEY")),
        qdrant_port=int(os.getenv("QDRANT_PORT", 6333)),
        qdrant_grpc_port=int(os.getenv("QDRANT_GRPC_PORT", 6334)),
        openai_api_key=os.getenv("OPENAI_KEY"),
    )
