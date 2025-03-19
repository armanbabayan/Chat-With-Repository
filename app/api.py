from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
from langchain_openai import ChatOpenAI
from app.config.config import get_config
from loaders.git_loader import git_loader
from splitters.text_splitter import get_python_splitter, get_markdown_splitter
from qdrant.qdrant import get_client
from embeddings.points import PointsCreator
from qdrant.qdrant_store import QdrantStore
from qdrant_client import models as qdrant_models
from encoders.encoder import get_code_encoder, get_text_encoder
from utils.answer_questions import answer_question
from retrievers import RetrieverFactory, QueryClassifier, Reranker
from chains import QAChainBuilder

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qdrant_client = get_client(config=get_config())
OpenAI_KEY = os.getenv("OPENAI_KEY")

app = FastAPI()


class UrlModel(BaseModel):
    url: str


class QueryModel(BaseModel):
    query: str


@app.post("/create_knowledge_base/")
def create_knowledge_base(url_model: UrlModel):
    url = url_model.url
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    try:
        logger.info(f"Loading documents from repository: {url}")
        # Load documents from the repository
        python_documents, md_documents = git_loader(repo_url=url, branch="main")

        logger.info("Splitting documents into chunks")
        # Split documents into chunks
        python_splitter = get_python_splitter()
        md_splitter = get_markdown_splitter()
        python_chunks = python_splitter.split_documents(python_documents)
        md_chunks = md_splitter.split_documents(md_documents)

        logger.info("Creating points for the vector store")
        # Create points for the vector store
        points_creator = PointsCreator(qdrant_models)
        code_points = points_creator.create_code_points(python_chunks)
        md_points = points_creator.create_text_points(md_chunks)

        logger.info("Creating collections in the vector store")
        # Create collections in the vector store
        vectorstore = QdrantStore(qdrant_client, qdrant_models)
        vectorstore.create_collection_from_encoder(
            collection_name="md_collection",
            encoder=get_text_encoder(),
            distance="COSINE",
        )
        vectorstore.create_collection_from_encoder(
            collection_name="code_collection",
            encoder=get_code_encoder(),
            distance="COSINE",
        )

        logger.info("Uploading points to the vector store")
        # Upload points to the vector store
        vectorstore.upload_points(collection_name="md_collection", points=md_points)
        vectorstore.upload_points(collection_name="code_collection", points=code_points)

        return {"message": "Knowledge base created successfully"}
    except Exception as e:
        logger.error(f"Error creating knowledge base: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/get_answer/")
def get_answer(query_model: QueryModel):
    query = query_model.query
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        logger.info(f"Processing query: {query}")
        llm = ChatOpenAI(
            model="gpt-4o", openai_api_key=OpenAI_KEY, temperature=0, max_retries=2
        )

        # Create retriever factory
        factory = RetrieverFactory(qdrant_client)
        # Create retrievers
        md_retriever = factory.get_retriever(
            collection_name="md_collection", embedding_model_name="all-MiniLM-L6-v2"
        )

        code_retriever = factory.get_retriever(
            collection_name="code_collection",
            embedding_model_name="microsoft/codebert-base",
        )

        classifier = QueryClassifier(llm)

        # Create reranker
        reranker = Reranker(
            model_name="BAAI/bge-reranker-base", max_length=512, top_k=3
        )

        # Create QA chain builder
        chain_builder = QAChainBuilder(llm)

        ensemble_retriever = factory.create_ensemble_retriever(
            retrievers=[md_retriever, code_retriever],
            weights=[0.5, 0.5],  # Equal weights initially
        )

        dynamic_chain = chain_builder.build_with_dynamic_weights(
            retrievers=[md_retriever, code_retriever],
            classifier=classifier,
            reranker=reranker,
        )

        static_chain = chain_builder.build_with_ensemble(
            ensemble_retriever=ensemble_retriever, reranker=reranker
        )

        answer = answer_question(query, dynamic_chain=dynamic_chain)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error getting answer: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
