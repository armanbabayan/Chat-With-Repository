
# Chat With Repository


Welcome to this repository, an advanced language model-based tool designed to assist with exploring, analyzing, and interacting with repositories. This project integrates natural language processing to provide intelligent responses, making it easier to understand and navigate large codebases, documentation, and files within a repository.

## Key Features

* Separate Collections for Code and Documentation: Code files (currently only Python files) and markdown files are processed and stored in separate collections within the Qdrant database for optimized retrieval.
* Dual Model Embedding System:
    * Plain Text (Markdown) Embeddings: The all-MiniLM-L6-v2 model (embedding dimension: 384) is used for generating embeddings for plain text (documentation and markdown files). [Model Link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
    * Code Embeddings: The microsoft/codebert-base model (embedding dimension: 768) is used for generating embeddings from Python code files. Trained on large code datasets, this model captures the unique semantic patterns in programming code, offering more meaningful embeddings for code. [Model Link](https://huggingface.co/microsoft/codebert-base), [Paper Link](https://arxiv.org/abs/2002.08155).
* Retriever System: The retrieval system uses an Assemble Retriever approach that combines chunks from both the code and documentation collections, ensuring a balanced retrieval of contextually relevant information.
* Reranker: The BAAI/bge-reranker-base model (a ranking model) is used as the reranker, which assigns weights to the retrieved documents based on relevance. The reranker considers whether the query is more related to code or documentation, adjusting the weighting accordingly. [Model Link](https://huggingface.co/BAAI/bge-reranker-base).
* Langchain Framework: The entire query and retrieval chain is powered by the Langchain framework, which handles the processing of user queries and the orchestration of the underlying models. If a query is outside the scope of the system, it responds with "I don't know."
* FastAPI: The app is currently served using FastAPI.

## How It Works
* Create Knowledge Base: The "Create Knowledge Base" button allows users to generate embeddings and store them in Qdrant by providing a GitHub repository link. This process indexes the code and markdown files separately into distinct collections for efficient querying.

* Get Answer: The "Get Answer" button lets users ask questions related to the code or documentation. Based on the query, the system retrieves relevant information from the preprocessed collections (code and markdown). The GPT-4o model is used to generate human-readable answers.

* Question Classification: Queries are classified to determine if they are more code-related or documentation-related. This classification helps the reranker adjust the importance of the retrieved documents to give more relevant answers.


## Environment Variables
To run this project, you will need to add all environment variable to your .env file. Please refer to .env.example file

## Run Locally
## Prerequisites

Before using this tool, make sure you have the following prerequisites installed:

- Python 3.12.4
- [Poetry](https://python-poetry.org/): You can install it following the [Poetry installation guide](https://python-poetry.org/docs/#installation).
- All dependencies can be installed using poetry. Simply run `poetry install` command.

Clone the project

```bash
  https://github.com/armanbabayan/Chat-With-Repository.git
```

Go to the project directory

```bash
  cd Chat-With-Repository
```

Install dependencies

```bash
  poetry install
```
## Docker
To create an image run the following command
```bash
docker build -t chat_with_repo .
```

Run the image with this command

```bash
docker run -p 8000:8000 -p 8501:8501 chat_with_repo
```
## Tech Stack

**Language:**  Python 3

**Library:** LangChain, Streamlit

**Models:** OpenAI GPT 4o, HuggingFace Models

**Vectore Database:** Qdrant

## Roadmap

- Additional language support

- Show source of the answer for more trancparency

- Automated documentation generation based on codebase

- Proper evaluation metrics


## Authors

- [@armanbabayan](https://github.com/armanbabayan)
