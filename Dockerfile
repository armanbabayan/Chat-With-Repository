FROM python:3.12-slim

LABEL authors="armanbabayan"

RUN apt-get update && apt-get install -y git
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
RUN pip install "poetry==1.8.3"
COPY pyproject.toml poetry.lock ./
RUN poetry self add poetry-plugin-export@1.8.0
RUN poetry export -f requirements.txt --without-hashes --without-urls >> requirements.txt

RUN cat requirements.txt
RUN pip install -r requirements.txt

COPY app app/
COPY .env .env

EXPOSE 8000
EXPOSE 8501

# Command to run both FastAPI and Streamlit
CMD ["sh", "-c", "poetry run uvicorn app.api:app --host 0.0.0.0 --port 8000 & sleep 5 && poetry run streamlit run app/streamlit_app.py"]
