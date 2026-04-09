FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY embedding_pipeline.py .
COPY generate_viz_data.py .
COPY corpus/ ./corpus/

# Switch to non-root user
USER user

# HF Spaces uses port 7860
EXPOSE 7860

# Pre-download the model at build time so startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

CMD ["python", "generate_viz_data.py", "--serve", "--port", "7860"]
