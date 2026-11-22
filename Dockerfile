# ==============================================================================
# TicketPilot Production Dockerfile
# Multi-mode container: FastAPI Backend (8000) & Streamlit UI (8501)
# ==============================================================================

FROM python:3.12-slim

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT_API=8000 \
    PORT_UI=8501

WORKDIR /app

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy Knowledge Base and Vector Index
COPY data/knowledge_base_clean.csv data/knowledge_base_clean.csv
COPY data/test.csv data/test.csv
COPY chroma_db/ chroma_db/
COPY models/ models/

# Copy application source code
COPY api.py .
COPY streamlit_app.py .
COPY routing_policy.py .
COPY decision_logger.py .
COPY text_cleaning.py .
COPY day14_reply_prompt.py .
COPY day15_end_to_end_pipeline.py .
COPY day24_dashboard.py .
COPY charts/ charts/

# Create logs directory
RUN mkdir -p logs

# Copy and setup entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Healthcheck to verify FastAPI backend readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["both"]
