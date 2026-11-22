# Sprint 5 Day 25: Unit Testing & Docker Containerization

**Project**: TicketPilot — Enterprise Customer Support AI Triage & Grounded RAG Assistant  
**Date**: November 22, 2026  
**Status**: Completed & Verified  

---

## 1. Executive Summary

Day 25 concludes **Sprint 5: Make It a Product**. To guarantee production stability, regression protection, and cloud portability, we delivered:
1. **Production Unit Test Suites** covering data sanitization ([text_cleaning.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/text_cleaning.py), [test_cleaning.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_cleaning.py)), three-tier compound routing policy ([test_routing.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_routing.py)), and FastAPI REST API endpoints ([test_api.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_api.py)).
2. **Consolidated Master Test Runner** ([run_all_tests.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/run_all_tests.py)) orchestrating all 7 subsystem test suites with automated scorecard reporting.
3. **Production Docker Containerization Architecture** ([Dockerfile](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/Dockerfile), [docker-compose.yml](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/docker-compose.yml), [entrypoint.sh](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/entrypoint.sh), [.dockerignore](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/.dockerignore), and [requirements.txt](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/requirements.txt)).
4. **Automated Container Setup Validator** ([verify_docker_setup.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/verify_docker_setup.py)).

All **7/7 test suites passed with a 100% success rate**.

---

## 2. Comprehensive Test Execution Scorecard

All tests were executed through the unified master test runner ([run_all_tests.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/run_all_tests.py)):

| Test Suite | File | Status | Duration | Coverage Scope |
| :--- | :--- | :---: | :---: | :--- |
| **1. Text Cleaning & Sanitization** | `test_cleaning.py` | **PASS** | 0.40s | Empty inputs, HTML tag stripping, whitespace normalization, deduplication, batch pipelines |
| **2. Compound Routing Policy** | `test_routing.py` | **PASS** | 0.02s | Tier 1, 2, 3 boundary checks (tau=0.40, score=4.0), risk overrides, flag verification |
| **3. FastAPI REST Endpoints** | `test_api.py` | **PASS** | 10.36s | Lifespan model startup, `/triage`, `/draft-reply`, `/health`, `/decisions`, error schemas |
| **4. Decision Logger & Audit Streams** | `test_day22_logging.py` | **PASS** | 23.07s | Thread-safe concurrency, JSONL audit persistence, CSV telemetry, stats aggregation |
| **5. Streamlit Interactive Web UI** | `test_day23_streamlit.py` | **PASS** | 10.74s | Headless AppTest execution, scenario loading, metric rendering, draft editing, buttons |
| **6. Plotly Operational Dashboard** | `test_day24_dashboard.py` | **PASS** | 0.41s | 100-ticket test batch evaluation, queue volume, priority mix, confidence, HTML output |
| **7. Docker Container Setup** | `verify_docker_setup.py` | **PASS** | 0.03s | Static Dockerfile validation, directives, port mappings, entrypoint multi-mode branching |
| **TOTAL / OVERALL** | **7 Suites** | **100% PASS** | **45.04s** | **Zero Failures Across Entire System** |

---

## 3. Unit Test Suites Architecture

### A. Data Cleaning & Sanitization ([text_cleaning.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/text_cleaning.py))
- **Edge-Case Resilience**: Cleans unicode artifacts, handles unescaped HTML entities (`&amp;`, `&lt;div&gt;`), strips `<p>`, `<b>` tags, and strips non-printable control characters.
- **Deduplication**: Detects if ticket bodies begin with exact redundant subject text and removes duplicate text automatically.
- **Pipeline Sanitization**: Evaluated on batch datasets to ensure non-English tickets (`language != 'en'`) and null bodies are cleanly dropped.

### B. Routing Policy Decision Boundaries ([test_routing.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_routing.py))
- **Mathematical Precision**: Tested exact boundary conditions:
  - Exact $\tau = 0.4000 \land \text{score} = 4.00 \implies$ **Tier 1 (Auto-Draft)**.
  - $\tau = 0.3990 \implies$ **Tier 3 (Human Assist)**.
  - $\text{score} = 3.990 \implies$ **Tier 3 (Human Assist)**.
- **Safety Overrides**: Confirmed that `priority == 'high'` or `queue == 'Service Outages and Maintenance'` strictly forces **Tier 2 (Expedited Review)**, regardless of 99% confidence or high relevance.

### C. FastAPI Microservice Endpoints ([test_api.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_api.py))
- Evaluated via `fastapi.testclient.TestClient` within async `lifespan` context (verifying models are loaded once at startup).
- Validated input validation schemas: empty requests properly return HTTP `422 Unprocessable Entity`.
- Verified `/triage` sub-second responses ($< 20$ ms).
- Verified grounded RAG citations `[KB-XXXXX]` on answerable tickets, refusal guardrails on ungrounded queries, and supervisor review flags on outages.

---

## 4. Production Docker Containerization Architecture

### Dockerfile Design ([Dockerfile](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/Dockerfile))
- **Base Image**: `python:3.12-slim` (minimal attack surface, small disk footprint).
- **Runtime Environment**:
  - `PYTHONUNBUFFERED=1` (instant real-time logging without buffering).
  - `PYTHONDONTWRITEBYTECODE=1` (prevents unnecessary bytecode file generation).
- **System Dependencies**: Installed `build-essential` and `curl` for system healthchecks.
- **Dependencies**: Pinned to exact verified versions in [requirements.txt](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/requirements.txt).
- **Artifact Packing**: Copies pre-computed ChromaDB index (`chroma_db/`), classification models (`models/`), clean Knowledge Base (`data/knowledge_base_clean.csv`), and application source code.
- **Healthcheck**: Configured built-in Docker health probe:
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
      CMD curl -f http://localhost:8000/health || exit 1
  ```
- **Ports Exposed**:
  - Port `8000`: FastAPI REST API & OpenAPI Swagger documentation (`/docs`).
  - Port `8501`: Streamlit Interactive Support Agent Workspace.

---

### Multi-Mode Entrypoint ([entrypoint.sh](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/entrypoint.sh))
The container entrypoint supports flexible microservice orchestration via a single parameter:
```bash
# 1. Run FastAPI REST backend only:
./entrypoint.sh api

# 2. Run Streamlit Agent UI only:
./entrypoint.sh ui

# 3. Run both concurrently with signal trapping (default):
./entrypoint.sh both
```

---

### Docker Compose Orchestration ([docker-compose.yml](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/docker-compose.yml))
Provides one-command enterprise orchestration with volume persistence for the Decision Logger:
```yaml
version: '3.8'

services:
  ticketpilot:
    build:
      context: .
      dockerfile: Dockerfile
    image: ticketpilot:latest
    container_name: ticketpilot_app
    ports:
      - "8000:8000"  # FastAPI REST API & Swagger UI
      - "8501:8501"  # Streamlit Support Workspace
    environment:
      - PORT_API=8000
      - PORT_UI=8501
    volumes:
      - ./logs:/app/logs  # Persistent audit logging for routing decisions
    restart: unless-stopped
    command: ["both"]
```

---

## 5. Deployment & Execution Instructions

### Local Container Build & Run (Docker CLI):
```bash
# 1. Build the production Docker image
docker build -t ticketpilot:latest .

# 2. Run container in background with volume-mapped logs
docker run -d \
  -p 8000:8000 \
  -p 8501:8501 \
  -v $(pwd)/logs:/app/logs \
  --name ticketpilot \
  ticketpilot:latest

# 3. Inspect running container health
docker ps
docker logs -f ticketpilot
```

### Or Using Docker Compose:
```bash
docker compose up -d --build
```

---

## 6. Definition of Done Check (Sprint Plan Day 25)

- [x] Comprehensive unit tests for data cleaning & normalization ([test_cleaning.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_cleaning.py)).
- [x] Unit tests for 3-tier routing decision boundaries ([test_routing.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_routing.py)).
- [x] Unit tests for FastAPI REST API endpoints ([test_api.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_api.py)).
- [x] Master test runner executing all 7 test suites ([run_all_tests.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/run_all_tests.py)) with 100% pass rate.
- [x] Production [Dockerfile](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/Dockerfile), [docker-compose.yml](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/docker-compose.yml), [entrypoint.sh](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/entrypoint.sh), and [.dockerignore](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/.dockerignore).
- [x] Static container validation verified in [verify_docker_setup.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/verify_docker_setup.py).
