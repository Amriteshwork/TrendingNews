# 📄 Resume Assessment Agent

An AI-powered agent that ingests resumes in multiple formats (PDF, DOCX, images, plain text), compares them with a given Job Description (JD), and produces:

- **Objective scores** (skills, experience, seniority, overall)
- **A detailed, RAG-grounded assessment**
- **Guardrailed, PII-masked output**
- **Persisted assessments** in a database for later analysis

This project is designed to demonstrate:

- The ability to **design, reason, and orchestrate AI agents**
- Use of **RAG**, **memory**, **tool calling**, and multi-step / multi-agent style collaboration
- Integration with **real-world systems**: APIs, databases, file systems, UI
- Application of **guardrails and safety controls**
- Use of **recent and advanced agent tech** (LangGraph, FAISS, structured LLM outputs)

---

## 1. High-Level Architecture

### 1.1 Components

- **LLM (OpenAI GPT-4o / GPT-4o-mini)**  
  - Structured extraction of resume & JD  
  - Estimation of relevant years of experience & seniority fit  
  - Generation of the narrative assessment (grounded with RAG)

- **LangGraph Orchestrator (`graph.py`)**  
  Multi-step agent pipeline:
  1. Parse resume & JD into structured objects
  2. Compute objective scores
  3. Generate assessment with RAG
  4. Apply guardrails & persist to DB

- **RAG Layer (`rag.py` + `data/` folder)**  
  - Loads `.md` guideline documents  
  - Embeds them via OpenAI embeddings  
  - Uses FAISS vector search to retrieve relevant chunks for the assessment

- **Database (SQLite + SQLAlchemy, `db.py`)**  
  - `assessments` table storing:
    - Candidate name  
    - JD title  
    - Scores (overall, skills, experience, seniority)  
    - Raw assessment text  
    - Timestamp

- **Backend API (FastAPI, `main.py`)**  
  - `POST /assess_resume`: core endpoint for programmatic access

- **Web UI (Gradio, `ui.py`)**  
  - Interactive interface for uploading a resume and JD
  - Returns human-readable assessment + JSON scores

- **File System Integration (`tools.py` + `data/`)**  
  - Multi-format resume parsing: PDF, DOCX, images (OCR), plain text  
  - Documentation storage for RAG

### 1.2 Architecture Diagram

```mermaid
flowchart LR
    %% Users
    U_API[User (API client)]
    U_UI[User (Web UI)]

    %% Backends
    API[FastAPI backend\n(main.py)]
    GRADIO[Gradio UI\n(ui.py)]
    GRAPH_APP[LangGraph app\n(graph.py)]

    %% Agents / nodes
    PARSE[Parser agent\n(node_parse)]
    SCORE[Scoring agent\n(node_score)]
    ASSESS[Assessment agent\n(node_assess)]
    SAFETY[Safety & persistence\n(node_guardrail_and_save)]

    %% RAG + DB + config
    RAG[RAG retriever\n(rag.py + data/*.md)]
    DB[(SQLite DB\nassessments.db)]
    CONFIG[Config & secrets\n(config.py + .env)]

    %% User → backend
    U_API -->|Resume + JD| API
    U_UI -->|Upload resume + JD| GRADIO

    %% Backend → graph
    GRADIO -->|Python call| GRAPH_APP
    API -->|invoke()| GRAPH_APP

    %% LangGraph pipeline
    GRAPH_APP --> PARSE --> SCORE --> ASSESS --> SAFETY --> DB

    %% RAG interactions
    ASSESS -->|query guidelines| RAG
    RAG -->|guideline chunks| ASSESS

    %% Config wiring
    CONFIG --> API
    CONFIG --> GRADIO
    CONFIG --> GRAPH_APP
    CONFIG --> RAG
