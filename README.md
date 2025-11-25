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
    U_API[User / Client<br/>HTTP] -->|Resume file + JD text| API[FastAPI Backend<br/>(main.py)]
    U_UI[User / Recruiter<br/>Browser] -->|Upload resume + JD| GRADIO[Gradio UI<br/>(ui.py)]

    %% Backend connections
    GRADIO -->|Python call| GRAPH_APP[LangGraph App<br/>(graph.py)]
    API -->|invoke()| GRAPH_APP

    %% LangGraph pipeline
    subgraph LANGGRAPH[LangGraph Orchestrated Agents]
        direction LR
        PARSE[Parser Agent<br/>node_parse<br/>(extract_* tools)]
        SCORE[Scoring Agent<br/>node_score<br/>(compute_scores)]
        ASSESS[Assessment Agent<br/>node_assess<br/>(generate_assessment + RAG)]
        SAFETY[Safety & Persistence Agent<br/>node_guardrail_and_save<br/>(mask_pii + DB)]

        PARSE --> SCORE --> ASSESS --> SAFETY
    end

    GRAPH_APP --> PARSE

    %% RAG
    ASSESS -->|query| RAG[RAG Retriever<br/>(rag.py + data/*.md)]
    RAG -->|guidelines chunks| ASSESS

    %% Database
    SAFETY -->|store assessment<br/>scores + text| DB[(SQLite DB<br/>assessments.db<br/>db.py)]

    %% Config
    CONFIG[Config & Secrets<br/>(config.py + .env)] --> API
    CONFIG --> GRADIO
    CONFIG --> RAG
    CONFIG --> LANGGRAPH
```

---

## 2. End-to-End Flow

### 2.1 Sequence Overview

1. **User input**
   - User provides:
     - A resume file (PDF / DOCX / image / TXT)
     - A Job Description text
   - Input comes via:
     - **FastAPI** endpoint `POST /assess_resume`, or
     - **Gradio UI** (web page) in `ui.py`

2. **Resume parsing**
   - The file is read into bytes.
   - `parse_resume_text(file_bytes, filename)` (in `tools.py`) detects the extension:
     - `.pdf` → `parse_pdf()` using `pypdf.PdfReader`
     - `.docx` → `parse_docx()` using `python-docx`
     - `.png/.jpg/.jpeg` → `parse_image()` using `PIL` + `pytesseract` OCR
     - Anything else → treated as UTF-8 plain text

3. **Agent pipeline (LangGraph)**

   Initial state:

   ```python
   state = {
       "resume_text": <parsed resume>,
       "jd_text": <JD text>,
   }
   ```

   Then:

   1. **`node_parse`**  
      - Calls:
        - `extract_resume_structured(resume_text)`
        - `extract_jd_structured(jd_text)`
      - Populates:
        - `state["resume_structured"]`
        - `state["jd_structured"]`

   2. **`node_score`**  
      - Calls:
        - `compute_scores(state["resume_structured"], state["jd_structured"])`
      - Populates:
        - `state["scores"]`  
          (`skills_score`, `experience_score`, `seniority_score`, `overall_score`)

   3. **`node_assess`**  
      - Calls:
        - `generate_assessment(resume_structured, jd_structured, scores)`
      - Uses RAG inside `generate_assessment` to load best-practice guidelines.
      - Populates:
        - `state["assessment_text"]`

   4. **`node_guardrail_and_save`**  
      - Calls:
        - `mask_pii(state["assessment_text"])` to remove emails/phones
        - `save_assessment_to_db(...)` to persist entry in SQLite
      - Populates:
        - `state["cleaned_assessment_text"]`

4. **Response back to user**

   - **FastAPI**:
     - Wraps `scores` + `cleaned_assessment_text` in `AssessmentResponse`.
     - Appends a safety disclaimer to the assessment text.
   - **Gradio**:
     - Displays assessment (Markdown) + JSON scores.
     - Also appends a disclaimer.

---

## 3. Module-by-Module Breakdown

### 3.1 `config.py` – Configuration

- Uses `dotenv` and `pydantic_settings` to load environment variables.
- `Settings` includes:
  - `openai_api_key` (from `OPENAI_API_KEY`)
  - `db_url` (default: `sqlite:///./assessments.db`)
  - `embedding_model` (default: `multi-qa-mpnet-base-dot-v1`)
- Exposes a singleton `settings` object used by other modules.

Typical `.env` file:

```env
OPENAI_API_KEY=sk-your-key
DB_URL=sqlite:///./assessments.db
EMBEDDING_MODEL=multi-qa-mpnet-base-dot-v1
```

---

### 3.2 `db.py` – Database Layer

- Initializes SQLAlchemy engine using `settings.db_url`.
- Uses `connect_args={"check_same_thread": False}` for SQLite.
- Defines:

```python
class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, index=True)
    candidate_name = Column(String, index=True)
    jd_title = Column(String, index=True)
    overall_score = Column(Float)
    skills_score = Column(Float)
    experience_score = Column(Float)
    seniority_score = Column(Float)
    raw_assessment = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
```

- `init_db()` creates all tables.

This is the **long-term memory** for each assessment.

---

### 3.3 `models.py` – Pydantic Models & Agent State

- **`AssessmentRequest`**
  - Holds `jd_text` (JD is sent as text; resume is file-upload).
- **`AssessmentResponse`**
  - `overall_score`
  - `skills_score`
  - `experience_score`
  - `seniority_score`
  - `assessment_text` (final cleaned assessment plus disclaimer)
- **`AgentState`** (TypedDict for LangGraph):

```python
class AgentState(TypedDict, total=False):
    resume_text: str
    jd_text: str
    resume_structured: Dict[str, Any]
    jd_structured: Dict[str, Any]
    scores: Dict[str, float]
    guidelines: str
    assessment_text: str
    cleaned_assessment_text: str
    errors: str
```

This is the **shared memory** that each node/agent reads and writes.

---

### 3.4 `rag.py` – RAG Retriever

Implements a lightweight RAG mechanism over local `.md` guideline files.

- Uses `OpenAI` embeddings (via `settings.embedding_model`).
- Uses FAISS (`IndexFlatL2`) as the vector store.

Key parts:

- `_embed(texts: List[str]) -> np.ndarray`
  - If no API key is set or an error occurs:
    - Returns zero embeddings but does not crash the app.
  - Otherwise:
    - Calls `client.embeddings.create(model=settings.embedding_model, input=texts)`  
      and converts embeddings to a NumPy array.

- `class RAGRetriever`
  - `__init__(self, data_dir="data")`  
    - Stores data directory, initializes `index` and `chunks`.
  - `build_index(self)`:
    - Ensures `data/` exists.
    - Reads `*.md` files.
    - Naively chunks content into 800-char pieces.
    - Embeds all chunks.
    - Builds a FAISS `IndexFlatL2` with those vectors.
  - `retrieve(self, query: str, k: int = 4) -> str`:
    - Builds index if needed.
    - Returns empty string if no chunks.
    - Embeds query, does FAISS search.
    - Returns top-`k` chunks joined with `\n\n`.

- `rag_retriever = RAGRetriever()` is the global instance.

Used in `generate_assessment` to bring **“resume evaluation best practices”** into the prompt.

---

### 3.5 `tools.py` – Tools & Agent Utilities

This is the toolbox that the LangGraph agents rely on.

#### 3.5.1 Parsing tools

- `parse_pdf(file_bytes: bytes) -> str`
  - Uses `PdfReader` and concatenates `page.extract_text()`.

- `parse_docx(file_bytes: bytes) -> str`
  - Uses `DocxDocument` to join paragraph texts.

- `parse_image(file_bytes: bytes) -> str`
  - Uses `PIL.Image.open` + `pytesseract.image_to_string`.
  - If `pytesseract` is missing, returns `[OCR not available: ...]`.
  - If OCR fails, returns `[OCR Error: ...]`.

- `parse_resume_text(file_bytes: bytes, filename: str) -> str`
  - Routing:
    - `.pdf` → `parse_pdf`
    - `.docx` → `parse_docx`
    - `.png/.jpg/.jpeg` → `parse_image`
    - Else → decode as text via `file_bytes.decode("utf-8", errors="ignore")`

This gives the system **multi-format resume ingestion**.

#### 3.5.2 LLM JSON helper

- `llm_json_system_prompt() -> str`
  - Forces the assistant to “Always respond with valid JSON only”.

- `call_llm_json(prompt: str) -> Dict[str, Any]`
  - Uses `OpenAI` Chat Completions:
    - `model="gpt-4o-mini"`
    - `messages = [system, user]`
    - `response_format={"type": "json_object"}`
  - Parses the JSON string into a Python dict.
  - If no API key or error:
    - Returns a dict with an `"error"` or empty `{}`.

This is effectively a **structured-output / JSON tool-calling wrapper**.

#### 3.5.3 Structured extraction

- `extract_resume_structured(resume_text: str) -> Dict[str, Any]`
  - Prompts the LLM to return JSON with:
    - `name: string`
    - `email: string`
    - `skills: list[str]`
    - `experience: list[{title, company, years, description}]`
    - `education: list[{degree, institution, year}]`
  - Truncates resume text (e.g., to 4000 chars) to stay within context.

- `extract_jd_structured(jd_text: str) -> Dict[str, Any]`
  - Prompts the LLM to return JSON with:
    - `title: string`
    - `required_skills: list[str]`
    - `preferred_skills: list[str]`
    - `seniority_level: "junior" | "mid" | "senior"`
    - `summary: string`

These functions transform unstructured text into **machine-friendly structured objects**.

#### 3.5.4 Objective scoring

- `compute_scores(resume: Dict[str, Any], jd: Dict[str, Any]) -> Dict[str, float]`:

  1. **Skills score**
     - Build:
       - `resume_skills` from `resume["skills"]`
       - `jd_skills` from `jd["required_skills"]`
       - Both lowercased and stripped.
     - `intersection = resume_skills & jd_skills`
     - `skills_score = len(intersection) / len(jd_skills)` (or `0.0` if no JD skills)

  2. **Experience & seniority fit (via LLM)**
     - Prompt the LLM with:
       - Resume experience entries
       - Full JD
     - Ask for JSON with:
       - `relevant_years: float`
       - `seniority_fit: float` in `[0.0, 1.0]`
     - Parse result, with defaults if missing.

  3. **Experience score**
     - `experience_score = min(relevant_years / 5.0, 1.0)`
       - Interprets 5 years as full score.

  4. **Overall score**
     - `overall = 0.5 * skills_score + 0.3 * experience_score + 0.2 * seniority_fit`

  5. **Return**
     - All scores rounded to 3 decimals:
       - `skills_score`, `experience_score`, `seniority_score`, `overall_score`

This introduces an **explicit formula** to make the assessment more objective and auditable.

#### 3.5.5 Assessment with RAG

- `generate_assessment(resume: Dict[str, Any], jd: Dict[str, Any], scores: Dict[str, float]) -> str`:

  - Fetches guidelines:
    - `guidelines = rag_retriever.retrieve("resume evaluation best practices")`
  - Builds a `user_prompt` including:
    - Structured resume
    - Structured JD
    - Objective scores
    - Retrieved guidelines
  - Calls `client.chat.completions.create` with:
    - `model="gpt-4o"`
    - System: “You are a fair, objective resume reviewer.”
    - User: `user_prompt`
  - Returns the LLM’s message content.

  The prompt instructs the LLM to structure the assessment as:

  1. Overall fit summary  
  2. Skills analysis  
  3. Experience analysis  
  4. Suggestions for improvement  

This ensures **clear rationales** for the scores and integrates **RAG-sourced best practices**.

#### 3.5.6 Guardrails: PII masking

- Regex patterns:
  - `EMAIL_RE` – basic email pattern
  - `PHONE_RE` – simple international-ish phone number pattern

- `mask_pii(text: str) -> str`:
  - Replaces all matched emails with `[REDACTED_EMAIL]`
  - Replaces phone-like strings with `[REDACTED_PHONE]`

This reduces risk of exposing sensitive candidate contact info in logs or UI.

#### 3.5.7 DB helpers

- `save_assessment_to_db(resume, jd, scores, assessment_text)`:
  - Opens a SQLAlchemy session.
  - Extracts:
    - `candidate_name` from `resume["name"]` or `"Unknown"`
    - `jd_title` from `jd["title"]` or `"Unknown"`
  - Creates an `Assessment` record:
    - `overall_score`, `skills_score`, `experience_score`, `seniority_score`
    - `raw_assessment` (unmasked assessment text)
  - Commits the transaction.
  - Rolls back and logs on error, always closes session.

---

### 3.6 `graph.py` – Agent Orchestration (LangGraph)

Defines the multi-step agent pipeline.

- **`node_parse(state: AgentState) -> AgentState`**
  - Uses `extract_resume_structured` and `extract_jd_structured`.
  - Stores results in `state["resume_structured"]`, `state["jd_structured"]`.

- **`node_score(state: AgentState) -> AgentState`**
  - Calls `compute_scores` on the structured data.
  - Stores `state["scores"]`.

- **`node_assess(state: AgentState) -> AgentState`**
  - Calls `generate_assessment`.
  - Stores `state["assessment_text"]`.

- **`node_guardrail_and_save(state: AgentState) -> AgentState`**
  - Calls `mask_pii` on `assessment_text`.
  - Stores `state["cleaned_assessment_text"]`.
  - Calls `save_assessment_to_db` to persist.
  - Returns state unchanged otherwise.

- **`build_graph()`**
  - Creates `workflow = StateGraph(AgentState)`.
  - Adds nodes: `"parse"`, `"score"`, `"assess"`, `"guardrail_and_save"`.
  - Edges:
    - `parse → score → assess → guardrail_and_save → END`
  - Entry point: `"parse"`.
  - Compiles and returns `app` (the LangGraph app).

Conceptually, each node acts as a **specialized agent** collaborating through shared state.

---

### 3.7 `main.py` – FastAPI Service

- Creates FastAPI app:

```python
app = FastAPI(title="Resume Assessment Agent")
```

- Adds permissive CORS (for demo / local usage).
- Builds graph and initializes DB:

```python
graph_app = build_graph()
init_db()
```

#### Endpoint: `POST /assess_resume`

```python
@app.post("/assess_resume", response_model=AssessmentResponse)
async def assess_resume(
    resume_file: UploadFile = File(...),
    jd_text: str = Form(...)
):
    ...
```

Flow:

1. Read bytes: `file_bytes = await resume_file.read()`.
2. Validate non-empty.
3. Parse resume text via `parse_resume_text(file_bytes, resume_file.filename)`.
4. Build initial `state`.
5. Invoke graph: `final_state = graph_app.invoke(state)`.
6. Extract:
   - `scores = final_state.get("scores", {})`
   - `cleaned_assessment = final_state.get("cleaned_assessment_text", "")`
7. Append disclaimer to `cleaned_assessment`.
8. Return `AssessmentResponse`:

   - `overall_score`, `skills_score`, `experience_score`, `seniority_score`
   - `assessment_text`

> **Important:** This endpoint is **POST-only**.  
> A simple `GET` on `/assess_resume` will return `{"detail": "Method Not Allowed"}`.  
> Use `POST` with multipart form data.

#### Endpoint: `GET /`

- Simple health check: returns `{"message": "Resume assessment agent is running."}`

---

### 3.8 `ui.py` – Gradio Web UI

- Builds graph and initializes DB:

```python
graph_app = build_graph()
init_db()
```

- **`assess_with_ui(resume_file, jd_text: str)`**

  - `resume_file` is a file path string (Gradio, `type="filepath"`).
  - Validates presence of file and JD text.
  - Reads file bytes, uses `parse_resume_text` to obtain `resume_text`.
  - Builds `state` and calls `graph_app.invoke(state)`.
  - Extracts:
    - `assessment = final_state.get("cleaned_assessment_text", "")`
    - `scores = final_state.get("scores", {})`
  - Adds Markdown disclaimer and returns:
    - Assessment markdown
    - Scores JSON

- **`create_demo()`**

  - Builds a Gradio `Blocks` UI:
    - Title & description
    - File upload (`.pdf`, `.docx`, `.txt`, `.png`, `.jpg`, `.jpeg`)
    - JD textarea
    - “Assess Resume” button
    - Outputs:
      - `Markdown` assessment
      - `JSON` scores

- If run directly (`python ui.py`):
  - Launches the Gradio demo locally.

---

## 4. How This Meets the Assignment Requirements

### 4.1 Design, reason, and orchestrate AI agents

- The system uses **LangGraph** to orchestrate a multi-step agent flow:
  - Parsing agents → Scoring agent → Assessment agent → Safety & Persistence agent.
- Each step has a clear responsibility, and data is passed via a shared `AgentState`.
- This makes the reasoning explainable and debuggable step-by-step.

### 4.2 Use of RAG, memory, tool calling, multi-agent collaboration

- **RAG**:
  - `rag.py` + `data/*.md` + FAISS index.
  - Used in `generate_assessment` to ground feedback in resume evaluation best practices.

- **Memory**:
  - **Long-term**: `assessments` table in SQLite stores scores and text of all past assessments.
  - **Knowledge**: RAG index preserves guidelines across runs.

- **Tool calling**:
  - Python-level tools: parsing, extraction, scoring, RAG, PII masking, DB persistence.
  - LLM-level structured outputs via `response_format={"type": "json_object"}`.

- **Multi-agent collaboration**:
  - Though implemented as nodes, conceptually they function as:
    - Parser Agent
    - Scoring Agent
    - Assessment Agent
    - Safety & Persistence Agent  
  - Each agent consumes and updates shared state, forming a collaborative pipeline.

### 4.3 Integration with real-world systems

- **APIs**:
  - FastAPI backend exposes `/assess_resume`, ready to be called by other services or frontends.

- **Databases**:
  - SQLite + SQLAlchemy integration for persistence and analytics.

- **File systems**:
  - Multi-format resume ingestion.
  - Local docs for RAG.

- **UI**:
  - Gradio web interface for recruiters / demo usage.

### 4.4 Guardrails and safety controls

- **PII masking** (emails and phone numbers).
- **Disclaimers** in both API and UI.
- **Graceful failure**:
  - Missing API key → dummy embeddings or explicit error messages.
  - Parsing errors → readable error strings instead of crashes.
- Separation of concerns:
  - Dedicated node (`node_guardrail_and_save`) for masking and storing.

### 4.5 Recent and advanced agent tech

- **LangGraph** for graph-based agent orchestration.
- **OpenAI GPT-4o/GPT-4o-mini** with structured JSON outputs.
- **FAISS** for vector search in RAG.
- **Multi-modal ingestion** (PDF, DOCX, images).

---

## 5. Running the Project

### 5.1 Prerequisites

- Python 3.9+
- (Optional, for image OCR) System-level `tesseract-ocr` installed.

### 5.2 Install dependencies

Example (adjust as needed):

```bash
pip install fastapi uvicorn gradio sqlalchemy pydantic pydantic-settings python-dotenv \
            openai langgraph faiss-cpu numpy pypdf python-docx pillow pytesseract \
            typing_extensions
```

### 5.3 Environment variables

Create a `.env` file at project root:

```env
OPENAI_API_KEY=sk-your-key
DB_URL=sqlite:///./assessments.db
EMBEDDING_MODEL=multi-qa-mpnet-base-dot-v1
```

### 5.4 Run the API

```bash
uvicorn main:app --reload
```

- Health check:  
  `GET http://127.0.0.1:8000/` → `{"message": "Resume assessment agent is running."}`

- Interactive docs (Swagger UI):  
  `http://127.0.0.1:8000/docs`

Example `curl` for `/assess_resume`:

```bash
curl -X POST "http://127.0.0.1:8000/assess_resume" \
  -F "resume_file=@/path/to/resume.pdf" \
  -F "jd_text=Senior Backend Engineer with Python, FastAPI, PostgreSQL"
```

> Remember: This endpoint is **POST-only**, not GET.

### 5.5 Run the Gradio UI

```bash
python ui.py
```

Open the printed local URL in your browser (e.g. `http://127.0.0.1:7860`).

---

## 6. Future Improvements & Suggestions

These are optional but good to mention in your submission / next iterations:

1. **Explicit multi-agent roles**
   - Wrap each node with clearer “agent” semantics and prompts:
     - `ParserAgent`, `ScoringAgent`, `AssessmentAgent`, `SafetyAgent`.
   - Add a **Critic Agent** that:
     - Reviews the assessment for bias / inconsistency.
     - Forces a re-write if issues are detected.

2. **Use historical memory more explicitly**
   - Add a node to:
     - Query previous assessments for the same candidate or JD.
     - Provide relative comments (“this candidate improved from 0.62 to 0.75 skills match since last submission”).

3. **Stronger guardrails**
   - Add checks to avoid:
     - References to protected attributes (age, gender, nationality, etc.).
     - Overly subjective or discriminatory language.
   - Provide structured reason codes for low scores (e.g. “missing required skill: Docker”).

4. **Test suite**
   - Introduce `pytest` tests for:
     - `compute_scores()` logic.
     - `mask_pii()` correctness.
     - `parse_resume_text()` routing.
     - A small integration test for `/assess_resume`.

5. **Configurable scoring weights**
   - Move the weights (0.5, 0.3, 0.2) into `config.py` so different organizations can tune emphasis on skills vs. experience vs. seniority.

6. **Enhanced RAG**
   - Extend `data/` to include:
     - Role-specific evaluation rubrics (e.g., Backend Engineer, Data Scientist).
     - Company-specific values or competencies.
   - Adapt RAG queries (e.g., `f"{jd_title} resume evaluation best practices"`) for more targeted guidelines.

---

## 7. Limitations

- Quality of assessment is dependent on:
  - The quality and completeness of resume & JD.
  - The underlying LLM and the RAG documents used.
- Regex-based PII masking may occasionally over-mask or under-mask.
- Experience estimation (`relevant_years`, `seniority_fit`) is still LLM-driven and may need calibration with real hiring data.

---

## 8. Summary

This project implements a modern **agentic resume assessment system** with:

- **Multi-step orchestration** via LangGraph
- **RAG-grounded** reasoning
- **Objective scoring** with clear formulas
- **Guardrails** (PII masking + disclaimers)
- **Integration** across API, DB, file system, and UI

It’s a strong demonstration of:
- Agent design & orchestration  
- RAG + memory + tool usage  
- Real-world integration  
- Safety & guardrails  
- Advanced LLM-based agent techniques
