
**Important:**  

- The `Then:` and the numbered list should be **outside** the Python code block.  
- So you must have:

  1. ``` ```python ``` to open the code block for `state = {...}`  
  2. Then the `state = { ... }` lines  
  3. Then a closing ``` ``` ``` (three backticks)  
  4. After that, on a new line: `Then:` and the 1–4 list

If you paste the whole README as I gave (inside `README.md`), the “Then:” section lives under:

> `## 2. End-to-End Flow` → `### 2.1 Sequence Overview`

You don’t need to put it in any other file. Everything inside that big ```markdown``` block should go directly into `README.md` at the root of your project.

---

## 2. Add an architecture graph (diagram) to the README

Here’s a **Mermaid diagram** of the overall architecture. GitHub and many renderers support Mermaid directly.  

👉 **Where to place it:**  
Put this right after **`### 1.1 Components`** as a new subsection, e.g. `### 1.2 Architecture Diagram`.

```markdown
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
        PARSE[Parser Agent<br/>node_parse<br/>(tools.extract_*)]
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
