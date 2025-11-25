# TrendingNews

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
