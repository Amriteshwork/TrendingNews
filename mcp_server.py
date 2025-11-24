from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, json, time
import requests
import numpy as np
from typing import List, Optional
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# Optional OpenAI import (if OPENAI_API_KEY provided)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    import openai
    openai.api_key = OPENAI_API_KEY

# Optional local embedding fallback
USE_LOCAL_EMBED = os.getenv("EMBEDDING_MODEL", "openai") == "local"
if USE_LOCAL_EMBED:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
if not NEWSDATA_API_KEY:
    print("Warning: NEWSDATA_API_KEY not set — fetch_news will fail until you set it.")

TOPIC_STORE_PATH = Path("topic_registry.json")
BOOTSTRAP_LIMIT = 5               # number of initial topics to allow before strict matching
BOOTSTRAP_THRESHOLD = 0.70       # if similarity < this during bootstrap -> create new topic
ASSIGN_THRESHOLD = 0.80          # normal assignment threshold
DRIFT_SPLIT_THRESHOLD = 0.60     # optional (not implemented auto-split here)

app = FastAPI(title="MCP Trending-Topic Registry")

# -------------------------
# Data models / utilities
# -------------------------
class Article(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    pubDate: Optional[str] = None
    source_name	: Optional[str] = None
    source_url	: Optional[str] = None

    # NEW FIELDS FROM NEWSDATA.IO
    ai_org: Optional[List[str]] = None
    ai_region: Optional[List[str]] = None
    sentiment: Optional[str] = None        # "positive" | "negative" | "neutral"
    sentiment_stat: Optional[float] = None # confidence score


class AssignResult(BaseModel):
    topic_id: str
    topic_name: str
    similarity: float
    created_new_topic: bool = False

def now_ts():
    return datetime.now(timezone.utc).isoformat()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return -1.0
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if den == 0 else (num / den)

# -------------------------
# Topic registry persistence
# -------------------------
def load_topics():
    if not TOPIC_STORE_PATH.exists():
        return []
    try:
        return json.loads(TOPIC_STORE_PATH.read_text())
    except Exception:
        return []

def save_topics(topics):
    TOPIC_STORE_PATH.write_text(json.dumps(topics, indent=2))

# Each topic stored as:
# {
#   "topic_id": "T001",
#   "name": "OpenAI AI Model Releases",
#   "centroid": [ ... ],
#   "article_count": 12,
#   "top_entities": ["OpenAI", "GPT-5"],
#   "last_updated": "2025-11-24T11:50:00Z",
#   "created_at": "..."
# }

topics = load_topics()

def next_topic_id():
    return f"T{len(topics)+1:04d}"

def update_topic_centroid_incremental(topic, new_embedding: np.ndarray):
    # centroid = (centroid * count + new_emb) / (count + 1)
    c = np.array(topic["centroid"], dtype=float)
    count = topic["article_count"]
    new_c = (c * count + new_embedding) / (count + 1)
    topic["centroid"] = new_c.tolist()
    topic["article_count"] = count + 1
    topic["last_updated"] = now_ts()
    return topic

# -------------------------
# Embedding helpers
# -------------------------
def embed_text_openai(text: str) -> np.ndarray:
    # uses text-embedding-3-small or text-embedding-3-large depending on your preference
    # truncated for simplicity — call OpenAI with your desired model and handle rate limits in production
    model = "text-embedding-3-small"
    resp = openai.Embeddings.create(model=model, input=text)
    vec = np.array(resp["data"][0]["embedding"], dtype=float)
    return vec

def embed_text_local(text: str) -> np.ndarray:
    vec = LOCAL_EMBED_MODEL.encode(text)
    return np.array(vec, dtype=float)

def get_embedding(text: str) -> np.ndarray:
    text = (text or "").strip()
    if len(text) == 0:
        return np.zeros(384, dtype=float)  # fallback
    if OPENAI_API_KEY and not USE_LOCAL_EMBED:
        return embed_text_openai(text)
    else:
        return embed_text_local(text)

# -------------------------
# Topic assignment logic
# -------------------------
def find_best_topic(embedding: np.ndarray):
    best = None
    best_score = -1.0
    for t in topics:
        centroid = np.array(t["centroid"], dtype=float)
        s = cosine_similarity(embedding, centroid)
        if s > best_score:
            best_score = s
            best = t
    return best, float(best_score)

def create_topic_from_article(article: Article, embedding: np.ndarray, name: Optional[str]=None):
    tid = next_topic_id()
    tname = name or generate_topic_name_stub(article)

    topic = {
        "topic_id": tid,
        "name": tname,
        "centroid": embedding.tolist(),
        "article_count": 1,

        # NEW FIELDS
        "organizations": article.ai_org or [],
        "regions": article.ai_region or [],
        "sentiment": {
            "negative": 0,
            "neutral": 0,
            "positive": 0
        },

        "created_at": now_ts(),
        "last_updated": now_ts(),
        "example_articles": [
            {
                "id": article.id,
                "title": article.title,
                "summary": article.summary,
                "url": article.source_url,
                "pubDate_at": article.pubDate,
                "ai_org": article.ai_org,
                "ai_region": article.ai_region,
                "sentiment": article.sentiment,
                "sentiment_stat": article.sentiment_stat
            }
        ]
    }

    # increment sentiment counter
    if article.sentiment in topic["sentiment"]:
        topic["sentiment"][article.sentiment] += 1

    topics.append(topic)
    save_topics(topics)
    return topic

def generate_topic_name_stub(article: Article):
    # Simple heuristic stub: prefer explicit entities in title
    title = (article.title or "").strip()
    if title:
        return title[:60]  # fallback short name
    return f"Topic {len(topics)+1}"

def assign_or_create_topic_logic(article: Article, embedding: np.ndarray,
                                 bootstrap_limit:int=BOOTSTRAP_LIMIT,
                                 bootstrap_threshold:float=BOOTSTRAP_THRESHOLD,
                                 assign_threshold:float=ASSIGN_THRESHOLD) -> AssignResult:
    # If no topics exist -> always create
    if len(topics) == 0:
        new_topic = create_topic_from_article(article, embedding)
        return AssignResult(topic_id=new_topic["topic_id"], topic_name=new_topic["name"],
                            similarity=1.0, created_new_topic=True)

    # Find best existing topic
    best_topic, best_score = find_best_topic(embedding)

    # Bootstrap phase: be permissive to create initial topics
    if len(topics) < bootstrap_limit:
        if best_score < bootstrap_threshold:
            new_topic = create_topic_from_article(article, embedding)
            return AssignResult(topic_id=new_topic["topic_id"], topic_name=new_topic["name"],
                                similarity=float(best_score), created_new_topic=True)
        else:
            # assign and update centroid
            update_topic_centroid_incremental(best_topic, embedding)
            # append example article
            best_topic.setdefault("example_articles", []).append({
                "id": article.id,
                "title": article.title,
                "summary": article.summary,
                "url": article.source_url,
                "published_at": article.pubDate,
                "ai_org": article.ai_org,
                "ai_region": article.ai_region,
                "sentiment": article.sentiment,
                "sentiment_stat": article.sentiment_stat
            })
            save_topics(topics)

        # Update topic entities and sentiment
        if article.ai_org:
            # extend only new orgs
            for org in article.ai_org:
                if org not in best_topic["organizations"]:
                    best_topic["organizations"].append(org)

        if article.ai_region:
            for region in article.ai_region:
                if region not in best_topic["regions"]:
                    best_topic["regions"].append(region)

        # Update sentiment counters
        if article.sentiment in best_topic["sentiment"]:
            best_topic["sentiment"][article.sentiment] += 1
            return AssignResult(topic_id=best_topic["topic_id"], topic_name=best_topic["name"],
                                similarity=float(best_score), created_new_topic=False)

    # Normal phase
    if best_score >= assign_threshold:
        update_topic_centroid_incremental(best_topic, embedding)
        best_topic.setdefault("example_articles", []).append({
            "id": article.id, "title": article.title, "summary": article.summary, "url": article.url, "published_at": article.pubDate
        })
        save_topics(topics)
        return AssignResult(topic_id=best_topic["topic_id"], topic_name=best_topic["name"],
                            similarity=float(best_score), created_new_topic=False)
    else:
        new_topic = create_topic_from_article(article, embedding)
        return AssignResult(topic_id=new_topic["topic_id"], topic_name=new_topic["name"],
                            similarity=float(best_score), created_new_topic=True)

# -------------------------
# NewsData.io fetch helper
# -------------------------
def newsdata_fetch(query: Optional[str]=None, language:Optional[str]=None, country:Optional[str]=None, page:int=1):
    if not NEWSDATA_API_KEY:
        raise RuntimeError("NEWSDATA_API_KEY not set")
    base = "https://newsdata.io/api/1/news"
    params = {"apikey": NEWSDATA_API_KEY, "page": page}
    if query: params["q"] = query
    if language: params["language"] = language
    if country: params["country"] = country
    resp = requests.get(base, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

# -------------------------
# FastAPI endpoints
# -------------------------
@app.post("/fetch_news")
def fetch_news(payload: dict):
    """
    POST /fetch_news
    Body JSON: { "query": "...", "category": "...", "language": "en", "page": 1 }
    """
    try:
        data = newsdata_fetch(query=payload.get("query"), category=payload.get("category"),
                              language=payload.get("language"), country=payload.get("country"),
                              page=payload.get("page", 1))
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_article")
def process_article(article: Article):
    """
    Light processing. If article.summary exists, keep it, otherwise pick description/content.
    This endpoint is intentionally small - integrate your LangChain pipeline here later.
    """
    art = article.dict()
    # choose summary
    summary = art.get("summary") or art.get("title") or ""
    # truncate if too long
    if len(summary) > 2000:
        summary = summary[:2000]
    art["summary"] = summary
    art["processed_at"] = now_ts()
    return art

@app.post("/assign_or_create")
def assign_or_create(article: Article):
    """
    Assign article to topic (or create new).
    Body: Article model
    """
    art = article
    text_for_embedding = (art.summary or art.title or "")
    emb = get_embedding(text_for_embedding)
    result = assign_or_create_topic_logic(art, emb)
    return result.dict()

@app.post("/ingest_batch")
def ingest_batch(payload: dict):
    """
    Convenience: fetch with newsdata params, process each article, compute embedding, assign topics
    Body: same params as /fetch_news
    Returns list of assignment results
    """
    res = newsdata_fetch(
        query=payload.get("query"),
        language=payload.get("language"),
        country=payload.get("country"),
        page=payload.get("page", 1)
    )

    items = res.get("results", []) if isinstance(res, dict) else []
    assignments = []
    for it in items:
        art = Article(
            id=it.get("guid") or it.get("source_url") or str(time.time()),
            title=it.get("title"),
            description=it.get("description"),
            content=it.get("content"),
            summary=it.get("summary") or it.get("description"),
            published_at=it.get("pubDate"),
            source=it.get("source_name"),
            url=it.get("source_url"),

            # NEW FIELDS
            ai_org=it.get("ai_org"),
            ai_region=it.get("ai_region"),
            sentiment=it.get("sentiment"),
            sentiment_stat=it.get("sentiment_stat")
        )
        processed = process_article(art)  # uses the small processor above
        text_for_embedding = processed.get("summary") or processed.get("description") or processed.get("title") or ""
        emb = get_embedding(text_for_embedding)
        assign_res = assign_or_create_topic_logic(art, emb)
        assignments.append(assign_res.dict())
    return {"assignments": assignments, "fetched": len(items)}

@app.get("/topics")
def list_all_topics():
    return {"topics": topics, "count": len(topics)}

@app.get("/topics/{topic_id}")
def get_topic(topic_id: str):
    for t in topics:
        if t["topic_id"] == topic_id:
            return t
    raise HTTPException(status_code=404, detail="topic not found")
