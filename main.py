import asyncio
import io
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from gradio_client import Client
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError, model_validator

os.environ["TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"] = "true"

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from fomo_scorer import evaluate_fomo
from live_stock_api import extract_ticker, get_live_stock_data
from stock_quiz import StockSuitabilityQuiz
from test_ocr_security import scan_screenshot
from volatility_alert import check_volatility

COLAB_API_URL = "https://bc58e19a8bd48db27f.gradio.live"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"
HISTORY_WINDOW = 6
HISTORY_LIMIT = 40
QUIZ_QUESTIONS = 3
QUIZ_LLM_ATTEMPTS = 2
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
WATCHLIST = ["RELIANCE.NS", "TATAMOTORS.NS", "ZOMATO.NS", "SUZLON.NS"]
DROP_THRESHOLD = -2.0
POLL_SECONDS = 60
SIMULATE_STREAM = True

ASSISTANT_PROMPT = (
    "You are BharatFinanceEdu, an expert Indian financial educator. "
    "Explain concepts simply, using analogies and dispelling misconceptions. "
    "Base your answers on the provided context. "
    "If the user acts impulsively, warn them based on SEBI rules."
)
QUIZ_PROMPT = (
    "You are a strict JSON quiz generator for an Indian financial education app. "
    "Generate exactly 3 multiple-choice questions assessing the user's readiness to invest in the given stock. "
    "Question 1 must assess Risk Tolerance and must use the provided Beta and expected drawdown figures. "
    "Question 2 must assess the Business Model. "
    "Question 3 must assess Capital Allocation. "
    "Output ONLY a valid JSON array of 3 objects with keys 'question', "
    "'options' (an object with keys A, B, C, D) and 'correct_answer' (one of A, B, C, D). "
    "Do not include markdown fences or any other text."
)

logger = logging.getLogger("relic")

llm_client: Optional[Client] = None
vector_db: Optional[Any] = None
chat_history: Dict[str, List[Dict[str, str]]] = {}
quiz_sessions: Dict[str, Dict[str, Any]] = {}
market_state: Dict[str, Any] = {"checked_at": None, "alerts": {}}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    rag_sources_used: bool
    live_data_used: bool


class QuizRequest(BaseModel):
    session_id: str
    target_stock: str


class QuizAnswer(BaseModel):
    question_index: int
    selected_option: Literal["A", "B", "C", "D"]


class QuizSubmit(BaseModel):
    session_id: str
    answers: List[QuizAnswer]


class QuizQuestion(BaseModel):
    question: str
    options: Dict[Literal["A", "B", "C", "D"], str]
    correct_answer: Literal["A", "B", "C", "D"]

    @model_validator(mode="after")
    def validate_options(self) -> "QuizQuestion":
        if len(self.options) != 4:
            raise ValueError("each question needs exactly 4 options")
        return self


class FomoRequest(BaseModel):
    message: str
    ticker: Optional[str] = None
    demo_mode: bool = False


def get_rag_context(query: str, k: int = 3) -> str:
    if vector_db is None:
        return ""
    try:
        docs = vector_db.similarity_search(query, k=k)
        return "\n".join(doc.page_content for doc in docs)
    except Exception:
        logger.exception("RAG lookup failed")
        return ""


def call_colab_llm(prompt: str, context: str, system_prompt: str) -> str:
    global llm_client
    try:
        if llm_client is None:
            llm_client = Client(COLAB_API_URL)
        return str(llm_client.predict(prompt, context, system_prompt, fn_index=0))
    except Exception:
        llm_client = None
        logger.exception("Colab inference call failed")
        raise HTTPException(status_code=503, detail="AI server is currently unreachable.")


def parse_quiz(text: str) -> List[QuizQuestion]:
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    questions = [QuizQuestion(**item) for item in json.loads(match.group(0) if match else text)]
    if len(questions) != QUIZ_QUESTIONS:
        raise ValueError(f"expected {QUIZ_QUESTIONS} questions, got {len(questions)}")
    return questions


def build_quiz_context(ticker: str, stock_name: str, beta: float, price: Any, pe_ratio: Any) -> str:
    expected_drop = round(beta * 10, 1)
    return (
        f"[LIVE DATA for {stock_name} ({ticker})]: Price: {price}. P/E: {pe_ratio}. "
        f"Beta: {beta:.2f}. If the NIFTY falls 10%, this stock could historically fall about {expected_drop}%.\n\n"
        f"SEBI guidance:\n{get_rag_context(f'risks of investing in {stock_name}')}"
    )


async def monitor_market() -> None:
    while True:
        for ticker in WATCHLIST:
            try:
                result = await asyncio.to_thread(check_volatility, ticker, DROP_THRESHOLD, SIMULATE_STREAM)
                market_state["alerts"][ticker] = result
            except Exception:
                logger.exception("Volatility check failed for %s", ticker)
        market_state["checked_at"] = datetime.now(timezone.utc).isoformat()
        await asyncio.sleep(POLL_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global vector_db
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    except Exception:
        logger.exception("ChromaDB unavailable, continuing without RAG")
    monitor = asyncio.create_task(monitor_market())
    yield
    monitor.cancel()


app = FastAPI(title="BharatFinanceEdu API", version="2.0", lifespan=lifespan)


@app.get("/")
def health_check() -> Dict[str, Any]:
    return {"status": "active", "rag_loaded": vector_db is not None, "colab_connected": llm_client is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    rag_context = get_rag_context(req.message)
    ticker = extract_ticker(req.message)
    live_data = get_live_stock_data(ticker) if ticker else ""
    context = "\n\n".join(part for part in (rag_context, live_data) if part)

    history = chat_history.setdefault(req.session_id, [])
    recent = "\n".join(f"{m['role']}: {m['content']}" for m in history[-HISTORY_WINDOW:])
    prompt = f"Previous Conversation:\n{recent}\n\nCurrent User Query: {req.message}" if recent else req.message

    reply = call_colab_llm(prompt, context, ASSISTANT_PROMPT)

    history.append({"role": "User", "content": req.message})
    history.append({"role": "AI", "content": reply})
    del history[:-HISTORY_LIMIT]

    return ChatResponse(reply=reply, rag_sources_used=bool(rag_context), live_data_used=bool(live_data))


@app.get("/chat/history/{session_id}")
def get_chat_history(session_id: str) -> Dict[str, Any]:
    return {"session_id": session_id, "history": chat_history.get(session_id, [])}


@app.post("/quiz/generate")
def generate_quiz(req: QuizRequest) -> Dict[str, Any]:
    ticker = extract_ticker(req.target_stock) or req.target_stock.strip().upper()
    stock = StockSuitabilityQuiz(ticker)
    info = stock.stock_info
    context = build_quiz_context(stock.ticker, info["name"], float(info["beta"]), info["price"], info["pe_ratio"])

    questions: Optional[List[QuizQuestion]] = None
    for _ in range(QUIZ_LLM_ATTEMPTS):
        raw = call_colab_llm(info["name"], context, QUIZ_PROMPT)
        try:
            questions = parse_quiz(raw)
            break
        except (ValueError, TypeError, ValidationError):
            logger.warning("Quiz JSON rejected, retrying")
    if questions is None:
        raise HTTPException(status_code=502, detail="AI server returned an invalid quiz.")

    quiz_sessions[req.session_id] = {"ticker": stock.ticker, "questions": questions}
    return {
        "session_id": req.session_id,
        "ticker": stock.ticker,
        "beta": info["beta"],
        "quiz": [{"question": q.question, "options": q.options} for q in questions],
    }


@app.post("/quiz/submit")
def submit_quiz(req: QuizSubmit) -> Dict[str, Any]:
    session = quiz_sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Quiz session not found.")

    questions: List[QuizQuestion] = session["questions"]
    indices = [a.question_index for a in req.answers]
    if sorted(indices) != list(range(len(questions))):
        raise HTTPException(status_code=400, detail="Provide exactly one answer for each question index.")
    del quiz_sessions[req.session_id]

    feedback = []
    for answer in req.answers:
        q = questions[answer.question_index]
        feedback.append(
            {
                "question": q.question,
                "user_answer": answer.selected_option,
                "correct_answer": q.correct_answer,
                "is_correct": answer.selected_option == q.correct_answer,
            }
        )

    score = sum(item["is_correct"] for item in feedback)
    eligible = score == len(questions)
    return {
        "ticker": session["ticker"],
        "score": score,
        "total": len(questions),
        "eligible": eligible,
        "readiness_rating": "Informed Investor" if eligible else "Speculative Buyer",
        "feedback": feedback,
    }


@app.post("/analyze-fomo")
def analyze_fomo(req: FomoRequest) -> Dict[str, Any]:
    ticker = req.ticker or extract_ticker(req.message)
    return evaluate_fomo(req.message, ticker, req.demo_mode)


@app.get("/market-alerts")
def market_alerts(triggered_only: bool = False) -> Dict[str, Any]:
    alerts = list(market_state["alerts"].values())
    if triggered_only:
        alerts = [a for a in alerts if a["triggered"]]
    return {
        "checked_at": market_state["checked_at"],
        "threshold_pct": DROP_THRESHOLD,
        "simulated": SIMULATE_STREAM,
        "alerts": alerts,
    }


@app.post("/scan-image")
def scan_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 5 MB limit.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not decode the image.")
    try:
        return scan_screenshot(image)
    except Exception:
        logger.exception("OCR pipeline failed")
        raise HTTPException(status_code=500, detail="OCR processing failed.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000)
