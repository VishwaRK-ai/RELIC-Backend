import asyncio
import base64
import io
import json
import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from gradio_client import Client
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError, model_validator

os.environ["TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"] = "true"

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from asr_and_tts import run_multilingual_asr, synthesize_speech
from fomo_scorer import evaluate_fomo
from live_stock_api import extract_ticker, get_live_stock_data
from stock_quiz import StockSuitabilityQuiz
from test_ocr_security import analyze_image
from volatility_alert import check_volatility

COLAB_API_URL = "https://6c13cf704acb0b0763.gradio.live"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"
HISTORY_WINDOW = 6
HISTORY_LIMIT = 40
QUIZ_QUESTIONS = 3
QUIZ_LLM_ATTEMPTS = 2
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_AUDIO_BYTES = 10 * 1024 * 1024
WATCHLIST = ["RELIANCE.NS", "TATAMOTORS.NS", "ZOMATO.NS", "SUZLON.NS"]
DROP_THRESHOLD = -2.0
POLL_SECONDS = 60
SIMULATE_STREAM = True
DEFAULT_LANGUAGE = "en"
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}

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
    "Keep each question under 30 words and do not repeat the options inside the question text. "
    "Do not include markdown fences or any other text."
)

logger = logging.getLogger("relic")

llm_client: Optional[Client] = None
vector_db: Optional[Any] = None
chat_history: Dict[str, List[Dict[str, str]]] = {}
quiz_sessions: Dict[str, Dict[str, Any]] = {}
session_languages: Dict[str, str] = {}
market_state: Dict[str, Any] = {"checked_at": None, "alerts": {}}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    rag_sources_used: bool
    live_data_used: bool


class LanguageRequest(BaseModel):
    session_id: str
    language: str


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

class SpeakRequest(BaseModel):
    session_id: str
    text: str
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


def normalize_question(item: Dict[str, Any]) -> QuizQuestion:
    return QuizQuestion(
        question=re.split(r"\n\s*A[).:]", str(item["question"]))[0].strip(),
        options={str(k).strip().upper()[:1]: str(v).strip() for k, v in item["options"].items()},
        correct_answer=str(item["correct_answer"]).strip().upper()[:1],
    )


def parse_quiz(text: str) -> List[QuizQuestion]:
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    questions = [normalize_question(item) for item in json.loads(match.group(0) if match else text)]
    if len(questions) != QUIZ_QUESTIONS:
        raise ValueError(f"expected {QUIZ_QUESTIONS} questions, got {len(questions)}")
    return questions


def speak(text: str, language: str) -> bytes:
    clean = " ".join(re.sub(r"[*#_`>~|]+", " ", text).split())
    if not clean:
        raise HTTPException(status_code=400, detail="Nothing to speak after removing formatting.")
    try:
        return synthesize_speech(clean, language)
    except Exception:
        logger.exception("TTS failed for language %s", language)
        raise HTTPException(status_code=503, detail="Speech service is currently unreachable.")
def transcribe(upload: UploadFile) -> str:
    data = upload.file.read(MAX_AUDIO_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio exceeds 10 MB limit.")
    suffix = os.path.splitext(upload.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
    try:
        text, _ = run_multilingual_asr(tmp.name)
    except Exception:
        logger.exception("ASR failed")
        raise HTTPException(status_code=422, detail="Could not transcribe the audio.")
    finally:
        os.remove(tmp.name)
    if not text.strip():
        raise HTTPException(status_code=422, detail="No speech detected.")
    return text.strip()

def run_chat(session_id: str, message: str) -> ChatResponse:
    rag_context = get_rag_context(message)
    ticker = extract_ticker(message)
    live_data = get_live_stock_data(ticker) if ticker else ""
    context = "\n\n".join(part for part in (rag_context, live_data) if part)

    language = LANGUAGES[session_languages.get(session_id, DEFAULT_LANGUAGE)]
    history = chat_history.setdefault(session_id, [])
    recent = "\n".join(f"{m['role']}: {m['content']}" for m in history[-HISTORY_WINDOW:])
    query = f"{message}\n\nAnswer in {language}."
    prompt = f"Previous Conversation:\n{recent}\n\nCurrent User Query: {query}" if recent else query

    reply = call_colab_llm(prompt, context, f"{ASSISTANT_PROMPT} Always answer in {language}.")

    history.append({"role": "User", "content": message})
    history.append({"role": "AI", "content": reply})
    del history[:-HISTORY_LIMIT]

    return ChatResponse(reply=reply, rag_sources_used=bool(rag_context), live_data_used=bool(live_data))


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
    return run_chat(req.session_id, req.message)


@app.get("/chat/history/{session_id}")
def get_chat_history(session_id: str) -> Dict[str, Any]:
    return {"session_id": session_id, "history": chat_history.get(session_id, [])}


@app.get("/languages")
def list_languages() -> Dict[str, Any]:
    return {"default": DEFAULT_LANGUAGE, "languages": LANGUAGES}


@app.post("/language")
def set_language(req: LanguageRequest) -> Dict[str, str]:
    if req.language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language. Choose from {list(LANGUAGES)}.")
    session_languages[req.session_id] = req.language
    return {"session_id": req.session_id, "language": req.language}


@app.post("/quiz/generate")
def generate_quiz(req: QuizRequest) -> Dict[str, Any]:
    ticker = extract_ticker(req.target_stock) or req.target_stock.strip().upper()
    stock = StockSuitabilityQuiz(ticker)
    info = stock.stock_info
    context = build_quiz_context(stock.ticker, info["name"], float(info["beta"]), info["price"], info["pe_ratio"])

    language = session_languages.get(req.session_id, DEFAULT_LANGUAGE)
    system_prompt = (
        f"{QUIZ_PROMPT} Write every question and option in {LANGUAGES[language]}. "
        "Keep the JSON keys and the correct_answer letter in English."
    )

    questions: Optional[List[QuizQuestion]] = None
    for _ in range(QUIZ_LLM_ATTEMPTS):
        raw = call_colab_llm(f"{system_prompt}\n\nStock: {info['name']}", context, system_prompt)
        try:
            questions = parse_quiz(raw)
            break
        except (ValueError, TypeError, KeyError, AttributeError, ValidationError) as e:
            logger.warning("Quiz JSON rejected (%s). Raw output: %s", e, raw[:1500])
    if questions is None:
        raise HTTPException(status_code=502, detail="AI server returned an invalid quiz.")

    quiz_sessions[req.session_id] = {"ticker": stock.ticker, "questions": questions}
    return {
        "session_id": req.session_id,
        "ticker": stock.ticker,
        "beta": info["beta"],
        "language": language,
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


@app.post("/voice/speak")
def voice_speak(req: SpeakRequest) -> Response:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Provide the text to speak.")
    language = session_languages.get(req.session_id, DEFAULT_LANGUAGE)
    return Response(content=speak(req.text, language), media_type="audio/mpeg")


@app.post("/voice/listen")
def voice_listen(session_id: str = Form(...), audio: UploadFile = File(...)) -> Dict[str, Any]:
    language = session_languages.get(session_id, DEFAULT_LANGUAGE)
    transcript = transcribe(audio)
    result = run_chat(session_id, transcript)
    return {
        "transcript": transcript,
        "language": language,
        "reply": result.reply,
        "reply_audio_base64": base64.b64encode(speak(result.reply, language)).decode("ascii"),
        "audio_mime": "audio/mpeg",
        "rag_sources_used": result.rag_sources_used,
        "live_data_used": result.live_data_used,
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
        return analyze_image(image)
    except Exception:
        logger.exception("OCR pipeline failed")
        raise HTTPException(status_code=500, detail="OCR processing failed.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000)