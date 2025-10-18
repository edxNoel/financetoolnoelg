from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from schemas import AnalyzeRequest, AnalyzeResponse
from providers import fetch_ohlc_stooq, fetch_news_stub
from reasoning import run_reasoning

load_dotenv()

app = FastAPI(title="Autonomous LangGraph Stock Explainer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    try:
        ohlc = await fetch_ohlc_stooq(req.ticker, req.start, req.end)
        news = await fetch_news_stub(req.ticker, req.start, req.end)
        result = run_reasoning(req.ticker, req.start, req.end, ohlc, news)
        result.setdefault("meta", {})
        result["meta"].update({
            "provider": "stooq",
            "ticker": req.ticker,
            "start": req.start,
            "end": req.end
        })
        return AnalyzeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
