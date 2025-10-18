import os, json
from typing import List, Dict, Any
from openai import OpenAI

SYSTEM = "You are a financial reasoning agent that builds an explanatory graph of why a stock changed in price."

def run_reasoning(ticker: str, start: str, end: str, ohlc: List[Dict[str, Any]], news: List[Dict[str, Any]]):
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    prompt = f"Ticker: {ticker}\nStart: {start}\nEnd: {end}\nExplain why the price changed using reasoning nodes."
    resp = client.chat.completions.create(model=model, temperature=0.6, messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt}
    ])
    text = resp.choices[0].message.content.strip()
    return {"nodes":[{"id":"N0","label":"Analysis","text":text}],"edges":[]}
