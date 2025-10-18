# backend/reasoning.py
import os
import json
from typing import Any, Dict, List, TypedDict
from openai import OpenAI
from langgraph.graph import StateGraph, END


# ---------- LangGraph state ----------
class State(TypedDict, total=False):
    ticker: str
    start: str
    end: str
    ohlc: List[Dict[str, Any]]
    news: List[Dict[str, Any]]
    result: Dict[str, Any]


# ---------- Prompts ----------
SYSTEM = """You are an autonomous financial reasoning agent.
Construct an EXPLANATORY GRAPH (left-to-right) of thought nodes that explains why a stock's price
changed between the provided dates. Use ONLY the supplied OHLC and news. If you hypothesize, mark it as hypothesis.

Output STRICT JSON with keys: nodes (array), edges (array), meta (object). Example:
{
  "nodes": [
    {"id":"N0","label":"Overall change","text":"...","subnodes":[
      {"id":"N0.1","label":"Macro","text":"...","subnodes":[]},
      {"id":"N0.2","label":"Company events","text":"...","subnodes":[]}
    ]}
  ],
  "edges": [{"source":"N0","target":"N0.1"}],
  "meta": {"notes": "optional"}
}
No markdown fences or comments. Valid JSON only.
"""


# ---------- Helpers ----------
def _summ(ohlc: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not ohlc:
        return dict(n=0, start_close=None, end_close=None, abs=None, pct=0.0)
    s = ohlc[0].get("close")
    e = ohlc[-1].get("close")
    abs_ch = None
    pct = 0.0
    if s is not None and e is not None:
        abs_ch = e - s
        if s != 0:
            pct = abs_ch / s * 100
    return dict(n=len(ohlc), start_close=s, end_close=e, abs=abs_ch, pct=pct)


def _json_or_fallback(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {
            "nodes": [{"id": "N0", "label": "Model Output (raw)", "text": text}],
            "edges": [],
            "meta": {"notes": "LLM returned non-JSON; raw text captured."},
        }


# ---------- LangGraph node ----------
def reason_step(state: State) -> State:
    api_key = os.getenv("OPENAI_API_KEY")
    project_id = os.getenv("OPENAI_PROJECT_ID")  # works for sk-proj keys (optional)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Use project-scoped client if project_id is set; otherwise fall back to global key
    client = OpenAI(api_key=api_key, project=project_id) if project_id else OpenAI(api_key=api_key)

    summary = _summ(state.get("ohlc", []))
    user_payload = {
        "ticker": state["ticker"],
        "start": state["start"],
        "end": state["end"],
        "summary": summary,
        "ohlc_head": state.get("ohlc", [])[:5],
        "ohlc_tail": state.get("ohlc", [])[-5:],
        "news": state.get("news", [])[:6],
    }

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    resp = client.chat.completions.create(model=model, temperature=0.6, messages=messages)
    text = resp.choices[0].message.content.strip()

    # strip accidental code fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    result = _json_or_fallback(text)
    new_state: State = dict(state)
    new_state["result"] = result
    return new_state


# ---------- LangGraph builder ----------
def _build_graph():
    g = StateGraph(State)
    g.add_node("reason", reason_step)
    g.set_entry_point("reason")
    g.add_edge("reason", END)
    return g.compile()


# ---------- Public API (used by main.py) ----------
def run_reasoning(ticker: str, start: str, end: str,
                  ohlc: List[Dict[str, Any]], news: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Drop-in replacement that uses LangGraph under the hood.
    main.py calls this function.
    """
    graph = _build_graph()
    state: State = {"ticker": ticker, "start": start, "end": end, "ohlc": ohlc, "news": news}
    final = graph.invoke(state)
    return final.get("result", {"nodes": [], "edges": [], "meta": {"notes": "empty"}})
