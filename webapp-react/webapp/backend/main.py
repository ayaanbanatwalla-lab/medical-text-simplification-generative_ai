"""
FastAPI backend for the Medical Text Simplification demo.

Wraps the PROPOSED model (GRU seq2seq + Luong attention, GloVe-50d init, fine-tuned,
20 epochs) from "Accuracy-Aware Medical Text Simplification Using Generative AI" as a
REST API, for the React frontend to call.

Run with:  uvicorn main:app --reload --port 8000
"""
import json
import re
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rouge_score import rouge_scorer
import textstat

from model import Seq2Seq, tokenize, PAD, UNK, SOS, EOS

BASE_DIR = Path(__file__).parent
CKPT_PATH = BASE_DIR / "checkpoints" / "proposed_model_weights.pt"
VOCAB_PATH = BASE_DIR / "checkpoints" / "vocab.json"
EXAMPLES_PATH = BASE_DIR / "sample_examples.json"

MAX_SRC, MAX_TGT = 90, 45

# ---------------------------------------------------------------- load ---
with open(VOCAB_PATH) as f:
    vocab = json.load(f)
word2idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

model = Seq2Seq(V)
state_dict = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

examples = []
if EXAMPLES_PATH.exists():
    with open(EXAMPLES_PATH) as f:
        examples = json.load(f)


# ------------------------------------------------------------- helpers ---
def encode(text, max_len):
    toks = tokenize(text)[: max_len - 1]
    return [word2idx.get(w, UNK) for w in toks] + [EOS]


def pad_seq(ids, max_len):
    return ids + [PAD] * (max_len - len(ids))


def decode_ids(ids):
    words = []
    for i in ids:
        i = int(i)
        if i == EOS:
            break
        if i in (PAD, SOS):
            continue
        words.append(vocab[i])
    return " ".join(words)


def simplify(text: str) -> str:
    if not text or not text.strip():
        return ""
    ids = pad_seq(encode(text, MAX_SRC), MAX_SRC)
    src = torch.tensor([ids])
    with torch.no_grad():
        gen = model.generate(src, max_len=MAX_TGT)
    return decode_ids(gen[0])


def readability(text: str):
    if not text or len(text.split()) < 3:
        return None, None
    try:
        return textstat.flesch_reading_ease(text), textstat.flesch_kincaid_grade(text)
    except Exception:
        return None, None


def numeric_overlap(pred: str, ref: str):
    ref_nums = set(re.findall(r"\d+(?:\.\d+)?", ref))
    pred_nums = set(re.findall(r"\d+(?:\.\d+)?", pred))
    if not ref_nums:
        return None
    return len(ref_nums & pred_nums) > 0


# --------------------------------------------------------------- API -----
app = FastAPI(title="Medical Text Simplification API")

# Allow the local React dev server to call this API (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimplifyRequest(BaseModel):
    source: str
    reference: str | None = None


class SimplifyResponse(BaseModel):
    output: str
    source_fre: float | None = None
    source_fkgl: float | None = None
    output_fre: float | None = None
    output_fkgl: float | None = None
    rouge1: float | None = None
    rouge2: float | None = None
    rougeL: float | None = None
    numeric_preserved: bool | None = None


@app.get("/api/examples")
def get_examples():
    """Returns the 5 preloaded real Cochrane test-set (source, reference) pairs."""
    return examples


@app.post("/api/simplify", response_model=SimplifyResponse)
def simplify_endpoint(req: SimplifyRequest):
    output = simplify(req.source)
    src_fre, src_fkgl = readability(req.source)
    out_fre, out_fkgl = readability(output)

    resp = SimplifyResponse(
        output=output,
        source_fre=src_fre,
        source_fkgl=src_fkgl,
        output_fre=out_fre,
        output_fkgl=out_fkgl,
    )

    if req.reference and req.reference.strip():
        scores = rouge.score(req.reference, output)
        resp.rouge1 = scores["rouge1"].fmeasure
        resp.rouge2 = scores["rouge2"].fmeasure
        resp.rougeL = scores["rougeL"].fmeasure
        resp.numeric_preserved = numeric_overlap(output, req.reference)

    return resp


@app.get("/api/health")
def health():
    return {"status": "ok", "vocab_size": V}
