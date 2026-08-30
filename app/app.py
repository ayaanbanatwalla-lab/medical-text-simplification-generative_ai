"""
Accuracy-Aware Medical Text Simplification — Gradio Demo
==========================================================
Interactive demo for the PROPOSED model (GRU seq2seq + Luong attention,
GloVe-50d embedding initialization, fine-tuned) from the paper:

  "Accuracy-Aware Medical Text Simplification Using Generative AI"
  Ayaan Banatwalla, Anjali Ashtankar — KJ Somaiya School of Engineering

This demo shows ONLY the proposed model. The baseline (random-embedding)
comparison, full training curves, and complete evaluation are in the
paper and notebook in this repository.

IMPORTANT: This is a research prototype trained on a small (900-example)
subsample of the Cochrane medical text simplification corpus, on CPU-only
hardware, for a course IA. It is NOT a validated medical tool. As shown
in the paper's error analysis, the model frequently invents specific
numbers and occasionally invents topics not present in the source text.
Do not use its output as a source of medical fact.
"""
import json
import re
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from rouge_score import rouge_scorer
import textstat

from model import Seq2Seq, tokenize, PAD, UNK, SOS, EOS

APP_DIR = Path(__file__).parent
CKPT_PATH = APP_DIR / "checkpoints" / "proposed_model_weights.pt"
VOCAB_PATH = APP_DIR / "checkpoints" / "vocab.json"
EXAMPLES_PATH = APP_DIR / "sample_examples.json"

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
    ids = [word2idx.get(w, UNK) for w in toks] + [EOS]
    return ids


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


# --------------------------------------------------------------- main ----
def run(source_text, reference_text):
    output = simplify(source_text)

    src_fre, src_fkgl = readability(source_text)
    out_fre, out_fkgl = readability(output)

    metrics_md = "### Readability (automated formula-based scores — not a human judgment)\n"
    metrics_md += "| | Source | Generated |\n|---|---|---|\n"
    metrics_md += f"| Flesch Reading Ease | {src_fre:.1f} | {out_fre:.1f} |\n" if src_fre and out_fre else "| N/A | N/A | N/A |\n"
    metrics_md += f"| Flesch-Kincaid Grade | {src_fkgl:.1f} | {out_fkgl:.1f} |\n" if src_fkgl and out_fkgl else ""

    if reference_text and reference_text.strip():
        scores = rouge.score(reference_text, output)
        metrics_md += "\n### Similarity vs. reference (coarse signal, not factuality)\n"
        metrics_md += "| ROUGE-1 | ROUGE-2 | ROUGE-L |\n|---|---|---|\n"
        metrics_md += f"| {scores['rouge1'].fmeasure:.3f} | {scores['rouge2'].fmeasure:.3f} | {scores['rougeL'].fmeasure:.3f} |\n"

        preserved = numeric_overlap(output, reference_text)
        if preserved is None:
            metrics_md += "\n**Numeric-detail preservation heuristic:** reference contains no specific numbers to check.\n"
        elif preserved:
            metrics_md += "\n**Numeric-detail preservation heuristic:** \u2705 at least one matching number preserved.\n"
        else:
            metrics_md += "\n**Numeric-detail preservation heuristic:** \u26a0\ufe0f no matching number from the reference found in the output (this is the common failure mode documented in the paper's error analysis).\n"

    return output, metrics_md


def load_example(idx):
    ex = examples[idx]
    return ex["source"], ex["reference"]


# ------------------------------------------------------------- UI ---
DISCLAIMER = """
> ⚠️ **Research prototype, not a medical tool.** Trained on a 900-example subsample of the Cochrane
> medical text simplification corpus (technical review abstracts, not real patient reports) on CPU-only
> hardware, for a Generative AI course IA. The model frequently invents specific numbers/details not in
> the source — see the paper's Qualitative & Error Analysis (Section XII) and Ethical Considerations
> (Section XV). Do not treat its output as medically accurate.
"""

with gr.Blocks(title="Accuracy-Aware Medical Text Simplification") as demo:
    gr.Markdown("# Accuracy-Aware Medical Text Simplification Using Generative AI")
    gr.Markdown(
        "GRU sequence-to-sequence + Luong attention, **proposed model only** "
        "(GloVe-50d embedding initialization, fine-tuned, 20 epochs). "
        "See the paper and notebook in this repository for the full baseline comparison and evaluation."
    )
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            source_box = gr.Textbox(
                label="Technical medical text (source)",
                lines=6,
                placeholder="Paste a technical medical passage here, or click a sample below...",
            )
            reference_box = gr.Textbox(
                label="Reference plain-language summary (optional — enables ROUGE / numeric-detail check)",
                lines=4,
                placeholder="Optional. Leave blank if you don't have a reference.",
            )
            run_btn = gr.Button("Simplify", variant="primary")

            if examples:
                gr.Markdown("**Sample test-set examples** (real Cochrane pairs used in the paper's evaluation):")
                with gr.Row():
                    for i in range(len(examples)):
                        btn = gr.Button(f"Example {i + 1}", size="sm")
                        btn.click(fn=lambda idx=i: load_example(idx), outputs=[source_box, reference_box])

        with gr.Column():
            output_box = gr.Textbox(label="Simplified output (generated)", lines=6)
            metrics_display = gr.Markdown()

    run_btn.click(fn=run, inputs=[source_box, reference_box], outputs=[output_box, metrics_display])

if __name__ == "__main__":
    demo.launch()
