# Demo App

Interactive Gradio demo for the **proposed model only** (GRU seq2seq + Luong attention,
GloVe-50d embedding initialization, fine-tuned, trained 20 epochs). For the full baseline
comparison and evaluation, see the paper and notebook in the parent repository.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Opens at `http://127.0.0.1:7860` by default.

## What it does

- Paste any technical medical text, or click one of 5 preloaded real test-set examples
- Generates a simplified version using the trained proposed model
- Shows automated readability scores (Flesch Reading Ease, FKGL) for source vs. generated output
- If a reference summary is provided (auto-filled for the preset examples), also shows
  ROUGE-1/2/L and a numeric-detail preservation check against that reference

## Files

- `app.py` — Gradio interface and evaluation display logic
- `model.py` — GRU + Luong attention architecture (must match the trained checkpoint exactly)
- `checkpoints/proposed_model_weights.pt` — trained weights (state_dict only, ~11 MB; optimizer
  state stripped out since it's not needed for inference)
- `checkpoints/vocab.json` — the 7,000-token vocabulary built from the training subsample
- `sample_examples.json` — 5 real Cochrane test-set (source, reference) pairs

## Important

This is a research prototype trained on a small subsample of the Cochrane corpus on CPU-only
hardware for a course IA. As documented in the paper's Qualitative & Error Analysis and Ethical
Considerations sections, it frequently invents specific numeric details not present in the
source text. It is not a validated medical tool and its output should not be treated as
medically accurate.
