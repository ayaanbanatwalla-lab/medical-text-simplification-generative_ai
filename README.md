# Accuracy-Aware Medical Text Simplification Using Generative AI

Generative AI IA-1 — Ayaan Banatwalla (16014223025), Anjali Ashtankar (16014223012)
KJ Somaiya School of Engineering

A Generative AI system that simplifies technical medical text (Cochrane systematic-review
abstracts) into plain language, built with a GRU sequence-to-sequence architecture and
Luong-style dot-product attention, trained from scratch on CPU-only hardware.

Two variants are compared under identical hyperparameters:
- **Baseline** — randomly initialized word embeddings
- **Proposed** — embeddings initialized from pretrained GloVe-50d vectors, fine-tuned during training

The full methodology, literature review, architecture, experimental results (ROUGE, readability,
GloVe-embedding similarity, numeric-detail preservation heuristic), qualitative/error analysis,
and honest discussion of the mixed baseline-vs-proposed outcome are in the paper.

**Read the paper for full context before drawing conclusions from the demo app or notebook alone** —
in particular, the paper is explicit that this is a small-scale research prototype, not a
validated medical tool, and that the model frequently invents specific numeric details not
present in the source text.

## Repository structure

```
├── app/            Interactive Gradio demo (proposed model only — see app/README.md)
├── notebook/        Full experiment notebook: dataset, both models, training, evaluation
├── paper/           Final research paper (IEEE format, PDF)
└── results/         Saved predictions, training logs, metrics, and figures from the actual runs
```

## Paper

[`paper/Accuracy-Aware_Medical_Text_Simplification.pdf`](paper/Accuracy-Aware_Medical_Text_Simplification.pdf)

## Notebook

[`notebook/medical_text_simplification.ipynb`](notebook/medical_text_simplification.ipynb) —
runs end-to-end: dataset acquisition and verification, preprocessing, both model variants,
training (with checkpoint/resume), full evaluation suite, comparison, and error analysis.

## Demo app

See [`app/README.md`](app/README.md) for setup and run instructions. Quick start:

```bash
cd app
pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints (typically `http://127.0.0.1:7860`).

## Results summary (n = 120 test examples, 20 epochs each)

| Metric | Baseline | Proposed |
|---|---:|---:|
| ROUGE-1 | 0.2803 | 0.2597 |
| ROUGE-2 | 0.0614 | 0.0479 |
| ROUGE-L | 0.1928 | 0.1814 |
| Flesch Reading Ease | 55.9 | 58.5 |
| Flesch-Kincaid Grade Level | 8.17 | 7.83 |
| GloVe cosine similarity (vs. ref.) | 0.9587 | 0.9573 |
| Numeric-detail preservation (of 84) | 5 | 4 |
| Final validation loss | 4.780 | 4.659 |

The proposed model consistently achieves lower validation loss, but this does not translate into
better ROUGE, similarity, or numeric-detail preservation — a genuinely mixed result discussed in
full in the paper's Comparative Analysis and Discussion sections.

## Dataset

Cochrane paragraph-level medical text simplification corpus (Devaraj et al., 2021, NAACL),
sourced from the [authors' repository](https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts),
CC-BY-4.0. A 900/100/120 train/validation/test subsample (seed = 42) was used due to the
1-CPU-core, no-GPU execution environment — see the paper's Dataset and Limitations sections.

## License

Code in this repository is provided for academic/coursework purposes. The Cochrane dataset
retains its original CC-BY-4.0 license from the source repository.
