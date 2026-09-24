# React UI — Accuracy-Aware Medical Text Simplification

A React frontend + FastAPI backend for the same proposed model (GRU seq2seq + Luong attention,
GloVe-50d embedding init, fine-tuned, 20 epochs) used in the paper and the Gradio demo. This is
an alternative UI calling the real trained model through a REST API — same model, same
checkpoint, different interface.

```
web-app/
├── backend/     FastAPI server wrapping the trained model
└── frontend/    React (Vite) UI
```

## Running it (two terminals, both must stay open)

**Terminal 1 — backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Leave this running. It serves the API at `http://127.0.0.1:8000`.

**Terminal 2 — frontend:**
```bash
cd frontend
npm install
npm run dev
```
Leave this running too. It prints a local URL, typically `http://localhost:5173`.

Open that URL in your browser — the page will call the backend automatically.

## How it works

- `backend/main.py` — FastAPI app with two endpoints: `GET /api/examples` (the 5 preloaded
  Cochrane test-set pairs) and `POST /api/simplify` (runs the real model on whatever text you
  send it, returns the generated output plus readability/ROUGE/numeric-detail metrics).
- `backend/model.py` — the exact GRU + Luong attention architecture, must match the trained
  checkpoint.
- `backend/checkpoints/` — the trained weights (`proposed_model_weights.pt`) and vocabulary.
- `frontend/src/App.jsx` — the React UI. Calls the backend via `fetch()`.

## Notes

- The backend must be running before the frontend, or you'll see a "Could not reach the backend
  API" banner in the UI.
- CORS is configured in `main.py` to only allow `localhost:5173` — if you change the frontend's
  port, update `allow_origins` in `main.py` to match.
- Same research-prototype disclaimer as the paper applies: trained on a 900-example subsample,
  frequently invents specific numeric details not in the source. See the paper's Section XII/XV.
