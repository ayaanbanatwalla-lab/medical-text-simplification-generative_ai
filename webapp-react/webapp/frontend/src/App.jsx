import { useState, useEffect } from "react";

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [examples, setExamples] = useState([]);
  const [activeExample, setActiveExample] = useState(null);
  const [source, setSource] = useState("");
  const [reference, setReference] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/examples`)
      .then((r) => r.json())
      .then(setExamples)
      .catch(() => setError("Could not reach the backend API. Is main.py running on port 8000?"));
  }, []);

  const loadExample = (idx) => {
    const ex = examples[idx];
    setSource(ex.source);
    setReference(ex.reference);
    setActiveExample(idx);
    setResult(null);
  };

  const handleSimplify = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/simplify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source, reference: reference || null }),
      });
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message || "Something went wrong calling the API.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <h1 style={styles.title}>Accuracy-Aware Medical Text Simplification Using Generative AI</h1>
        <p style={styles.subtitle}>
          GRU sequence-to-sequence + Luong attention, <b>proposed model only</b> (GloVe-50d
          embedding initialization, fine-tuned, 20 epochs). See the paper and notebook in the
          repository for the full baseline comparison and evaluation.
        </p>

        <div style={styles.disclaimer}>
          <b>⚠ Research prototype, not a medical tool.</b> Trained on a 900-example subsample of the
          Cochrane medical text simplification corpus (technical review abstracts, not real patient
          reports) on CPU-only hardware, for a Generative AI course IA. The model frequently invents
          specific numbers/details not in the source — see the paper's Qualitative &amp; Error
          Analysis (Section XII) and Ethical Considerations (Section XV). Do not treat its output as
          medically accurate.
        </div>

        {error && <div style={styles.errorBanner}>{error}</div>}

        <div style={styles.row}>
          <div style={styles.col}>
            <div style={styles.panel}>
              <div style={styles.panelLabel}>Technical medical text (source)</div>
              <textarea
                style={styles.textarea}
                rows={7}
                placeholder="Paste a technical medical passage here, or click a sample below..."
                value={source}
                onChange={(e) => {
                  setSource(e.target.value);
                  setActiveExample(null);
                }}
              />
            </div>

            <div style={styles.panel}>
              <div style={styles.panelLabel}>
                Reference plain-language summary (optional — enables ROUGE / numeric-detail check)
              </div>
              <textarea
                style={styles.textarea}
                rows={5}
                placeholder="Optional. Leave blank if you don't have a reference."
                value={reference}
                onChange={(e) => setReference(e.target.value)}
              />
            </div>

            <button style={styles.button} onClick={handleSimplify} disabled={loading || !source.trim()}>
              {loading ? "Simplifying..." : "Simplify"}
            </button>

            {examples.length > 0 && (
              <>
                <div style={styles.examplesLabel}>
                  Sample test-set examples (real Cochrane pairs used in the paper's evaluation):
                </div>
                <div style={styles.examplesRow}>
                  {examples.map((_, i) => (
                    <button
                      key={i}
                      style={{
                        ...styles.exampleBtn,
                        ...(activeExample === i ? styles.exampleBtnActive : {}),
                      }}
                      onClick={() => loadExample(i)}
                    >
                      Example {i + 1}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>

          <div style={styles.col}>
            <div style={{ ...styles.panel, minHeight: 140 }}>
              <div style={styles.panelLabel}>Simplified output (generated)</div>
              <div style={styles.outputText}>{result ? result.output : ""}</div>
            </div>

            {result && (
              <>
                <h3 style={styles.sectionHeading}>
                  Readability (automated formula-based scores — not a human judgment)
                </h3>
                <table style={styles.table}>
                  <thead>
                    <tr>
                      <th style={styles.th}></th>
                      <th style={styles.th}>Source</th>
                      <th style={styles.th}>Generated</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td style={styles.tdLabel}>Flesch Reading Ease</td>
                      <td style={styles.td}>{fmt(result.source_fre)}</td>
                      <td style={styles.td}>{fmt(result.output_fre)}</td>
                    </tr>
                    <tr>
                      <td style={styles.tdLabel}>Flesch-Kincaid Grade</td>
                      <td style={styles.td}>{fmt(result.source_fkgl)}</td>
                      <td style={styles.td}>{fmt(result.output_fkgl)}</td>
                    </tr>
                  </tbody>
                </table>

                {result.rouge1 !== null && result.rouge1 !== undefined && (
                  <>
                    <h3 style={styles.sectionHeading}>
                      Similarity vs. reference (coarse signal, not factuality)
                    </h3>
                    <table style={styles.table}>
                      <thead>
                        <tr>
                          <th style={styles.th}>ROUGE-1</th>
                          <th style={styles.th}>ROUGE-2</th>
                          <th style={styles.th}>ROUGE-L</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td style={styles.td}>{fmt(result.rouge1, 3)}</td>
                          <td style={styles.td}>{fmt(result.rouge2, 3)}</td>
                          <td style={styles.td}>{fmt(result.rougeL, 3)}</td>
                        </tr>
                      </tbody>
                    </table>

                    <div style={{ fontSize: 12.5 }}>
                      <b>Numeric-detail preservation heuristic:</b>{" "}
                      {result.numeric_preserved ? (
                        <span style={{ color: "#34d399" }}>
                          ✅ at least one matching number preserved.
                        </span>
                      ) : (
                        <span style={{ color: "#fbbf24" }}>
                          ⚠️ no matching number from the reference found in the output (this is the
                          common failure mode documented in the paper's error analysis).
                        </span>
                      )}
                    </div>
                  </>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function fmt(n, decimals = 1) {
  if (n === null || n === undefined) return "—";
  return n.toFixed(decimals);
}

const styles = {
  page: {
    background: "#0b0f19",
    color: "#e5e7eb",
    minHeight: "100vh",
    fontFamily: "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
  },
  container: { maxWidth: 1180, margin: "0 auto", padding: "28px 40px" },
  title: { fontSize: 24, fontWeight: 700, color: "#f9fafb", marginBottom: 6 },
  subtitle: { fontSize: 13.5, color: "#9ca3af", marginBottom: 14, lineHeight: 1.5 },
  disclaimer: {
    background: "#2a1f0d",
    border: "1px solid #7c4a03",
    borderRadius: 8,
    padding: "12px 16px",
    fontSize: 12.5,
    color: "#fcd34d",
    lineHeight: 1.55,
    marginBottom: 22,
  },
  errorBanner: {
    background: "#2a0d0d",
    border: "1px solid #7c0303",
    borderRadius: 8,
    padding: "10px 16px",
    fontSize: 13,
    color: "#fca5a5",
    marginBottom: 16,
  },
  row: { display: "flex", gap: 20, flexWrap: "wrap" },
  col: { flex: 1, minWidth: 420 },
  panel: {
    background: "#151a26",
    border: "1px solid #2a3040",
    borderRadius: 10,
    padding: 16,
    marginBottom: 16,
  },
  panelLabel: { fontSize: 12.5, color: "#9ca3af", marginBottom: 8 },
  textarea: {
    width: "100%",
    background: "transparent",
    border: "none",
    outline: "none",
    resize: "vertical",
    color: "#e5e7eb",
    fontSize: 13.5,
    lineHeight: 1.55,
    fontFamily: "inherit",
  },
  outputText: { fontSize: 13.5, lineHeight: 1.55, whiteSpace: "pre-wrap" },
  button: {
    width: "100%",
    background: "#ea580c",
    color: "white",
    border: "none",
    textAlign: "center",
    fontWeight: 600,
    padding: 13,
    borderRadius: 8,
    fontSize: 14.5,
    marginBottom: 18,
    cursor: "pointer",
  },
  examplesLabel: { fontSize: 13, fontWeight: 600, color: "#d1d5db", marginBottom: 10 },
  examplesRow: { display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 10 },
  exampleBtn: {
    background: "#2a3040",
    border: "1px solid #3a4152",
    borderRadius: 7,
    padding: "9px 18px",
    fontSize: 12.5,
    color: "#d1d5db",
    cursor: "pointer",
  },
  exampleBtnActive: { background: "#3a4152", borderColor: "#ea580c", color: "#fff" },
  sectionHeading: { fontSize: 15, color: "#f3f4f6", margin: "18px 0 8px", fontWeight: 700 },
  table: { width: "100%", borderCollapse: "collapse", marginBottom: 14, fontSize: 12.5 },
  th: {
    border: "1px solid #2a3040",
    padding: "7px 10px",
    background: "#1e2433",
    color: "#d1d5db",
    fontWeight: 600,
  },
  td: { border: "1px solid #2a3040", padding: "7px 10px", textAlign: "center", color: "#e5e7eb" },
  tdLabel: { border: "1px solid #2a3040", padding: "7px 10px", textAlign: "left", color: "#e5e7eb" },
};
