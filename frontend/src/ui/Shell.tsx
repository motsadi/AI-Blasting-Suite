import { useEffect, useMemo, useState } from "react";

type Session = { token: string; email: string };
type Props = {
  apiBaseUrl: string;
  session: Session;
  onLogout: () => void;
};

type TabKey = "data" | "predict" | "feature" | "backbreak" | "flyrock" | "slope" | "delay";

const TABS: Array<{ key: TabKey; title: string; desc: string }> = [
  { key: "data", title: "Data", desc: "Upload / preview datasets (GCS-backed later)" },
  { key: "predict", title: "Predict", desc: "Empirical + ML outputs (API)" },
  { key: "feature", title: "Feature Importance", desc: "Model feature importances" },
  { key: "backbreak", title: "Backbreak", desc: "RF model from CSV" },
  { key: "flyrock", title: "Flyrock", desc: "Auto-train + predict (API later)" },
  { key: "slope", title: "Slope", desc: "Auto-train classifier (API later)" },
  { key: "delay", title: "Delay", desc: "Delay prediction & plan view (API later)" },
];

export function Shell({ apiBaseUrl, session, onLogout }: Props) {
  const [tab, setTab] = useState<TabKey>("predict");
  const [meta, setMeta] = useState<any>(null);
  const [metaErr, setMetaErr] = useState<string | null>(null);

  const headerRight = useMemo(() => {
    return (
      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <span className="pill">{session.email}</span>
        <button onClick={onLogout} className="btn">
          Logout
        </button>
      </div>
    );
  }, [onLogout, session.email]);

  useEffect(() => {
    if (!apiBaseUrl) return;
    const url = `${apiBaseUrl.replace(/\/$/, "")}/v1/meta`;
    (async () => {
      try {
        const res = await fetch(url);
        const j = await res.json();
        setMeta(j);
      } catch (e: any) {
        setMetaErr(String(e?.message ?? e));
      }
    })();
  }, [apiBaseUrl]);

  return (
    <div className="container">
      <div className="topbar">
        <div>
          <div className="title">AI Blasting Suite</div>
          <div className="subtitle">Secure auth • Cloud Run API • GCS assets • Modern UI</div>
        </div>
        {headerRight}
      </div>

      <div className="layout">
        <aside className="sidebar">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`navItem ${tab === t.key ? "navItemActive" : ""}`}
            >
              <div style={{ fontWeight: 650 }}>{t.title}</div>
              <div style={{ fontSize: 12, color: tab === t.key ? "var(--text)" : "var(--muted)" }}>
                {t.desc}
              </div>
            </button>
          ))}

          <div style={{ marginTop: 18, paddingTop: 16, borderTop: "1px solid #e2e8f0" }}>
            <div className="label">Backend</div>
            <div style={{ fontSize: 12, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>
              {apiBaseUrl || "(set VITE_API_BASE_URL)"}
            </div>
          </div>
        </aside>

        <main style={{ minHeight: 600 }}>
          {metaErr && <div className="error">{metaErr}</div>}
          {tab === "predict" ? (
            <PredictPanel apiBaseUrl={apiBaseUrl} token={session.token} meta={meta} />
          ) : tab === "data" ? (
            <DataPanel apiBaseUrl={apiBaseUrl} token={session.token} />
          ) : tab === "feature" ? (
            <FeaturePanel apiBaseUrl={apiBaseUrl} token={session.token} />
          ) : tab === "backbreak" ? (
            <BackbreakPanel apiBaseUrl={apiBaseUrl} token={session.token} />
          ) : tab === "flyrock" ? (
            <FlyrockPanel apiBaseUrl={apiBaseUrl} token={session.token} />
          ) : tab === "slope" ? (
            <SlopePanel apiBaseUrl={apiBaseUrl} token={session.token} />
          ) : tab === "delay" ? (
            <DelayPanel apiBaseUrl={apiBaseUrl} token={session.token} />
          ) : (
            <PlaceholderPanel title={TABS.find((t) => t.key === tab)!.title} />
          )}
        </main>
      </div>
    </div>
  );
}

function PlaceholderPanel({ title }: { title: string }) {
  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: "-0.02em" }}>{title}</div>
      <div style={{ color: "var(--muted)", marginTop: 8 }}>
        UI scaffolded. Next step is wiring this tab to FastAPI endpoints and GCS.
      </div>
    </div>
  );
}

function DataPanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState<string | null>(null);

  async function runPreview() {
    if (!apiBaseUrl || !file) return;
    setErr(null);
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/data/preview`, {
        method: "POST",
        headers: { authorization: `Bearer ${token}` },
        body: fd,
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.error ?? "Preview failed");
      setPreview(json);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  async function uploadToGcs() {
    if (!apiBaseUrl || !file) return;
    setErr(null);
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/data/upload`, {
        method: "POST",
        headers: { authorization: `Bearer ${token}` },
        body: fd,
      });
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Upload failed");
      setUploaded(json.gs_uri);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Data</div>
      <div className="subtitle">Upload CSV/XLSX, preview and store to GCS.</div>

      <div style={{ marginTop: 12 }} className="grid2">
        <div>
          <label className="label">Dataset file</label>
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            className="input"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
            <button className="btn btnPrimary" onClick={runPreview} disabled={!file || busy}>
              {busy ? "Working…" : "Preview"}
            </button>
            <button className="btn" onClick={uploadToGcs} disabled={!file || busy}>
              {busy ? "Uploading…" : "Upload to GCS"}
            </button>
          </div>
          {uploaded && <div className="pill" style={{ marginTop: 8 }}>{uploaded}</div>}
        </div>
        <div>
          {err && <div className="error">{err}</div>}
          {preview && (
            <div>
              <div className="label">Rows: {preview.rows}</div>
              <div className="label">Columns: {preview.columns?.length}</div>
            </div>
          )}
        </div>
      </div>

      {preview?.sample?.length ? (
        <div style={{ marginTop: 12, overflow: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr>
                {preview.columns.map((c: string) => (
                  <th key={c} style={{ textAlign: "left", borderBottom: "1px solid var(--border)", padding: "6px 4px" }}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.sample.map((row: any, idx: number) => (
                <tr key={idx}>
                  {preview.columns.map((c: string) => (
                    <td key={c} style={{ padding: "6px 4px", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                      {row?.[c] ?? ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}

function PredictPanel({ apiBaseUrl, token, meta }: { apiBaseUrl: string; token: string; meta: any }) {
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<any>(null);
  const [inputs, setInputs] = useState<Record<string, number>>({});
  const outputs: string[] = meta?.outputs ?? ["Ground Vibration", "Airblast", "Fragmentation"];
  const empiricalDefaults = meta?.empirical_defaults ?? {
    K_ppv: 1000,
    beta: 1.6,
    K_air: 170,
    B_air: 20,
    A_kuz: 22,
    RWS: 115,
  };

  useEffect(() => {
    if (!meta?.input_labels) return;
    const defaults: Record<string, number> = {};
    for (const k of meta.input_labels) defaults[k] = 0;
    setInputs(defaults);
  }, [meta]);

  async function run() {
    if (!apiBaseUrl) {
      setOut({ error: "Set VITE_API_BASE_URL in the frontend env." });
      return;
    }
    setBusy(true);
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/predict`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          inputs,
          hpd_override: 1,
          empirical: empiricalDefaults,
          want_ml: true
        }),
      });
      const json = await res.json();
      setOut({ status: res.status, json });
    } catch (e: any) {
      setOut({ error: String(e?.message ?? e) });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div className="card">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Predict</div>
            <div style={{ color: "var(--muted)", marginTop: 6 }}>
              Calls FastAPI <code>/v1/predict</code> (wrapping existing Python functions, unchanged).
            </div>
          </div>
          <button onClick={run} disabled={busy} className={`btn btnPrimary`}>
            {busy ? "Running…" : "Run sample prediction"}
          </button>
        </div>

        <div style={{ marginTop: 12 }} className="grid3">
          {outputs.map((k) => (
            <div key={k} className="kpi">
              <div className="kpiTitle">{k}</div>
              <div className="kpiValue">
                {out?.json?.ml?.[k] != null
                  ? Number(out.json.ml[k]).toFixed(2)
                  : out?.json?.empirical?.[k] != null
                    ? Number(out.json.empirical[k]).toFixed(2)
                    : "—"}
              </div>
              <div style={{ fontSize: 12, color: "var(--muted)" }}>
                {out?.json?.ml?.[k] != null ? "ML" : out?.json?.empirical?.[k] != null ? "Empirical" : ""}
              </div>
            </div>
          ))}
        </div>
      </div>

      {meta?.input_labels?.length ? (
        <div className="card">
          <div className="label" style={{ marginBottom: 8 }}>Inputs</div>
          <div className="grid3">
            {meta.input_labels.map((k: string) => (
              <div key={k}>
                <label className="label">{k}</label>
                <input
                  className="input"
                  type="number"
                  value={inputs[k] ?? 0}
                  onChange={(e) => setInputs({ ...inputs, [k]: Number(e.target.value) })}
                />
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="card">
        <div className="label" style={{ marginBottom: 10 }}>Response</div>
        <pre style={pre}>{JSON.stringify(out, null, 2)}</pre>
      </div>
    </div>
  );
}

function FlyrockPanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [resp, setResp] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [inputs, setInputs] = useState<Record<string, number>>({});

  async function run() {
    if (!file || !apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      if (Object.keys(inputs).length) fd.append("inputs_json", JSON.stringify(inputs));
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/flyrock/predict`, {
        method: "POST",
        headers: { authorization: `Bearer ${token}` },
        body: fd,
      });
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Flyrock failed");
      setResp(json);
      if (json?.feature_stats) {
        const next: Record<string, number> = {};
        Object.keys(json.feature_stats).forEach((k) => {
          next[k] = json.feature_stats[k].median;
        });
        setInputs(next);
      }
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Flyrock</div>
      <div className="subtitle">Upload dataset and get predicted flyrock distance.</div>
      <input type="file" accept=".csv,.xlsx,.xls" className="input" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
        <button className="btn btnPrimary" onClick={run} disabled={!file || busy}>{busy ? "Running…" : "Predict"}</button>
      </div>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      {resp?.prediction != null && (
        <div className="kpi" style={{ marginTop: 12 }}>
          <div className="kpiTitle">Predicted flyrock distance</div>
          <div className="kpiValue">{Number(resp.prediction).toFixed(2)}</div>
          <div className="label">Train R²: {Number(resp.train_r2).toFixed(3)}</div>
        </div>
      )}
    </div>
  );
}

function SlopePanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [resp, setResp] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    if (!file || !apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/slope/predict`, {
        method: "POST",
        headers: { authorization: `Bearer ${token}` },
        body: fd,
      });
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Slope failed");
      setResp(json);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Slope Stability</div>
      <div className="subtitle">Upload slope dataset and get stability probability.</div>
      <input type="file" accept=".csv,.xlsx,.xls" className="input" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
        <button className="btn btnPrimary" onClick={run} disabled={!file || busy}>{busy ? "Running…" : "Predict"}</button>
      </div>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      {resp?.prob_stable != null && (
        <div className="kpi" style={{ marginTop: 12 }}>
          <div className="kpiTitle">P(Stable)</div>
          <div className="kpiValue">{(Number(resp.prob_stable) * 100).toFixed(1)}%</div>
        </div>
      )}
    </div>
  );
}

function DelayPanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [resp, setResp] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    if (!file || !apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/delay/predict`, {
        method: "POST",
        headers: { authorization: `Bearer ${token}` },
        body: fd,
      });
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Delay failed");
      setResp(json);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Delay & Plan View</div>
      <div className="subtitle">Upload hole dataset and get predicted delays.</div>
      <input type="file" accept=".csv,.xlsx,.xls" className="input" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
        <button className="btn btnPrimary" onClick={run} disabled={!file || busy}>{busy ? "Running…" : "Predict"}</button>
      </div>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      {resp?.points?.length ? (
        <div className="kpi" style={{ marginTop: 12 }}>
          <div className="kpiTitle">Predicted points</div>
          <div className="kpiValue">{resp.points.length}</div>
        </div>
      ) : null}
    </div>
  );
}

function FeaturePanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [resp, setResp] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/feature-importance`, {
        headers: { authorization: `Bearer ${token}` },
      });
      const json = await res.json();
      if (!res.ok) throw new Error("Failed");
      setResp(json);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Feature Importance</div>
      <div className="subtitle">Uses model feature importances from the RF models.</div>
      <div style={{ marginTop: 10 }}>
        <button className="btn btnPrimary" onClick={run} disabled={busy}>
          {busy ? "Loading…" : "Load Importances"}
        </button>
      </div>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      {resp?.feature_importance && (
        <div style={{ marginTop: 12 }}>
          {Object.entries(resp.feature_importance).map(([name, items]: any) => (
            <div key={name} style={{ marginBottom: 12 }}>
              <div className="label">{name}</div>
              <div>
                {(items as any[]).slice(0, 8).map((it) => (
                  <div key={it.feature} className="kpi" style={{ marginTop: 6 }}>
                    <div className="kpiTitle">{it.feature}</div>
                    <div className="kpiValue">{Number(it.importance).toFixed(3)}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function BackbreakPanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [resp, setResp] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    if (!file || !apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/backbreak/predict`, {
        method: "POST",
        headers: { authorization: `Bearer ${token}` },
        body: fd,
      });
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Backbreak failed");
      setResp(json);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Backbreak</div>
      <div className="subtitle">Upload dataset and get predicted backbreak.</div>
      <input type="file" accept=".csv,.xlsx,.xls" className="input" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      <div style={{ marginTop: 10 }}>
        <button className="btn btnPrimary" onClick={run} disabled={!file || busy}>{busy ? "Running…" : "Predict"}</button>
      </div>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      {resp?.prediction != null && (
        <div className="kpi" style={{ marginTop: 12 }}>
          <div className="kpiTitle">Predicted Backbreak</div>
          <div className="kpiValue">{Number(resp.prediction).toFixed(2)}</div>
        </div>
      )}
    </div>
  );
}

const pre = {
  background: "#0b1220",
  color: "#e2e8f0",
  borderRadius: 12,
  padding: 14,
  overflow: "auto",
  maxHeight: 420,
} as const;

