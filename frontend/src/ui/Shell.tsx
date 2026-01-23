import { useEffect, useMemo, useState } from "react";
import { REQUIRE_AUTH } from "../instant";

type Session = { token: string; email: string };
type Props = {
  apiBaseUrl: string;
  session: Session;
  onLogout: () => void;
};

type TabKey =
  | "data"
  | "predict"
  | "feature"
  | "param"
  | "cost"
  | "backbreak"
  | "flyrock"
  | "slope"
  | "delay";

const TABS: Array<{ key: TabKey; title: string; desc: string }> = [
  { key: "data", title: "Data", desc: "Upload / preview datasets (GCS-backed later)" },
  { key: "predict", title: "Predict", desc: "Empirical + ML outputs + RR" },
  { key: "feature", title: "Feature Importance", desc: "RF importance + PCA" },
  { key: "param", title: "Parameter Optimisation", desc: "Surface + goal seek" },
  { key: "cost", title: "Cost Optimisation", desc: "KPIs, optimise, Pareto" },
  { key: "backbreak", title: "Backbreak", desc: "RF model from CSV" },
  { key: "flyrock", title: "Flyrock", desc: "ML + empirical" },
  { key: "slope", title: "Slope", desc: "Stable/Failure classifier" },
  { key: "delay", title: "Delay", desc: "Delay prediction & plan view" },
];

const authHeaders = (token: string) =>
  REQUIRE_AUTH ? { authorization: `Bearer ${token}` } : {};

export function Shell({ apiBaseUrl, session, onLogout }: Props) {
  const [tab, setTab] = useState<TabKey>("predict");
  const [meta, setMeta] = useState<any>(null);
  const [metaErr, setMetaErr] = useState<string | null>(null);

  const headerRight = useMemo(() => {
    if (!REQUIRE_AUTH) return null;
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
          ) : tab === "param" ? (
            <ParamPanel apiBaseUrl={apiBaseUrl} token={session.token} />
          ) : tab === "cost" ? (
            <CostPanel apiBaseUrl={apiBaseUrl} token={session.token} />
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
  const [preview, setPreview] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploadResp, setUploadResp] = useState<any>(null);

  async function runPreview(customFile?: File | null) {
    if (!apiBaseUrl) return;
    setErr(null);
    setBusy(true);
    try {
      if (customFile) {
        const fd = new FormData();
        fd.append("file", customFile);
        const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/data/preview`, {
          method: "POST",
          headers: { ...authHeaders(token) },
          body: fd,
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json?.error ?? "Preview failed");
        setPreview(json);
        return;
      }
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/data/default`, {
        headers: { ...authHeaders(token) },
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

  async function runUpload() {
    if (!apiBaseUrl || !file) return;
    setErr(null);
    setBusy(true);
    setUploadResp(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/data/upload`, {
        method: "POST",
        headers: { ...authHeaders(token) },
        body: fd,
      });
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Upload failed");
      setUploadResp(json);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Data</div>
      <div className="subtitle">Upload a CSV or preview the default combined dataset.</div>

      <div style={{ marginTop: 12 }} className="grid2">
        <div>
          <label className="label">Upload CSV</label>
          <input
            className="input"
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
            <button className="btn btnPrimary" onClick={() => runPreview(file)} disabled={busy || !file}>
              {busy ? "Loading…" : "Preview Upload"}
            </button>
            <button className="btn" onClick={() => runPreview(null)} disabled={busy}>
              {busy ? "Loading…" : "Preview Default"}
            </button>
            <button className="btn" onClick={runUpload} disabled={busy || !file}>
              {busy ? "Uploading…" : "Upload to GCS"}
            </button>
          </div>
        </div>
        <div>
          {err && <div className="error">{err}</div>}
          {uploadResp?.gs_uri && (
            <div className="pill" style={{ marginTop: 8 }}>
              Uploaded: {uploadResp.gs_uri}
            </div>
          )}
          {preview && (
            <div style={{ marginTop: 8 }}>
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
                  <th
                    key={c}
                    style={{ textAlign: "left", borderBottom: "1px solid var(--border)", padding: "6px 4px" }}
                  >
                    {c}
                  </th>
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
  const [rrN, setRrN] = useState(1.8);
  const [rrMode, setRrMode] = useState<"manual" | "estimate">("estimate");
  const [rrXov, setRrXov] = useState(500);
  const [hpdOverride, setHpdOverride] = useState(1);
  const [wantMl, setWantMl] = useState(true);
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
    for (const k of meta.input_labels) {
      defaults[k] = meta?.input_stats?.[k]?.median ?? 0;
    }
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
          ...authHeaders(token),
        },
        body: JSON.stringify({
          inputs,
          hpd_override: hpdOverride,
          empirical: empiricalDefaults,
          want_ml: wantMl,
          rr_n: rrN,
          rr_mode: rrMode,
          rr_x_ov: rrXov
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
          <div className="label" style={{ marginBottom: 8 }}>Inputs (defaults from combinedv2Orapa.csv)</div>
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
          <div className="grid3" style={{ marginTop: 12 }}>
            <div>
              <label className="label">HPD override</label>
              <input
                className="input"
                type="number"
                value={hpdOverride}
                onChange={(e) => setHpdOverride(Number(e.target.value))}
              />
            </div>
            <div>
              <label className="label">Use ML</label>
              <select className="input" value={wantMl ? "yes" : "no"} onChange={(e) => setWantMl(e.target.value === "yes")}>
                <option value="yes">Yes</option>
                <option value="no">No (empirical only)</option>
              </select>
            </div>
            <div />
          </div>
        </div>
      ) : null}

      {out?.json?.rr ? (
        <div className="card">
          <div className="label" style={{ marginBottom: 8 }}>Rosin–Rammler (CDF)</div>
          <div className="grid3">
            <div>
              <label className="label">n mode</label>
              <select className="input" value={rrMode} onChange={(e) => setRrMode(e.target.value as any)}>
                <option value="estimate">Estimate</option>
                <option value="manual">Manual</option>
              </select>
            </div>
            <div>
              <label className="label">n (manual)</label>
              <input className="input" type="number" value={rrN} onChange={(e) => setRrN(Number(e.target.value))} />
            </div>
            <div>
              <label className="label">Oversize (mm)</label>
              <input className="input" type="number" value={rrXov} onChange={(e) => setRrXov(Number(e.target.value))} />
            </div>
          </div>
          <RRChart rr={out.json.rr} />
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
  const [resp, setResp] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [inputs, setInputs] = useState<Record<string, number>>({});
  const [file, setFile] = useState<File | null>(null);

  async function run() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      if (file) fd.append("file", file);
      if (Object.keys(inputs).length) fd.append("inputs_json", JSON.stringify(inputs));
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/flyrock/predict`, {
        method: "POST",
        headers: { ...authHeaders(token) },
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
      <div className="subtitle">Upload a CSV or use the default dataset.</div>
      <div style={{ marginTop: 10 }} className="grid2">
        <div>
          <label className="label">Upload CSV</label>
          <input className="input" type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </div>
        <div style={{ display: "flex", alignItems: "flex-end" }}>
          <button className="btn btnPrimary" onClick={run} disabled={busy}>
            {busy ? "Running…" : "Predict"}
          </button>
        </div>
      </div>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      {resp?.prediction != null && (
        <div className="kpi" style={{ marginTop: 12 }}>
          <div className="kpiTitle">Predicted flyrock distance</div>
          <div className="kpiValue">{Number(resp.prediction).toFixed(2)}</div>
          <div className="label">Train R²: {Number(resp.train_r2).toFixed(3)}</div>
        </div>
      )}
      {resp?.empirical && (
        <div style={{ marginTop: 12 }}>
          {Object.entries(resp.empirical).map(([k, v]) => (
            <div key={k} className="kpi" style={{ marginTop: 6 }}>
              <div className="kpiTitle">{k}</div>
              <div className="kpiValue">{Number(v).toFixed(2)}</div>
            </div>
          ))}
        </div>
      )}
      {resp?.feature_stats && (
        <div style={{ marginTop: 12 }}>
          <div className="label">Inputs</div>
          <div className="grid3" style={{ marginTop: 8 }}>
            {Object.keys(resp.feature_stats).map((k) => (
              <div key={k}>
                <label className="label">{k}</label>
                <input
                  className="input"
                  type="number"
                  value={inputs[k] ?? resp.feature_stats[k]?.median ?? 0}
                  onChange={(e) => setInputs({ ...inputs, [k]: Number(e.target.value) })}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function SlopePanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [resp, setResp] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [inputs, setInputs] = useState<Record<string, number>>({});
  const [file, setFile] = useState<File | null>(null);

  async function run() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      if (file) fd.append("file", file);
      if (Object.keys(inputs).length) fd.append("inputs_json", JSON.stringify(inputs));
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/slope/predict`, {
        method: "POST",
        headers: { ...authHeaders(token) },
        body: fd,
      });
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Slope failed");
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
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Slope Stability</div>
      <div className="subtitle">Upload a CSV or use the default dataset.</div>
      <div style={{ marginTop: 10 }} className="grid2">
        <div>
          <label className="label">Upload CSV</label>
          <input className="input" type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </div>
        <div style={{ display: "flex", alignItems: "flex-end" }}>
          <button className="btn btnPrimary" onClick={run} disabled={busy}>{busy ? "Running…" : "Predict"}</button>
        </div>
      </div>
      {Object.keys(inputs).length ? (
        <div className="grid3" style={{ marginTop: 10 }}>
          {Object.entries(inputs).map(([k, v]) => (
            <div key={k}>
              <label className="label">{k}</label>
              <input className="input" type="number" value={v} onChange={(e) => setInputs({ ...inputs, [k]: Number(e.target.value) })} />
            </div>
          ))}
        </div>
      ) : null}
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
  const [resp, setResp] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);

  async function run() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      if (file) fd.append("file", file);
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/delay/predict`, {
        method: "POST",
        headers: { ...authHeaders(token) },
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
      <div className="subtitle">Upload a CSV or use the default dataset (Hole_data_v1.csv).</div>
      <div style={{ marginTop: 10 }} className="grid2">
        <div>
          <label className="label">Upload CSV</label>
          <input className="input" type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </div>
        <div style={{ display: "flex", alignItems: "flex-end" }}>
          <button className="btn btnPrimary" onClick={run} disabled={busy}>{busy ? "Running…" : "Predict"}</button>
        </div>
      </div>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      {resp?.points?.length ? (
        <div className="kpi" style={{ marginTop: 12 }}>
          <div className="kpiTitle">Predicted points</div>
          <div className="kpiValue">{resp.points.length}</div>
        </div>
      ) : null}
      {resp?.points?.length ? <PlanView points={resp.points} /> : null}
    </div>
  );
}

function FeaturePanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [resp, setResp] = useState<any>(null);
  const [pca, setPca] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/feature/importance`, {
        headers: { ...authHeaders(token) },
      });
      const json = await res.json();
      if (!res.ok) throw new Error("Failed");
      setResp(json);
      const p = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/feature/pca`, {
        headers: { ...authHeaders(token) },
      });
      setPca(await p.json());
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
      {pca?.explained_variance_ratio && (
        <div style={{ marginTop: 12 }}>
          <div className="label">PCA Explained Variance</div>
          <div className="pill" style={{ marginTop: 6 }}>
            {pca.explained_variance_ratio.map((v: number, i: number) => `PC${i + 1}: ${(v * 100).toFixed(1)}%`).join(" · ")}
          </div>
        </div>
      )}
    </div>
  );
}

function BackbreakPanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [resp, setResp] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [inputs, setInputs] = useState<Record<string, number>>({});
  const [file, setFile] = useState<File | null>(null);

  async function run() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      if (file) fd.append("file", file);
      if (Object.keys(inputs).length) fd.append("inputs_json", JSON.stringify(inputs));
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/backbreak/predict`, {
        method: "POST",
        headers: { ...authHeaders(token) },
        body: fd,
      });
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Backbreak failed");
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
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Backbreak</div>
      <div className="subtitle">Upload a CSV or use the default dataset.</div>
      <div style={{ marginTop: 10 }} className="grid2">
        <div>
          <label className="label">Upload CSV</label>
          <input className="input" type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </div>
        <div style={{ display: "flex", alignItems: "flex-end" }}>
          <button className="btn btnPrimary" onClick={run} disabled={busy}>{busy ? "Running…" : "Predict"}</button>
        </div>
      </div>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      {resp?.prediction != null && (
        <div className="kpi" style={{ marginTop: 12 }}>
          <div className="kpiTitle">Predicted Backbreak</div>
          <div className="kpiValue">{Number(resp.prediction).toFixed(2)}</div>
        </div>
      )}
      {resp?.feature_stats && (
        <div style={{ marginTop: 12 }}>
          <div className="label">Inputs</div>
          <div className="grid3" style={{ marginTop: 8 }}>
            {Object.keys(resp.feature_stats).map((k) => (
              <div key={k}>
                <label className="label">{k}</label>
                <input
                  className="input"
                  type="number"
                  value={inputs[k] ?? resp.feature_stats[k]?.median ?? 0}
                  onChange={(e) => setInputs({ ...inputs, [k]: Number(e.target.value) })}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function RRChart({ rr }: { rr: any }) {
  const xs = rr?.xs ?? [];
  const ys = rr?.cdf ?? [];
  if (!xs.length) return null;
  const w = 600;
  const h = 240;
  const xlog = xs.map((x: number) => Math.log10(Math.max(0.1, x)));
  const xmin = Math.min(...xlog);
  const xmax = Math.max(...xlog);
  const ymin = 0;
  const ymax = 100;
  const pts = xlog.map((x: number, i: number) => {
    const px = ((x - xmin) / (xmax - xmin)) * (w - 20) + 10;
    const py = h - ((ys[i] - ymin) / (ymax - ymin)) * (h - 20) - 10;
    return `${px},${py}`;
  });
  return (
    <div style={{ marginTop: 10 }}>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "rgba(2,6,23,0.35)", borderRadius: 12 }}>
        <polyline fill="none" stroke="#60a5fa" strokeWidth="2" points={pts.join(" ")} />
      </svg>
      <div className="label" style={{ marginTop: 6 }}>
        n={rr.n?.toFixed(2)} · Xm={rr.xm?.toFixed(1)} mm · X50={rr.x50?.toFixed(1)} mm · Oversize@{rr.x_ov}={rr.oversize_pct?.toFixed(1)}%
      </div>
    </div>
  );
}

function PlanView({ points }: { points: Array<{ X: number; Y: number; Delay: number }> }) {
  const w = 620;
  const h = 360;
  const xs = points.map((p) => p.X);
  const ys = points.map((p) => p.Y);
  const ds = points.map((p) => p.Delay);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const dmin = Math.min(...ds);
  const dmax = Math.max(...ds);
  const norm = (v: number, a: number, b: number) => (b - a === 0 ? 0.5 : (v - a) / (b - a));
  return (
    <div style={{ marginTop: 12 }}>
      <div className="label">Plan View (color by delay)</div>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "rgba(2,6,23,0.35)", borderRadius: 12 }}>
        {points.slice(0, 800).map((p, i) => {
          const x = 10 + norm(p.X, xmin, xmax) * (w - 20);
          const y = 10 + (1 - norm(p.Y, ymin, ymax)) * (h - 20);
          const t = norm(p.Delay, dmin, dmax);
          const color = `hsl(${(1 - t) * 220}, 80%, 60%)`;
          return <circle key={i} cx={x} cy={y} r={3} fill={color} />;
        })}
      </svg>
    </div>
  );
}

function formatNum(v: any) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return Math.abs(n) >= 1000 ? n.toFixed(0) : n.toFixed(2);
}

function SurfaceHeatmap({
  gridX,
  gridY,
  Z,
  best,
  x1,
  x2,
}: {
  gridX: number[];
  gridY: number[];
  Z: number[][];
  best?: { value: number; inputs: Record<string, number> };
  x1: string;
  x2: string;
}) {
  if (!gridX?.length || !gridY?.length || !Z?.length) return null;
  const w = 620;
  const h = 360;
  const flat = Z.flat().filter((v) => Number.isFinite(v));
  const zmin = Math.min(...flat);
  const zmax = Math.max(...flat);
  const dx = w / gridX.length;
  const dy = h / gridY.length;
  const norm = (v: number) => (zmax - zmin === 0 ? 0.5 : (v - zmin) / (zmax - zmin));

  const bestX = best?.inputs?.[x1];
  const bestY = best?.inputs?.[x2];
  const bx = bestX == null ? null : (gridX.length > 1 ? ((bestX - gridX[0]) / (gridX[gridX.length - 1] - gridX[0])) * w : w / 2);
  const by = bestY == null ? null : (gridY.length > 1 ? (1 - (bestY - gridY[0]) / (gridY[gridY.length - 1] - gridY[0])) * h : h / 2);

  return (
    <div style={{ marginTop: 10 }}>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "rgba(2,6,23,0.35)", borderRadius: 12 }}>
        {Z.map((row, i) =>
          row.map((v, j) => {
            const t = norm(v);
            const color = `hsl(${220 - 200 * t}, 80%, 55%)`;
            return (
              <rect
                key={`${i}-${j}`}
                x={i * dx}
                y={h - (j + 1) * dy}
                width={dx + 0.5}
                height={dy + 0.5}
                fill={color}
                opacity={0.9}
              />
            );
          })
        )}
        {bx != null && by != null ? <circle cx={bx} cy={by} r={5} fill="#fbbf24" /> : null}
      </svg>
      <div className="label" style={{ marginTop: 6 }}>
        {x1} vs {x2} · min {formatNum(zmin)} / max {formatNum(zmax)}
      </div>
    </div>
  );
}

function BarChart({ labels, values }: { labels: string[]; values: number[] }) {
  const w = 620;
  const h = 220;
  const vmax = Math.max(...values.map((v) => Number(v) || 0), 1);
  const barW = w / Math.max(1, labels.length);
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "rgba(2,6,23,0.35)", borderRadius: 12 }}>
      {values.map((v, i) => {
        const val = Number(v) || 0;
        const bh = (val / vmax) * (h - 30);
        const x = i * barW + 12;
        const y = h - bh - 20;
        return (
          <g key={labels[i]}>
            <rect x={x} y={y} width={barW - 24} height={bh} fill="#60a5fa" rx={6} />
            <text x={x + (barW - 24) / 2} y={h - 6} fill="#94a3b8" fontSize="10" textAnchor="middle">
              {labels[i]}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function ParetoScatter({ rows }: { rows: Array<Record<string, any>> }) {
  if (!rows?.length) return null;
  const w = 620;
  const h = 260;
  const xs = rows.map((r) => Number(r.cost) || 0);
  const ys = rows.map((r) => Number(r["Oversize%"]) || 0);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const norm = (v: number, a: number, b: number) => (b - a === 0 ? 0.5 : (v - a) / (b - a));
  return (
    <div style={{ marginTop: 10 }}>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "rgba(2,6,23,0.35)", borderRadius: 12 }}>
        {rows.map((r, i) => {
          const x = 10 + norm(Number(r.cost) || 0, xmin, xmax) * (w - 20);
          const y = 10 + (1 - norm(Number(r["Oversize%"]) || 0, ymin, ymax)) * (h - 20);
          const t = norm(Number(r.PPV) || 0, Math.min(...rows.map((x) => Number(x.PPV) || 0)), Math.max(...rows.map((x) => Number(x.PPV) || 0)));
          const color = `hsl(${220 - 200 * t}, 80%, 55%)`;
          return <circle key={i} cx={x} cy={y} r={4} fill={color} opacity={0.9} />;
        })}
      </svg>
      <div className="label" style={{ marginTop: 6 }}>
        X: Cost · Y: Oversize% · Color: PPV
      </div>
    </div>
  );
}

function ParamPanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [meta, setMeta] = useState<any>(null);
  const [resp, setResp] = useState<any>(null);
  const [goal, setGoal] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [output, setOutput] = useState("");
  const [x1, setX1] = useState("");
  const [x2, setX2] = useState("");
  const [objective, setObjective] = useState<"min" | "max">("max");
  const [target, setTarget] = useState(0);
  const [grid, setGrid] = useState(25);
  const [samples, setSamples] = useState(40);

  useEffect(() => {
    if (!apiBaseUrl) return;
    (async () => {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/param/meta`, {
        headers: { ...authHeaders(token) },
      });
      const json = await res.json();
      setMeta(json);
      if (json?.outputs?.length) setOutput(json.outputs[0]);
      if (json?.inputs?.length > 1) {
        setX1(json.inputs[0]);
        setX2(json.inputs[1]);
      }
    })();
  }, [apiBaseUrl, token]);

  async function runSurface() {
    if (!apiBaseUrl || !output || !x1 || !x2) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/param/surface`, {
        method: "POST",
        headers: { "content-type": "application/json", ...authHeaders(token) },
        body: JSON.stringify({ output, x1, x2, objective, grid, samples }),
      });
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Surface failed");
      setResp(json);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  async function runGoal() {
    if (!apiBaseUrl || !output) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/param/goal-seek`, {
        method: "POST",
        headers: { "content-type": "application/json", ...authHeaders(token) },
        body: JSON.stringify({ output, target }),
      });
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Goal seek failed");
      setGoal(json);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Parameter Optimisation</div>
      <div className="subtitle">Using default dataset: combinedv2Orapa.csv</div>
      <div className="grid3" style={{ marginTop: 10 }}>
        <div>
          <label className="label">Output</label>
          <select className="input" value={output} onChange={(e) => setOutput(e.target.value)}>
            {(meta?.outputs ?? []).map((o: string) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="label">X-axis</label>
          <select className="input" value={x1} onChange={(e) => setX1(e.target.value)}>
            {(meta?.inputs ?? []).map((o: string) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="label">Y-axis</label>
          <select className="input" value={x2} onChange={(e) => setX2(e.target.value)}>
            {(meta?.inputs ?? []).map((o: string) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </div>
      </div>
      <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
        <button className="btn btnPrimary" onClick={runSurface} disabled={busy}>
          {busy ? "Running…" : "Optimise Surface"}
        </button>
        <select className="input" value={objective} onChange={(e) => setObjective(e.target.value as any)} style={{ width: 120 }}>
          <option value="max">Maximise</option>
          <option value="min">Minimise</option>
        </select>
        <input
          className="input"
          type="number"
          value={grid}
          onChange={(e) => setGrid(Number(e.target.value))}
          style={{ width: 120 }}
          placeholder="Grid"
        />
        <input
          className="input"
          type="number"
          value={samples}
          onChange={(e) => setSamples(Number(e.target.value))}
          style={{ width: 140 }}
          placeholder="Samples"
        />
      </div>
      <div className="grid2" style={{ marginTop: 12 }}>
        <div>
          <label className="label">Goal Seek Target</label>
          <input className="input" type="number" value={target} onChange={(e) => setTarget(Number(e.target.value))} />
          <button className="btn" style={{ marginTop: 8 }} onClick={runGoal} disabled={busy}>
            {busy ? "Running…" : "Goal Seek"}
          </button>
        </div>
        <div>{err && <div className="error">{err}</div>}</div>
      </div>
      {resp?.best && (
        <div className="kpi" style={{ marginTop: 12 }}>
          <div className="kpiTitle">Best {resp.output}</div>
          <div className="kpiValue">{Number(resp.best.value).toFixed(2)}</div>
        </div>
      )}
      {resp?.Z && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="label">Surface</div>
          <SurfaceHeatmap
            gridX={resp.grid_x}
            gridY={resp.grid_y}
            Z={resp.Z}
            best={resp.best}
            x1={resp.x1}
            x2={resp.x2}
          />
        </div>
      )}
      {resp?.best?.inputs && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="label">Best Inputs</div>
          <div className="grid3" style={{ marginTop: 8 }}>
            {Object.entries(resp.best.inputs).map(([k, v]: any) => (
              <div key={k} className="kpi">
                <div className="kpiTitle">{k}</div>
                <div className="kpiValue">{formatNum(v)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      {goal?.best && (
        <div className="kpi" style={{ marginTop: 12 }}>
          <div className="kpiTitle">Goal Seek Predicted</div>
          <div className="kpiValue">{Number(goal.best.predicted).toFixed(2)}</div>
        </div>
      )}
      {goal?.best?.inputs && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="label">Goal Seek Inputs</div>
          <div className="grid3" style={{ marginTop: 8 }}>
            {Object.entries(goal.best.inputs).map(([k, v]: any) => (
              <div key={k} className="kpi">
                <div className="kpiTitle">{k}</div>
                <div className="kpiValue">{formatNum(v)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function CostPanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [defaults, setDefaults] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);
  const [resp, setResp] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [weights, setWeights] = useState({ frag: 1.0, ppv: 1.0, air: 0.7 });
  const [useFrag, setUseFrag] = useState(true);
  const [usePpv, setUsePpv] = useState(true);
  const [useAir, setUseAir] = useState(true);
  const [method, setMethod] = useState("SLSQP");
  const [pareto, setPareto] = useState<any[] | null>(null);
  const [paretoBusy, setParetoBusy] = useState(false);

  useEffect(() => {
    if (!apiBaseUrl) return;
    (async () => {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/cost/defaults`, {
        headers: { ...authHeaders(token) },
      });
      const json = await res.json();
      setDefaults(json);
    })();
  }, [apiBaseUrl, token]);

  async function runCompute() {
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/cost/compute`, {
        method: "POST",
        headers: { "content-type": "application/json", ...authHeaders(token) },
        body: JSON.stringify({ ...defaults, weights, use_frag: useFrag, use_ppv: usePpv, use_air: useAir, method }),
      });
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Compute failed");
      setResp(json);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  async function runOptimize() {
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/cost/optimize`, {
        method: "POST",
        headers: { "content-type": "application/json", ...authHeaders(token) },
        body: JSON.stringify({ ...defaults, weights, use_frag: useFrag, use_ppv: usePpv, use_air: useAir, method }),
      });
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Optimise failed");
      setResp(json?.result ?? json);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  async function runPareto() {
    setParetoBusy(true);
    setErr(null);
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/cost/pareto`, {
        method: "POST",
        headers: { "content-type": "application/json", ...authHeaders(token) },
        body: JSON.stringify({ ...defaults, weights, use_frag: useFrag, use_ppv: usePpv, use_air: useAir, method }),
      });
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Pareto failed");
      setPareto(json?.rows ?? []);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setParetoBusy(false);
    }
  }

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Cost Optimisation</div>
      <div className="subtitle">Mirrors the CTk cost model with KPI + Pareto visuals.</div>
      <div className="grid3" style={{ marginTop: 10 }}>
        {Object.keys(defaults).map((k) => (
          <div key={k}>
            <label className="label">{k}</label>
            <input
              className="input"
              type="number"
              value={defaults[k]}
              onChange={(e) => setDefaults({ ...defaults, [k]: Number(e.target.value) })}
            />
          </div>
        ))}
      </div>
      <div className="grid3" style={{ marginTop: 12 }}>
        <div>
          <label className="label">Method</label>
          <select className="input" value={method} onChange={(e) => setMethod(e.target.value)}>
            <option value="SLSQP">SLSQP</option>
            <option value="trust-constr">trust-constr</option>
          </select>
        </div>
        <div>
          <label className="label">Weights</label>
          <div className="grid3">
            <input className="input" type="number" value={weights.frag} onChange={(e) => setWeights({ ...weights, frag: Number(e.target.value) })} />
            <input className="input" type="number" value={weights.ppv} onChange={(e) => setWeights({ ...weights, ppv: Number(e.target.value) })} />
            <input className="input" type="number" value={weights.air} onChange={(e) => setWeights({ ...weights, air: Number(e.target.value) })} />
          </div>
        </div>
        <div>
          <label className="label">Constraints</label>
          <div style={{ display: "grid", gap: 6 }}>
            <label className="label"><input type="checkbox" checked={useFrag} onChange={(e) => setUseFrag(e.target.checked)} /> Use fragmentation</label>
            <label className="label"><input type="checkbox" checked={usePpv} onChange={(e) => setUsePpv(e.target.checked)} /> Constrain PPV</label>
            <label className="label"><input type="checkbox" checked={useAir} onChange={(e) => setUseAir(e.target.checked)} /> Constrain Airblast</label>
          </div>
        </div>
      </div>
      <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <button className="btn btnPrimary" onClick={runCompute} disabled={busy}>
          {busy ? "Working…" : "Compute KPIs"}
        </button>
        <button className="btn" onClick={runOptimize} disabled={busy}>
          {busy ? "Optimising…" : "Optimise"}
        </button>
        <button className="btn" onClick={runPareto} disabled={paretoBusy}>
          {paretoBusy ? "Running…" : "Pareto"}
        </button>
      </div>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      {resp && (
        <div style={{ marginTop: 12, display: "grid", gap: 12 }}>
          <div className="grid3">
            <div className="kpi">
              <div className="kpiTitle">Cost</div>
              <div className="kpiValue">{formatNum(resp.cost)}</div>
            </div>
            <div className="kpi">
              <div className="kpiTitle">PPV</div>
              <div className="kpiValue">{formatNum(resp.PPV)}</div>
            </div>
            <div className="kpi">
              <div className="kpiTitle">Airblast</div>
              <div className="kpiValue">{formatNum(resp.L)}</div>
            </div>
            <div className="kpi">
              <div className="kpiTitle">X50</div>
              <div className="kpiValue">{formatNum(resp.X50)}</div>
            </div>
            <div className="kpi">
              <div className="kpiTitle">Oversize %</div>
              <div className="kpiValue">{formatNum(resp.oversize * 100)}</div>
            </div>
            <div className="kpi">
              <div className="kpiTitle">PF</div>
              <div className="kpiValue">{formatNum(resp.derived?.PF)}</div>
            </div>
          </div>
          {resp.cost_break && (
            <div className="card">
              <div className="label">Cost Breakdown</div>
              <BarChart
                labels={["Initiation", "Explosive", "Drilling"]}
                values={resp.cost_break}
              />
            </div>
          )}
          <div className="card">
            <div className="label">Derived</div>
            <pre style={pre}>{JSON.stringify(resp.derived, null, 2)}</pre>
          </div>
        </div>
      )}
      {pareto && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="label">Pareto (Cost vs Oversize%)</div>
          <ParetoScatter rows={pareto} />
          <pre style={{ ...pre, marginTop: 10 }}>{JSON.stringify(pareto.slice(0, 12), null, 2)}</pre>
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

