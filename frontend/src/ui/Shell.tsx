import { useEffect, useMemo, useState } from "react";

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

  async function runPreview() {
    if (!apiBaseUrl) return;
    setErr(null);
    setBusy(true);
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/data/default`, {
        headers: { authorization: `Bearer ${token}` },
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

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Data</div>
      <div className="subtitle">Using default dataset: combinedv2Orapa.csv</div>

      <div style={{ marginTop: 12 }} className="grid2">
        <div>
          <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
            <button className="btn btnPrimary" onClick={runPreview} disabled={busy}>
              {busy ? "Loading…" : "Load Preview"}
            </button>
          </div>
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
  const [rrN, setRrN] = useState(1.8);
  const [rrMode, setRrMode] = useState<"manual" | "estimate">("estimate");
  const [rrXov, setRrXov] = useState(500);
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
          authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          inputs,
          hpd_override: 1,
          empirical: empiricalDefaults,
          want_ml: true,
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

  async function run() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
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
      <div className="subtitle">Using default dataset: flyrock_synth.csv</div>
      <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
        <button className="btn btnPrimary" onClick={run} disabled={busy}>{busy ? "Running…" : "Predict"}</button>
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
    </div>
  );
}

function SlopePanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [resp, setResp] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [inputs, setInputs] = useState<Record<string, number>>({});

  async function run() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      if (Object.keys(inputs).length) fd.append("inputs_json", JSON.stringify(inputs));
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/slope/predict`, {
        method: "POST",
        headers: { authorization: `Bearer ${token}` },
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
      <div className="subtitle">Using default dataset: slope data.csv</div>
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
      <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
        <button className="btn btnPrimary" onClick={run} disabled={busy}>{busy ? "Running…" : "Predict"}</button>
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
  const [resp, setResp] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/delay/predict`, {
        method: "POST",
        headers: { authorization: `Bearer ${token}` },
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
      <div className="subtitle">Using default dataset: Hole_data_v1.csv (fallback to v2)</div>
      <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
        <button className="btn btnPrimary" onClick={run} disabled={busy}>{busy ? "Running…" : "Predict"}</button>
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
        headers: { authorization: `Bearer ${token}` },
      });
      const json = await res.json();
      if (!res.ok) throw new Error("Failed");
      setResp(json);
      const p = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/feature/pca`, {
        headers: { authorization: `Bearer ${token}` },
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

  async function run() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/backbreak/predict`, {
        method: "POST",
        headers: { authorization: `Bearer ${token}` },
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
      <div className="subtitle">Using default dataset: Backbreak.csv</div>
      <div style={{ marginTop: 10 }}>
        <button className="btn btnPrimary" onClick={run} disabled={busy}>{busy ? "Running…" : "Predict"}</button>
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

  useEffect(() => {
    if (!apiBaseUrl) return;
    (async () => {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/param/meta`, {
        headers: { authorization: `Bearer ${token}` },
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
        headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
        body: JSON.stringify({ output, x1, x2, objective }),
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
        headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
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
      {goal?.best && (
        <div className="kpi" style={{ marginTop: 12 }}>
          <div className="kpiTitle">Goal Seek Predicted</div>
          <div className="kpiValue">{Number(goal.best.predicted).toFixed(2)}</div>
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

  useEffect(() => {
    if (!apiBaseUrl) return;
    (async () => {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/cost/defaults`, {
        headers: { authorization: `Bearer ${token}` },
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
        headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
        body: JSON.stringify(defaults),
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
        headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
        body: JSON.stringify(defaults),
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

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Cost Optimisation</div>
      <div className="subtitle">Uses CTk cost model defaults (no dataset selection).</div>
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
      <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <button className="btn btnPrimary" onClick={runCompute} disabled={busy}>
          {busy ? "Working…" : "Compute KPIs"}
        </button>
        <button className="btn" onClick={runOptimize} disabled={busy}>
          {busy ? "Optimising…" : "Optimise"}
        </button>
      </div>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      {resp && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="label">Result</div>
          <pre style={pre}>{JSON.stringify(resp, null, 2)}</pre>
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

