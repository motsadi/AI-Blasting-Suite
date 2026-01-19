import { useEffect, useMemo, useState } from "react";

type Session = { token: string; email: string };
type Props = {
  apiBaseUrl: string;
  session: Session;
  onLogout: () => void;
};

type TabKey = "data" | "predict" | "flyrock" | "slope" | "delay";

const TABS: Array<{ key: TabKey; title: string; desc: string }> = [
  { key: "data", title: "Data", desc: "Upload / preview datasets (GCS-backed later)" },
  { key: "predict", title: "Predict", desc: "Empirical + ML outputs (API)" },
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
            <DataPanel />
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

function DataPanel() {
  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: "-0.02em" }}>Data</div>
      <div style={{ color: "var(--muted)", marginTop: 8 }}>
        Next: upload CSV/XLSX here (client-side preview first), then we’ll add a backend endpoint to
        persist datasets to GCS and track metadata in InstantDB.
      </div>
    </div>
  );
}

function PredictPanel({ apiBaseUrl, token, meta }: { apiBaseUrl: string; token: string; meta: any }) {
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<any>(null);
  const outputs: string[] = meta?.outputs ?? ["Ground Vibration", "Airblast", "Fragmentation"];
  const empiricalDefaults = meta?.empirical_defaults ?? {
    K_ppv: 1000,
    beta: 1.6,
    K_air: 170,
    B_air: 20,
    A_kuz: 22,
    RWS: 115,
  };

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
          inputs: {
            "Hole depth (m)": 10,
            "Hole diameter (mm)": 102,
            "Burden (m)": 3,
            "Spacing (m)": 3.3,
            "Stemming (m)": 1.8,
            "Distance (m)": 300,
            "Powder factor (kg/m³)": 0.8,
            "Rock density (t/m³)": 2.7,
            "Linear charge (kg/m)": 15,
            "Explosive mass (kg)": 0,
            "Blast volume (m³)": 0,
            "# Holes": 30
          },
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

      <div className="card">
        <div className="label" style={{ marginBottom: 10 }}>Response</div>
        <pre style={pre}>{JSON.stringify(out, null, 2)}</pre>
      </div>
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

