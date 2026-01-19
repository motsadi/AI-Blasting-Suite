import { useMemo, useState } from "react";

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

  const headerRight = useMemo(() => {
    return (
      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <span style={{ fontSize: 13, color: "#475569" }}>{session.email}</span>
        <button onClick={onLogout} style={btn("secondary")}>
          Logout
        </button>
      </div>
    );
  }, [onLogout, session.email]);

  return (
    <div style={page}>
      <div style={topbar}>
        <div>
          <div style={{ fontWeight: 700, letterSpacing: "-0.02em" }}>AI Blasting Suite</div>
          <div style={{ fontSize: 12, color: "#64748b" }}>
            Modern React UI • FastAPI backend • InstantDB auth • GCS storage
          </div>
        </div>
        {headerRight}
      </div>

      <div style={layout}>
        <aside style={sidebar}>
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              style={navItem(tab === t.key)}
            >
              <div style={{ fontWeight: 650 }}>{t.title}</div>
              <div style={{ fontSize: 12, color: tab === t.key ? "#1f2937" : "#64748b" }}>
                {t.desc}
              </div>
            </button>
          ))}

          <div style={{ marginTop: 18, paddingTop: 16, borderTop: "1px solid #e2e8f0" }}>
            <div style={{ fontSize: 12, color: "#64748b" }}>Backend</div>
            <div style={{ fontSize: 12, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>
              {apiBaseUrl || "(set VITE_API_BASE_URL)"}
            </div>
          </div>
        </aside>

        <main style={main}>
          {tab === "predict" ? (
            <PredictPanel apiBaseUrl={apiBaseUrl} token={session.token} />
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
    <div style={card}>
      <div style={{ fontSize: 18, fontWeight: 700 }}>{title}</div>
      <div style={{ color: "#64748b", marginTop: 8 }}>
        UI scaffolded. Next step is wiring this tab to FastAPI endpoints and GCS.
      </div>
    </div>
  );
}

function PredictPanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<any>(null);

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
          empirical: { K_ppv: 1000, beta: 1.6, K_air: 170, B_air: 20, A_kuz: 22, RWS: 115 },
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
      <div style={card}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: "-0.02em" }}>Predict</div>
            <div style={{ color: "#64748b", marginTop: 6 }}>
              Calls FastAPI <code>/v1/predict</code> (wrapping existing Python functions, unchanged).
            </div>
          </div>
          <button onClick={run} disabled={busy} style={btn("primary")}>
            {busy ? "Running…" : "Run sample prediction"}
          </button>
        </div>
      </div>

      <div style={card}>
        <div style={{ fontSize: 13, color: "#64748b", marginBottom: 10 }}>Response</div>
        <pre style={pre}>{JSON.stringify(out, null, 2)}</pre>
      </div>
    </div>
  );
}

function btn(variant: "primary" | "secondary") {
  return {
    border: "1px solid",
    borderColor: variant === "primary" ? "#0f172a" : "#e2e8f0",
    background: variant === "primary" ? "#0f172a" : "white",
    color: variant === "primary" ? "white" : "#0f172a",
    padding: "10px 12px",
    borderRadius: 12,
    fontWeight: 650,
    cursor: "pointer",
  } as const;
}

const page = { minHeight: "100vh", background: "#0b1220", padding: 16 } as const;
const topbar = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  background: "linear-gradient(135deg, #ffffff, #f1f5f9)",
  border: "1px solid #e2e8f0",
  borderRadius: 16,
  padding: "14px 16px",
  boxShadow: "0 10px 30px rgba(15, 23, 42, 0.10)",
} as const;
const layout = { display: "grid", gridTemplateColumns: "320px 1fr", gap: 16, marginTop: 16 } as const;
const sidebar = {
  background: "white",
  borderRadius: 16,
  border: "1px solid #e2e8f0",
  padding: 12,
  boxShadow: "0 10px 30px rgba(15, 23, 42, 0.08)",
} as const;
const main = { minHeight: 600 } as const;
const card = {
  background: "white",
  borderRadius: 16,
  border: "1px solid #e2e8f0",
  padding: 16,
  boxShadow: "0 10px 30px rgba(15, 23, 42, 0.08)",
} as const;
const pre = {
  background: "#0b1220",
  color: "#e2e8f0",
  borderRadius: 12,
  padding: 14,
  overflow: "auto",
  maxHeight: 420,
} as const;
const navItem = (active: boolean) =>
  ({
    width: "100%",
    textAlign: "left",
    padding: "12px 12px",
    borderRadius: 14,
    border: "1px solid",
    borderColor: active ? "#0f172a" : "#e2e8f0",
    background: active ? "#0f172a" : "white",
    color: active ? "white" : "#0f172a",
    cursor: "pointer",
    marginBottom: 10,
  }) as const;

