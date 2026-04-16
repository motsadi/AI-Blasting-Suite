import { useEffect, useMemo, useRef, useState } from "react";

type Session = { token: string; email: string };
type Props = {
  apiBaseUrl: string;
  session: Session;
  onLogout: () => void;
};

type TabKey =
  | "home"
  | "data"
  | "predict"
  | "feature"
  | "param"
  | "cost"
  | "backbreak"
  | "flyrock"
  | "slope"
  | "delay";

const TAB_META: Record<TabKey, { title: string; desc: string; icon: string }> = {
  home: { title: "Welcome", desc: "Overview and quick access cards", icon: "🏠" },
  data: { title: "Data Manager", desc: "Load, preview, filter and export datasets", icon: "🗂️" },
  predict: { title: "Prediction", desc: "Empirical + ML outputs + RR", icon: "📊" },
  feature: { title: "Feature Importance", desc: "RF importance + PCA", icon: "🧭" },
  param: { title: "Parameter Optimisation", desc: "Surface + goal seek", icon: "🧪" },
  cost: { title: "Cost Optimisation", desc: "KPIs, optimise, Pareto", icon: "💥" },
  backbreak: { title: "Back Break", desc: "RF model from CSV", icon: "🔧" },
  flyrock: { title: "Flyrock (ML + Empirical)", desc: "ML + empirical lines", icon: "🪨" },
  slope: { title: "Slope Stability", desc: "Stable/Failure classifier", icon: "🧱" },
  delay: { title: "Delay Prediction", desc: "Delay prediction & plan view", icon: "⏱️" },
};

const NAV_GROUPS: Array<{ title: string; items: TabKey[] }> = [
  { title: "Analysis", items: ["predict", "feature", "param"] },
  { title: "Operations", items: ["cost", "delay"] },
  { title: "Safety / Geo", items: ["slope", "backbreak", "flyrock"] },
  { title: "Admin", items: ["data"] },
];

const authHeaders = (token: string) => ({ authorization: `Bearer ${token}` });

export function Shell({ apiBaseUrl, session, onLogout }: Props) {
  const [tab, setTab] = useState<TabKey>("home");
  const [meta, setMeta] = useState<any>(null);
  const [metaErr, setMetaErr] = useState<string | null>(null);
  const [dataset, setDataset] = useState<{
    file?: File | null;
    rows: Array<Record<string, any>>;
    columns: string[];
  }>({ file: null, rows: [], columns: [] });
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [theme, setTheme] = useState<"System" | "Light" | "Dark">("System");
  const [accent, setAccent] = useState<"blue" | "green" | "dark-blue">("blue");

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

  const datasetChoices: string[] = useMemo(() => {
    const fromMeta = meta?.combined_dataset_choices as string[] | undefined;
    if (Array.isArray(fromMeta) && fromMeta.length) return fromMeta;
    // Fallback (backend not yet deployed with dataset registry)
    return ["combinedv2Orapa.csv", "combinedv2Jwaneng.csv"];
  }, [meta?.combined_dataset_choices]);

  const activeCombinedDataset: string = useMemo(() => {
    return (meta?.combined_dataset ?? meta?.default_dataset ?? datasetChoices[0] ?? "") as string;
  }, [meta?.combined_dataset, meta?.default_dataset, datasetChoices]);

  async function readJsonOrText(res: Response): Promise<any> {
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    if (ct.includes("application/json")) {
      try {
        return await res.json();
      } catch {
        return null;
      }
    }
    try {
      return await res.text();
    } catch {
      return null;
    }
  }

  function errorFromBody(res: Response, body: any): string {
    if (body && typeof body === "object") {
      const msg = body?.detail ?? body?.error;
      if (msg) return String(msg);
    }
    if (typeof body === "string") {
      const s = body.trim();
      if (s) return s.length > 400 ? s.slice(0, 400) + "…" : s;
    }
    return `HTTP ${res.status}`;
  }

  async function refreshMeta() {
    if (!apiBaseUrl) return;
    const url = `${apiBaseUrl.replace(/\/$/, "")}/v1/meta`;
    const res = await fetch(url);
    const body = await readJsonOrText(res);
    if (!res.ok) throw new Error(errorFromBody(res, body));
    setMeta(body);
    setMetaErr(null);
  }

  useEffect(() => {
    if (!apiBaseUrl) return;
    (async () => {
      try {
        await refreshMeta();
      } catch (e: any) {
        setMetaErr(String(e?.message ?? e));
      }
    })();
  }, [apiBaseUrl]);

  async function setCombinedDataset(name: string) {
    if (!apiBaseUrl) return;
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/datasets/combined`, {
        method: "POST",
        headers: { "content-type": "application/json", ...authHeaders(session.token) },
        body: JSON.stringify({ name }),
      });
      const body = await readJsonOrText(res);
      if (!res.ok) throw new Error(errorFromBody(res, body));
      // Clear any shared in-memory dataset preview so modules use the active backend dataset.
      setDataset({ file: null, rows: [], columns: [] });
      await refreshMeta();
    } catch (e: any) {
      setMetaErr(String(e?.message ?? e));
    }
  }

  useEffect(() => {
    const body = document.body;
    if (theme === "System") {
      delete body.dataset.theme;
    } else {
      body.dataset.theme = theme.toLowerCase();
    }
  }, [theme]);

  useEffect(() => {
    document.body.dataset.accent = accent;
  }, [accent]);

  return (
    <div className="container">
      <div className="header">
        <div className="headerLeft">
          <button className="iconBtn" onClick={() => setSidebarOpen((v) => !v)} aria-label="Toggle sidebar">
            ☰
          </button>
          <div className="headerBrand">
            <button className="headerTitle" onClick={() => setTab("home")} aria-label="Go to home">
              Blasting Optimization Suite
            </button>
            <div className="headerSubtext">
              AI-driven blast design, cost-aware optimisation and safety analytics in one workspace.
            </div>
          </div>
        </div>

        <div className="headerControls">
          <label className="selectWrap">
            <span className="label">Dataset</span>
            <select
              className="select"
              value={activeCombinedDataset}
              onChange={(e) => setCombinedDataset(e.target.value)}
              disabled={!apiBaseUrl}
              aria-label="Select dataset"
            >
              {datasetChoices.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
          <label className="selectWrap">
            <span className="label">Theme</span>
            <select className="select" value={theme} onChange={(e) => setTheme(e.target.value as typeof theme)}>
              <option value="System">System</option>
              <option value="Light">Light</option>
              <option value="Dark">Dark</option>
            </select>
          </label>
          <label className="selectWrap">
            <span className="label">Accent</span>
            <select className="select" value={accent} onChange={(e) => setAccent(e.target.value as typeof accent)}>
              <option value="blue">blue</option>
              <option value="green">green</option>
              <option value="dark-blue">dark-blue</option>
            </select>
          </label>
          <button className="btn" onClick={() => window.location.reload()}>
            Reload
          </button>
        </div>

        <div className="headerRight">{headerRight}</div>
      </div>

      <div className={`layout ${sidebarOpen ? "" : "layoutCollapsed"}`}>
        {sidebarOpen && (
          <aside className="sidebar">
            {NAV_GROUPS.map((group) => (
              <div key={group.title} className="sidebarGroup">
                <div className="sidebarTitle">{group.title}</div>
                {group.items.map((key) => {
                  const meta = TAB_META[key];
                  return (
                    <button
                      key={key}
                      onClick={() => setTab(key)}
                      className={`sidebarButton ${tab === key ? "sidebarButtonActive" : ""}`}
                    >
                      <div className="sidebarButtonLabel">
                        <span>{meta.icon}</span>
                        <span>{meta.title}</span>
                      </div>
                      <div className="sidebarButtonDesc">{meta.desc}</div>
                    </button>
                  );
                })}
              </div>
            ))}

            <div className="sidebarFooter">
              <div className="label">Backend</div>
              <div className="mono">{apiBaseUrl || "(set VITE_API_BASE_URL)"}</div>
            </div>
          </aside>
        )}

        <main className="mainContent">
          {metaErr && <div className="error">{metaErr}</div>}
          {tab === "home" ? (
            <HomePanel onOpen={setTab} />
          ) : tab === "predict" ? (
            <PredictPanel apiBaseUrl={apiBaseUrl} token={session.token} meta={meta} dataset={dataset} />
          ) : tab === "data" ? (
            <DataPanel apiBaseUrl={apiBaseUrl} token={session.token} dataset={dataset} onDatasetChange={setDataset} />
          ) : tab === "feature" ? (
            <FeaturePanel
              apiBaseUrl={apiBaseUrl}
              token={session.token}
              dataset={dataset}
              activeDatasetName={activeCombinedDataset}
            />
          ) : tab === "param" ? (
            <ParamPanel
              apiBaseUrl={apiBaseUrl}
              token={session.token}
              dataset={dataset}
              activeDatasetName={activeCombinedDataset}
            />
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
            <PlaceholderPanel title={TAB_META[tab]?.title ?? "Module"} />
          )}
        </main>
      </div>
    </div>
  );
}

function HomePanel({ onOpen }: { onOpen: (t: TabKey) => void }) {
  return (
    <div className="card">
      <div className="homeTitle">Welcome 👋</div>
      <div className="homeSubtitle">
        AI-driven blast design • Cost &amp; constraint-aware optimisation • USBM + Kuz–Ram empirical baselines
      </div>
      <div className="homeHighlights">
        <div className="highlightPill">Prediction: empirical + ML outputs + RR</div>
        <div className="highlightPill">Cost Optimisation: KPI, Pareto and penalties</div>
        <div className="highlightPill">Flyrock: ML estimator + empirical checks</div>
      </div>
      <div className="homeGrid">
        <div className="homeCard">
          <div className="homeCardIcon">{TAB_META.predict.icon}</div>
          <div className="homeCardTitle">{TAB_META.predict.title}</div>
          <div className="homeCardDesc">
            Run ML &amp; empirical predictions (USBM PPV/Air, Kuz–Ram Xm + RR curve).
          </div>
          <div className="homeCardActions">
            <button className="btn btnPrimary" onClick={() => onOpen("predict")}>
              Open
            </button>
          </div>
        </div>
        <div className="homeCard">
          <div className="homeCardIcon">{TAB_META.cost.icon}</div>
          <div className="homeCardTitle">{TAB_META.cost.title}</div>
          <div className="homeCardDesc">
            Minimise cost with penalties for PPV, airblast, fragmentation (Xm→RR X50).
          </div>
          <div className="homeCardActions">
            <button className="btn btnPrimary" onClick={() => onOpen("cost")}>
              Open
            </button>
          </div>
        </div>
        <div className="homeCard">
          <div className="homeCardIcon">{TAB_META.flyrock.icon}</div>
          <div className="homeCardTitle">{TAB_META.flyrock.title}</div>
          <div className="homeCardDesc">
            Predict flyrock (ML + empirical lines); check limits and distances.
          </div>
          <div className="homeCardActions">
            <button className="btn btnPrimary" onClick={() => onOpen("flyrock")}>
              Open
            </button>
          </div>
        </div>
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

function DataPanel({
  apiBaseUrl,
  token,
  dataset,
  onDatasetChange,
}: {
  apiBaseUrl: string;
  token: string;
  dataset: { file?: File | null; rows: Array<Record<string, any>>; columns: string[] };
  onDatasetChange: (d: { file?: File | null; rows: Array<Record<string, any>>; columns: string[] }) => void;
}) {
  const [data, setData] = useState<Array<Record<string, any>>>(dataset.rows ?? []);
  const [filtered, setFiltered] = useState<Array<Record<string, any>>>(dataset.rows ?? []);
  const [columns, setColumns] = useState<string[]>(dataset.columns ?? []);
  const [tab, setTab] = useState<"table" | "summary" | "visuals" | "corr" | "filters" | "calib" | "export">("table");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Rows: 0 | Columns: 0");
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState<Record<string, { min?: string; max?: string }>>({});
  const [auditInputs, setAuditInputs] = useState({ ppv: "12.5", air: "134", hpd: "1" });
  const [calibHpd, setCalibHpd] = useState("1");
  const [calibLog, setCalibLog] = useState<string[]>([]);
  const [siteModel, setSiteModel] = useState<Record<string, any>>({});
  const [plotType, setPlotType] = useState("Scatter");
  const [xVar, setXVar] = useState("");
  const [yVar, setYVar] = useState("");
  const [logX, setLogX] = useState(false);
  const [logY, setLogY] = useState(false);
  const [addRowValues, setAddRowValues] = useState<Record<string, string>>({});
  const loadInputRef = useRef<HTMLInputElement | null>(null);
  const appendInputRef = useRef<HTMLInputElement | null>(null);
  const didAutoLoad = useRef(false);

  const numericCols = useMemo(() => getNumericColumns(data, columns), [data, columns]);

  useEffect(() => {
    setStatus(`Rows: ${filtered.length} | Columns: ${columns.length}`);
  }, [filtered.length, columns.length]);

  useEffect(() => {
    if (!columns.length) return;
    if (!xVar) setXVar(columns[0]);
    if (!yVar) setYVar(columns[1] ?? columns[0]);
  }, [columns, xVar, yVar]);

  useEffect(() => {
    if (didAutoLoad.current) return;
    if (!apiBaseUrl) return;
    if (dataset?.file || dataset?.rows?.length) return;
    didAutoLoad.current = true;
    loadDefaultSample();
  }, [apiBaseUrl, dataset?.file, dataset?.rows?.length]);

  async function loadDefaultSample() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/data/default`, {
        headers: { ...authHeaders(token) },
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.error ?? "Preview failed");
      const rows = json.sample ?? [];
      const cols = json.columns ?? [];
      setData(rows);
      setFiltered(rows);
      setColumns(cols);
      onDatasetChange({ file: null, rows, columns: cols });
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  async function handleLoad(file: File | null, append: boolean) {
    if (!file) return;
    setBusy(true);
    setErr(null);
    try {
      const text = await file.text();
      const parsed = parseCsv(text);
      const nextCols = parsed.columns;
      const rows = parsed.rows;
      if (!append) {
        setColumns(nextCols);
        setData(rows);
        setFiltered(rows);
        setFilters({});
        setQuery("");
        setSearch("");
        onDatasetChange({ file, rows, columns: nextCols });
      } else {
        const mergedCols = Array.from(new Set([...columns, ...nextCols]));
        const normal = rows.map((r) => normalizeRow(r, mergedCols));
        const existing = data.map((r) => normalizeRow(r, mergedCols));
        const merged = [...existing, ...normal];
        setColumns(mergedCols);
        setData(merged);
        setFiltered(merged);
        onDatasetChange({ file: null, rows: merged, columns: mergedCols });
      }
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  function resetFilters() {
    setFiltered(data);
    setFilters({});
    setQuery("");
    setSearch("");
  }

  function applySearch() {
    if (!search.trim()) {
      setFiltered(data);
      return;
    }
    const q = search.toLowerCase();
    const rows = data.filter((row) =>
      columns.some((c) => String(row[c] ?? "").toLowerCase().includes(q))
    );
    setFiltered(rows);
  }

  function applyFilters() {
    let rows = [...data];
    numericCols.forEach((c) => {
      const f = filters[c];
      if (!f) return;
      if (f.min != null && f.min !== "") {
        const v = Number(f.min);
        if (Number.isFinite(v)) rows = rows.filter((r) => Number(r[c]) >= v);
      }
      if (f.max != null && f.max !== "") {
        const v = Number(f.max);
        if (Number.isFinite(v)) rows = rows.filter((r) => Number(r[c]) <= v);
      }
    });
    setFiltered(rows);
  }

  function applyQuery() {
    if (!query.trim()) return;
    try {
      const fn = buildRowQuery(query);
      setFiltered(filtered.filter((row) => fn(row)));
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    }
  }

  function addRow() {
    if (!columns.length) return;
    const row: Record<string, any> = {};
    columns.forEach((c) => {
      const v = addRowValues[c];
      row[c] = v == null || v === "" ? "" : v;
    });
    const withId = ensureRowId(row);
    const next = [...data, withId];
    setData(next);
    setFiltered([...filtered, withId]);
    setAddRowValues({});
    onDatasetChange({ file: dataset.file ?? null, rows: next, columns });
  }

  function updateCell(idx: number, col: string, value: string) {
    const row = filtered[idx];
    if (!row) return;
    const id = row.__id;
    const nextFiltered = filtered.map((r, i) => (i === idx ? { ...r, [col]: value } : r));
    const nextData = data.map((r) => (r.__id === id ? { ...r, [col]: value } : r));
    setFiltered(nextFiltered);
    setData(nextData);
    onDatasetChange({ file: dataset.file ?? null, rows: nextData, columns });
  }

  const summaryText = useMemo(() => buildSummary(filtered, numericCols), [filtered, numericCols]);
  const auditText = useMemo(
    () => buildAudit(filtered, auditInputs),
    [filtered, auditInputs]
  );

  function runCalibration(kind: "ppv" | "air" | "frag") {
    const result = calibrateSiteModel(kind, filtered, calibHpd);
    if (result.error) {
      setErr(result.error);
      return;
    }
    if (result.entry) {
      setCalibLog((prev) => [...prev, result.entry]);
    }
    if (result.modelUpdate) {
      setSiteModel((prev) => ({ ...prev, ...result.modelUpdate }));
    }
  }

  return (
    <div className="card">
      <div className="dataHeader">
        <div className="dataHeaderLeft">
          <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Data Management</div>
        </div>
        <div className="dataHeaderActions">
          <input
            ref={loadInputRef}
            className="fileInput"
            type="file"
            accept=".csv"
            onChange={(e) => handleLoad(e.target.files?.[0] ?? null, false)}
          />
          <button className="btn btnPrimary" onClick={() => loadInputRef.current?.click()} disabled={busy}>
            {busy ? "Loading…" : "Load CSV"}
          </button>
          <input
            ref={appendInputRef}
            className="fileInput"
            type="file"
            accept=".csv"
            onChange={(e) => handleLoad(e.target.files?.[0] ?? null, true)}
          />
          <button className="btn" onClick={() => appendInputRef.current?.click()} disabled={busy}>
            {busy ? "Loading…" : "Append CSV"}
          </button>
          <button className="btn" onClick={resetFilters} disabled={busy}>
            Reset Filters
          </button>
        </div>
        <div className="pill">{status}</div>
      </div>

      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}

      <div className="tabPills">
        {[
          { key: "table", label: "Table" },
          { key: "summary", label: "Summary" },
          { key: "visuals", label: "Visuals" },
          { key: "corr", label: "Correlations" },
          { key: "filters", label: "Filters" },
          { key: "calib", label: "Calibration" },
          { key: "export", label: "Export" },
        ].map((t) => (
          <button
            key={t.key}
            className={`tabPill ${tab === t.key ? "tabPillActive" : ""}`}
            onClick={() => setTab(t.key as any)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "table" && (
        <div style={{ marginTop: 12 }}>
          <div className="tableToolbar">
            <div className="label">Search (contains):</div>
            <input
              className="input tableInput"
              placeholder="Search (contains)"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <button className="btn" onClick={applySearch}>Apply</button>
            <button className="btn" onClick={() => { setSearch(""); setFiltered(data); }}>Clear</button>
          </div>
          <div style={{ marginTop: 12, overflow: "auto", maxHeight: 420 }} className="dataTableWrap">
            <table className="dataTable">
              <thead>
                <tr>
                  {columns.map((c) => (
                    <th key={c} className="dataTableHeader">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((row, idx) => (
                  <tr key={idx}>
                    {columns.map((c) => (
                      <td key={c} className="dataTableCell">
                        {String(row[c] ?? "")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="addRowPanel">
            <div className="label">Add Row</div>
            <div className="addRowGrid">
              {columns.map((c) => (
                <label key={c} className="addRowField">
                  <span className="label">{c}</span>
                  <input
                    className="input"
                    value={addRowValues[c] ?? ""}
                    onChange={(e) => setAddRowValues((prev) => ({ ...prev, [c]: e.target.value }))}
                  />
                </label>
              ))}
            </div>
            <div className="addRowActions">
              <button className="btn btnPrimary" onClick={addRow}>Add Row</button>
            </div>
          </div>
        </div>
      )}

      {tab === "summary" && (
        <div style={{ marginTop: 12 }} className="grid2">
          <div className="card">
            <div className="label">Descriptive Statistics</div>
            <pre style={pre}>{summaryText || "No data loaded."}</pre>
          </div>
          <div className="card">
            <div className="label">Quick Audits (computed columns)</div>
            <div className="grid3" style={{ marginTop: 8 }}>
              <input className="input" placeholder="PPV limit" value={auditInputs.ppv} onChange={(e) => setAuditInputs({ ...auditInputs, ppv: e.target.value })} />
              <input className="input" placeholder="Air limit" value={auditInputs.air} onChange={(e) => setAuditInputs({ ...auditInputs, air: e.target.value })} />
              <input className="input" placeholder="HPD" value={auditInputs.hpd} onChange={(e) => setAuditInputs({ ...auditInputs, hpd: e.target.value })} />
            </div>
            <pre style={{ ...pre, marginTop: 10 }}>{auditText || "No KPIs computed."}</pre>
          </div>
        </div>
      )}

      {tab === "visuals" && (
        <div style={{ marginTop: 12 }} className="grid2">
          <div className="card">
            <div className="sectionTitle">Plot Controls</div>
            <select className="input" value={plotType} onChange={(e) => setPlotType(e.target.value)}>
              {[
                "Scatter",
                "Line",
                "Bar",
                "Histogram",
                "Box",
                "Hexbin",
                "PPV vs Scaled Distance",
                "Airblast Scaling",
                "Fragmentation vs PF",
              ].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
            <div className="grid2" style={{ marginTop: 8 }}>
              <select className="input" value={xVar} onChange={(e) => setXVar(e.target.value)}>
                {columns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
              <select className="input" value={yVar} onChange={(e) => setYVar(e.target.value)}>
                {columns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
              <label className="label"><input type="checkbox" checked={logX} onChange={(e) => setLogX(e.target.checked)} /> Log X</label>
              <label className="label"><input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} /> Log Y</label>
            </div>
          </div>
          <div className="card">
            <div className="sectionTitle">Plot</div>
            <DataPlot
              type={plotType}
              data={filtered}
              x={xVar}
              y={yVar}
              logX={logX}
              logY={logY}
            />
          </div>
        </div>
      )}

      {tab === "corr" && (
        <div style={{ marginTop: 12 }} className="card">
          <div className="sectionTitle">Correlation Heatmap</div>
          <CorrelationHeatmap data={filtered} columns={numericCols} />
        </div>
      )}

      {tab === "filters" && (
        <div style={{ marginTop: 12 }} className="card">
          <div className="label">Filters</div>
          <div className="grid3" style={{ marginTop: 8 }}>
            {numericCols.map((c) => (
              <div key={c}>
                <div className="label">{c}</div>
                <div style={{ display: "flex", gap: 6 }}>
                  <input className="input" placeholder="min" value={filters[c]?.min ?? ""} onChange={(e) => setFilters({ ...filters, [c]: { ...(filters[c] ?? {}), min: e.target.value } })} />
                  <input className="input" placeholder="max" value={filters[c]?.max ?? ""} onChange={(e) => setFilters({ ...filters, [c]: { ...(filters[c] ?? {}), max: e.target.value } })} />
                </div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 10 }}>
            <input className="input" placeholder="Query (use backticks for columns)" value={query} onChange={(e) => setQuery(e.target.value)} />
          </div>
          <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
            <button className="btn btnPrimary" onClick={applyFilters}>Apply Filters</button>
            <button className="btn" onClick={applyQuery}>Apply Query</button>
            <button className="btn" onClick={resetFilters}>Clear Filters</button>
          </div>
        </div>
      )}

      {tab === "calib" && (
        <div style={{ marginTop: 12 }} className="grid2">
          <div className="card">
            <div className="label">Site Calibration</div>
            <input className="input" placeholder="HPD (holes/delay)" value={calibHpd} onChange={(e) => setCalibHpd(e.target.value)} />
            <div style={{ marginTop: 8, display: "grid", gap: 8 }}>
              <button className="btn" onClick={() => runCalibration("ppv")}>Calibrate PPV (K, β)</button>
              <button className="btn" onClick={() => runCalibration("air")}>Calibrate Airblast (K_air, B_air)</button>
              <button className="btn" onClick={() => runCalibration("frag")}>Calibrate Fragmentation (A_kuz, exponent)</button>
              <button className="btn" onClick={() => downloadJson(siteModel, "site_model.json")}>Save Site Model JSON</button>
            </div>
          </div>
          <div className="card">
            <div className="label">Calibration Log</div>
            <pre style={pre}>{calibLog.join("\n") || "No calibration yet."}</pre>
          </div>
        </div>
      )}

      {tab === "export" && (
        <div style={{ marginTop: 12 }} className="card">
          <div className="label">Export</div>
          <button className="btn btnPrimary" onClick={() => downloadCsv(filtered, columns, "filtered.csv")}>
            Export Filtered → CSV
          </button>
          <button className="btn" style={{ marginLeft: 8 }} onClick={() => downloadCsv(filtered, columns, "filtered.xlsx")}>
            Export Filtered → Excel
          </button>
          <div className="subtitle" style={{ marginTop: 10 }}>
            Tip: Use Filters/Query to subset by compliance (e.g., <code>Ground Vibration &lt;= 12.5</code>).
          </div>
        </div>
      )}
    </div>
  );
}

function PredictPanel({
  apiBaseUrl,
  token,
  meta,
  dataset,
}: {
  apiBaseUrl: string;
  token: string;
  meta: any;
  dataset: { file?: File | null; rows: Array<Record<string, any>>; columns: string[] };
}) {
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [inputs, setInputs] = useState<Record<string, number>>({});
  const [ranges, setRanges] = useState<Record<string, { min: number; max: number; median: number }>>({});
  const [empirical, setEmpirical] = useState({
    K_ppv: 1000,
    beta: 1.6,
    K_air: 170,
    B_air: 20,
    A_kuz: 22,
    RWS: 115,
  });
  const [hpdOverride, setHpdOverride] = useState(1);
  const [rrMode, setRrMode] = useState<"manual" | "estimate">("estimate");
  const [rrN, setRrN] = useState(1.8);
  const [rrXov, setRrXov] = useState(500);
  const [tab, setTab] = useState<"outputs" | "rr">("outputs");
  const [wantMl, setWantMl] = useState(true);
  const [thresholds, setThresholds] = useState<Record<string, number>>({});
  const outputs: string[] = meta?.outputs ?? ["Ground Vibration", "Airblast", "Fragmentation"];
  const activeDatasetName = dataset?.file?.name ?? meta?.combined_dataset ?? meta?.default_dataset ?? "(default)";
  const datasetSource = dataset?.file ? "Uploaded dataset" : "Shared combined dataset";

  useEffect(() => {
    if (!meta?.input_labels) return;
    const statsFromDataset = computeInputStatsFromDataset(meta.input_labels, dataset);
    const nextInputs: Record<string, number> = {};
    const nextRanges: Record<string, { min: number; max: number; median: number }> = {};
    for (const k of meta.input_labels) {
      const stat = statsFromDataset[k] ?? meta?.input_stats?.[k] ?? { min: 0, max: 1, median: 0 };
      nextInputs[k] = stat.median ?? 0;
      nextRanges[k] = { min: stat.min ?? 0, max: stat.max ?? 1, median: stat.median ?? 0 };
    }
    setInputs(nextInputs);
    setRanges(nextRanges);
  }, [meta, dataset.rows, dataset.columns]);

  useEffect(() => {
    if (!meta?.empirical_defaults) return;
    setEmpirical(meta.empirical_defaults);
  }, [meta]);

  function useMedians() {
    const next = { ...inputs };
    Object.entries(ranges).forEach(([k, r]) => {
      next[k] = r.median;
    });
    setInputs(next);
  }

  function resetRanges() {
    if (!meta?.input_labels) return;
    const nextRanges: Record<string, { min: number; max: number; median: number }> = {};
    for (const k of meta.input_labels) {
      const stat = meta?.input_stats?.[k] ?? { min: 0, max: 1, median: 0 };
      nextRanges[k] = { min: stat.min ?? 0, max: stat.max ?? 1, median: stat.median ?? 0 };
    }
    setRanges(nextRanges);
  }

  async function run() {
    if (!apiBaseUrl) {
      setOut({ error: "Set VITE_API_BASE_URL in the frontend env." });
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      let res: Response;
      // Only use the upload endpoint when the user explicitly provided a dataset file.
      // (The shared Data Manager "default sample" is a 20-row preview; uploading it prevents ML training.)
      if (dataset?.file) {
        const fd = new FormData();
        fd.append("file", dataset.file);
        fd.append("inputs_json", JSON.stringify(inputs));
        fd.append("hpd_override", String(hpdOverride));
        fd.append("empirical_json", JSON.stringify(empirical));
        fd.append("want_ml", String(wantMl));
        if (rrMode === "manual") fd.append("rr_n", String(rrN));
        if (rrMode) fd.append("rr_mode", rrMode);
        fd.append("rr_x_ov", String(rrXov));
        res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/predict/upload`, {
          method: "POST",
          headers: { ...authHeaders(token) },
          body: fd,
        });
      } else {
        res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/predict`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            ...authHeaders(token),
          },
          body: JSON.stringify({
            inputs,
            hpd_override: hpdOverride,
            empirical,
            want_ml: wantMl,
            rr_n: rrMode === "manual" ? rrN : undefined,
            rr_mode: rrMode,
            rr_x_ov: rrXov,
          }),
        });
      }
      const json = await res.json();
      if (!res.ok || json?.error) {
        throw new Error(json?.detail ?? json?.error ?? `HTTP ${res.status}`);
      }
      setOut({ status: res.status, json });
    } catch (e: any) {
      const msg = String(e?.message ?? e);
      setErr(msg);
      setOut({ error: msg });
    } finally {
      setBusy(false);
    }
  }

  const logText = useMemo(() => {
    if (!out?.json) return "";
    const emp = out.json.empirical ?? {};
    const ml = out.json.ml ?? {};
    const lines: string[] = [];
    outputs.forEach((o) => {
      const mlv = ml?.[o];
      const empv = emp?.[o];
      const mlTxt = Number.isFinite(mlv) ? Number(mlv).toFixed(3) : "NA";
      const empTxt = Number.isFinite(empv) ? Number(empv).toFixed(3) : "NA";
      lines.push(`${o}: ML=${mlTxt} | Emp=${empTxt}`);
    });
    const alerts: string[] = [];
    outputs.forEach((o) => {
      const thr = thresholds[o];
      if (!thr) return;
      const v = Number.isFinite(out?.json?.ml?.[o]) ? out.json.ml[o] : out?.json?.empirical?.[o];
      if (Number.isFinite(v) && v > thr) {
        alerts.push(`${o} exceeds ${thr.toFixed(3)} (=${Number(v).toFixed(3)})`);
      }
    });
    if (alerts.length) {
      lines.push("");
      lines.push("Alerts (checked against ML if present, otherwise empirical):");
      alerts.forEach((a) => lines.push(`• ${a}`));
    }
    return lines.join("\n");
  }, [out, outputs, thresholds]);

  function exportResult() {
    if (!out?.json) return;
    const row: Record<string, any> = { ...inputs };
    outputs.forEach((o) => {
      row[`ML ${o}`] = out.json.ml?.[o] ?? "";
      row[`Empirical ${o}`] = out.json.empirical?.[o] ?? "";
    });
    row["RR n"] = out.json.rr?.n ?? "";
    row["KuzRam Xm (mm)"] = out.json.rr?.xm ?? "";
    row["RR X50 (mm)"] = out.json.rr?.x50 ?? "";
    downloadCsv([row], Object.keys(row), "prediction_result.csv");
  }

  if (!meta?.input_labels?.length) {
    return (
      <div className="card">
        <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Simultaneous Prediction</div>
        <div className="subtitle" style={{ marginTop: 8 }}>
          Loading prediction inputs and dataset statistics...
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "grid", gap: 14 }}>
      {err && <div className="error">{err}</div>}
      <div className="card">
        <div className="dataHeader">
          <div>
            <div style={{ fontSize: 20, fontWeight: 900, letterSpacing: "-0.02em" }}>Prediction Module</div>
            <div className="subtitle">
              Desktop-style simultaneous prediction with ML outputs, empirical baselines and Rosin-Rammler fragmentation.
            </div>
          </div>
          <div className="dataHeaderActions">
            <div className="chip">{datasetSource}</div>
            <div className="chip">{activeDatasetName}</div>
            <div className="chip">{outputs.length} outputs</div>
          </div>
        </div>
      </div>
      <div className="grid2">
        <div className="card">
          <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Inputs</div>
          <div className="subtitle">
            Dataset: {activeDatasetName}
          </div>

          <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
            <button className="btn" onClick={useMedians}>Use Medians</button>
            <button className="btn" onClick={resetRanges}>Reset Ranges</button>
          </div>

          <div style={{ marginTop: 10, display: "grid", gap: 10, maxHeight: 360, overflow: "auto" }}>
            {(meta?.input_labels ?? []).map((k: string) => {
              const r = ranges[k] ?? { min: 0, max: 1, median: 0 };
              const value = inputs[k] ?? r.median ?? 0;
              return (
                <div key={k}>
                  <label className="label">{k}</label>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 90px", gap: 8 }}>
                    <input
                      className="input"
                      type="range"
                      min={r.min}
                      max={r.max}
                      step={(r.max - r.min) / 200 || 0.01}
                      value={value}
                      onChange={(e) => setInputs({ ...inputs, [k]: Number(e.target.value) })}
                    />
                    <input
                      className="input"
                      type="number"
                      value={value}
                      onChange={(e) => setInputs({ ...inputs, [k]: Number(e.target.value) })}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="card" style={{ marginTop: 12 }}>
            <div className="label">Empirical settings (USBM & Kuz–Ram)</div>
            <div className="grid3" style={{ marginTop: 8 }}>
              {Object.entries(empirical).map(([k, v]) => (
                <div key={k}>
                  <label className="label">{k}</label>
                  <input className="input" type="number" value={v} onChange={(e) => setEmpirical({ ...empirical, [k]: Number(e.target.value) })} />
                </div>
              ))}
            </div>
            <div className="grid3" style={{ marginTop: 8 }}>
              <div>
                <label className="label">HPD (holes/delay)</label>
                <input className="input" type="number" value={hpdOverride} onChange={(e) => setHpdOverride(Number(e.target.value))} />
              </div>
              <div>
                <label className="label">Oversize threshold (mm)</label>
                <input className="input" type="number" value={rrXov} onChange={(e) => setRrXov(Number(e.target.value))} />
              </div>
              <div>
                <label className="label">RR n mode</label>
                <select className="input" value={rrMode} onChange={(e) => setRrMode(e.target.value as any)}>
                  <option value="estimate">Estimate (Kuz–Ram)</option>
                  <option value="manual">Manual (n)</option>
                </select>
              </div>
            </div>
            <div className="grid3" style={{ marginTop: 8 }}>
              <div>
                <label className="label">Manual n</label>
                <input className="input" type="number" value={rrN} onChange={(e) => setRrN(Number(e.target.value))} />
              </div>
              <div>
                <label className="label">Use ML</label>
                <select className="input" value={wantMl ? "yes" : "no"} onChange={(e) => setWantMl(e.target.value === "yes")}>
                  <option value="yes">Yes</option>
                  <option value="no">No (empirical only)</option>
                </select>
              </div>
              <div>
                <label className="label">n display</label>
                <div className="pill">
                  {rrMode === "manual" ? `n (manual): ${rrN.toFixed(3)}` : `n (estimated): ${out?.json?.rr?.n?.toFixed?.(3) ?? "—"}`}
                </div>
              </div>
            </div>
            <div className="subtitle" style={{ marginTop: 8 }}>
              PPV = K_ppv·(R/√Qd)^-β; Air = K_air + B_air·log10(Qd^(1/3)/R).
            </div>
          </div>
        </div>

        <div className="card">
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
            <div>
              <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Simultaneous Prediction</div>
              <div className="subtitle">Predict (ML + Empirical) with RR curve.</div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn btnPrimary" onClick={run} disabled={busy}>
                {busy ? "Running…" : "Predict (ML + Empirical)"}
              </button>
              <button className="btn" onClick={exportResult} disabled={!out?.json}>
                Export Result CSV
              </button>
            </div>
          </div>

          <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
            <button className={`btn ${tab === "outputs" ? "btnPrimary" : ""}`} onClick={() => setTab("outputs")}>
              Outputs (Empirical vs ML)
            </button>
            <button className={`btn ${tab === "rr" ? "btnPrimary" : ""}`} onClick={() => setTab("rr")}>
              Fragmentation (RR Curve)
            </button>
          </div>

          {tab === "outputs" && (
            <div style={{ marginTop: 12 }}>
              <BarCompareChart outputs={outputs} empirical={out?.json?.empirical} ml={out?.json?.ml} thresholds={thresholds} />
            </div>
          )}
          {tab === "rr" && (
            <div style={{ marginTop: 12 }}>
              {out?.json?.rr ? <RRChart rr={out.json.rr} /> : <div className="subtitle">Run prediction to see RR curve.</div>}
            </div>
          )}

          <div className="card" style={{ marginTop: 12 }}>
            <div className="label">Thresholds & Alerts</div>
            <div className="grid3" style={{ marginTop: 8 }}>
              {outputs.map((o) => (
                <div key={o}>
                  <label className="label">{o}</label>
                  <input
                    className="input"
                    type="number"
                    value={thresholds[o] ?? ""}
                    onChange={(e) => setThresholds({ ...thresholds, [o]: Number(e.target.value) })}
                  />
                </div>
              ))}
            </div>
            <pre style={{ ...pre, marginTop: 10 }}>{logText || "Run predictions to populate log."}</pre>
          </div>
        </div>
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
  const [xAxis, setXAxis] = useState("");
  const [yAxis, setYAxis] = useState("");
  const [surface, setSurface] = useState<any>(null);

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
      if (json?.feature_stats && !Object.keys(inputs).length) {
        const next: Record<string, number> = {};
        Object.keys(json.feature_stats).forEach((k) => {
          next[k] = json.feature_stats[k].median;
        });
        setInputs(next);
      }
      if (json?.features?.length) {
        setXAxis(json.features[0]);
        setYAxis(json.features[1] ?? json.features[0]);
      }
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  async function runSurface(xName?: string, yName?: string) {
    if (!apiBaseUrl) return;
    const x = xName ?? xAxis;
    const y = yName ?? yAxis;
    if (!x || !y || x === y) return;
    try {
      let res: Response;
      if (file) {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("payload_json", JSON.stringify({ x_name: x, y_name: y, inputs_json: inputs }));
        res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/flyrock/surface/upload`, {
          method: "POST",
          headers: { ...authHeaders(token) },
          body: fd,
        });
      } else {
        res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/flyrock/surface`, {
          method: "POST",
          headers: { "content-type": "application/json", ...authHeaders(token) },
          body: JSON.stringify({ x_name: x, y_name: y, inputs_json: inputs }),
        });
      }
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Surface failed");
      setSurface(json);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    }
  }

  const empiricalAuto = useMemo(() => {
    if (resp?.empirical_auto != null) {
      return { value: resp.empirical_auto, method: resp.empirical_method };
    }
    const emp = resp?.empirical ?? {};
    const order = [
      ["Lundborg_1981", "Lundborg (1981): 143*d_in*(q-0.2)"],
      ["McKenzie_SDoB", "McKenzie/SDoB: 10*d_mm^0.667*SDoB^-2.167*(ρ/2.6)"],
      ["Lundborg_Legacy", "Legacy d-only: 30.745*d_mm^0.66"],
    ];
    for (const [k, method] of order) {
      const v = emp?.[k];
      if (Number.isFinite(Number(v))) return { value: Number(v), method };
    }
    return null;
  }, [resp]);

  useEffect(() => {
    if (xAxis && yAxis) runSurface(xAxis, yAxis);
  }, [xAxis, yAxis]);

  useEffect(() => {
    if (!resp?.feature_stats || !Object.keys(inputs).length) return;
    const t = window.setTimeout(() => {
      run();
      if (xAxis && yAxis) runSurface(xAxis, yAxis);
    }, 350);
    return () => window.clearTimeout(t);
  }, [inputs]);

  return (
    <div className="card">
      <div className="grid2">
        <div>
          <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Flyrock — ML + Empirical</div>
          <div className="subtitle">Load a CSV to train and explore the surface.</div>

          <div style={{ marginTop: 10 }}>
            <label className="label">Load CSV</label>
            <input className="input" type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </div>

          <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
            <button className="btn btnPrimary" onClick={run} disabled={busy}>{busy ? "Running…" : "Predict"}</button>
            <button className="btn" onClick={() => runSurface()} disabled={!resp?.features?.length}>Redraw surface</button>
          </div>

          <div className="kpi" style={{ marginTop: 12 }}>
            <div className="kpiTitle">Predicted flyrock</div>
            <div className="kpiValue">{resp?.prediction != null ? formatNum(resp.prediction) : "—"}</div>
            {resp?.train_r2 != null && <div className="label">Train R²: {Number(resp.train_r2).toFixed(3)}</div>}
            {resp?.test_r2 != null && <div className="label">Test R²: {Number(resp.test_r2).toFixed(3)}</div>}
          </div>

          <div className="kpi" style={{ marginTop: 10 }}>
            <div className="kpiTitle">Empirical estimate</div>
            <div className="kpiValue">{empiricalAuto ? formatNum(empiricalAuto.value) : "—"} m</div>
            {empiricalAuto?.method && <div className="label">{empiricalAuto.method}</div>}
          </div>

          {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}

          {resp?.feature_importance?.length ? (
            <div className="card" style={{ marginTop: 12 }}>
              <div className="label">Model feature importance</div>
              <HorizontalBarChart
                labels={resp.feature_importance.map((it: any) => it.feature)}
                values={resp.feature_importance.map((it: any) => Number(it.importance))}
              />
            </div>
          ) : null}

          {resp?.feature_stats && (
            <div style={{ marginTop: 12, maxHeight: 420, overflow: "auto" }}>
              <div className="label">Adjust Inputs</div>
              {Object.entries(resp.feature_stats).map(([k, stat]: any) => (
                <div key={k} style={{ marginTop: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span className="label">{k}</span>
                    <span className="label">{formatNum(inputs[k] ?? stat?.median ?? 0)}</span>
                  </div>
                  <input
                    className="input"
                    type="range"
                    min={stat?.min ?? 0}
                    max={stat?.max ?? 1}
                    step={(stat?.max - stat?.min) / 200 || 0.01}
                    value={inputs[k] ?? stat?.median ?? 0}
                    onChange={(e) => setInputs({ ...inputs, [k]: Number(e.target.value) })}
                  />
                </div>
              ))}
            </div>
          )}
        </div>

        <div>
          <div className="card">
            <div className="label">Surface axes</div>
            <div className="grid2" style={{ marginTop: 8 }}>
              <select className="input" value={xAxis} onChange={(e) => setXAxis(e.target.value)}>
                {(resp?.features ?? []).map((f: string) => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
              <select className="input" value={yAxis} onChange={(e) => setYAxis(e.target.value)}>
                {(resp?.features ?? []).map((f: string) => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
            </div>
            {surface?.Z ? (
              <SurfaceIsoPlot
                gridX={surface.grid_x}
                gridY={surface.grid_y}
                Z={surface.Z}
                xLabel={surface?.x_name ?? xAxis}
                yLabel={surface?.y_name ?? yAxis}
                zLabel="Predicted flyrock"
              />
            ) : (
              <div className="subtitle" style={{ marginTop: 10 }}>
                Run prediction to enable the flyrock surface.
              </div>
            )}
            <div className="subtitle" style={{ marginTop: 8 }}>
              Empirical: auto-chooses Lundborg (1981), McKenzie/SDoB, or legacy d-only.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SlopePanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [resp, setResp] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [params, setParams] = useState({
    H: 10,
    beta: 30,
    c: 50,
    phi: 30,
    gamma: 20,
    ru: 0.2,
    B: 4,
  });
  const modelReadyRef = useRef(false);
  const seededFromDataRef = useRef(false);
  const backendUnavailableRef = useRef(false);
  const fileKey = file?.name ?? "__default__";
  const probStable = useMemo(() => {
    const direct = toNum(resp?.prob_stable ?? resp?.prediction_probability ?? resp?.probability ?? resp?.prediction);
    if (Number.isFinite(direct)) {
      if (direct >= 0 && direct <= 1) return direct;
      if (direct > 1 && direct <= 100) return direct / 100;
    }
    const label = String(resp?.predicted_class ?? resp?.label ?? "").trim().toLowerCase();
    if (label === "stable") return 1;
    if (label === "failure" || label === "failed" || label === "unstable") return 0;
    return null;
  }, [resp]);

  function buildInputsFromParams() {
    return {
      H_m: params.H,
      beta_deg: params.beta,
      c_kPa: params.c,
      phi_deg: params.phi,
      gamma_kN_m3: params.gamma,
      ru: params.ru,
    };
  }

  function localSlopeEstimate() {
    const H = Math.max(0.1, Number(params.H));
    const betaDeg = Math.min(89, Math.max(1, Number(params.beta)));
    const phiDeg = Math.min(89, Math.max(0.1, Number(params.phi)));
    const c = Math.max(0, Number(params.c));
    const gamma = Math.max(1e-3, Number(params.gamma));
    const ru = Math.min(1, Math.max(0, Number(params.ru)));
    const betaRad = (betaDeg * Math.PI) / 180;
    const phiRad = (phiDeg * Math.PI) / 180;
    const sigma = gamma * H * Math.cos(betaRad) ** 2;
    const tau = Math.max(1e-6, gamma * H * Math.sin(betaRad) * Math.cos(betaRad));
    const sigmaEff = sigma * (1 - ru);
    const shearResistance = c + Math.max(0, sigmaEff) * Math.tan(phiRad);
    const fs = shearResistance / tau;
    const prob = Math.min(1, Math.max(0, 1 / (1 + Math.exp(-4 * (fs - 1)))));
    return {
      prob_stable: prob,
      prediction: prob,
      predicted_class: prob >= 0.5 ? "stable" : "failure",
      mode: "local_fallback",
      factor_of_safety: fs,
      feature_stats: {
        H_m: { min: 1, max: 50, median: H },
        beta_deg: { min: 5, max: 80, median: betaDeg },
        c_kPa: { min: 1, max: 200, median: c },
        phi_deg: { min: 5, max: 60, median: phiDeg },
        gamma_kN_m3: { min: 14, max: 28, median: gamma },
        ru: { min: 0, max: 1, median: ru },
      },
    };
  }

  function slopeUrls(): string[] {
    const base = apiBaseUrl.replace(/\/$/, "");
    const sameOrigin = window.location.origin.replace(/\/$/, "");
    if (!base) return [];
    // If a backend URL is explicitly configured, trust it and avoid same-origin fallbacks
    // that can produce hosting-layer NOT_FOUND pages.
    if (base !== sameOrigin) return [`${base}/v1/slope/predict`];
    return [`${base}/v1/slope/predict`, `${base}/api/v1/slope/predict`];
  }

  async function run(options?: { preserveSliders?: boolean }) {
    if (!apiBaseUrl) {
      setResp(localSlopeEstimate());
      setErr("Backend URL is not configured. Showing local slope estimate.");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      if (options?.preserveSliders) {
        const local = localSlopeEstimate();
        setResp((prev: any) => ({ ...(prev ?? {}), ...local }));
        return;
      }
      const fd = new FormData();
      if (file) fd.append("file", file);
      fd.append("inputs_json", JSON.stringify(buildInputsFromParams()));
      const urls = slopeUrls();
      let lastErr: any = null;
      let gotBackendResponse = false;
      const attemptErrors: string[] = [];

      for (const url of urls) {
        try {
          const controller = new AbortController();
          const timeoutId = window.setTimeout(() => controller.abort(), 120000);
          let res: Response;
          try {
            res = await fetch(url, {
              method: "POST",
              headers: { ...authHeaders(token) },
              body: fd,
              signal: controller.signal,
            });
          } finally {
            window.clearTimeout(timeoutId);
          }
          const raw = await res.text();
          let json: any = {};
          try {
            json = raw ? JSON.parse(raw) : {};
          } catch {
            json = {};
          }
          const detail =
            json?.error ??
            json?.detail ??
            (raw && !raw.trim().startsWith("<") ? raw.trim() : "") ??
            "";
          if (!res.ok || json?.error || json?.detail) {
            throw new Error(detail || `Slope request failed (${res.status})`);
          }

          gotBackendResponse = true;
          backendUnavailableRef.current = false;
          setResp(json);
          if (json?.feature_stats && !options?.preserveSliders && !seededFromDataRef.current) {
            const H = Number(json.feature_stats?.H_m?.median ?? params.H);
            setParams({
              H,
              beta: Number(json.feature_stats?.beta_deg?.median ?? params.beta),
              c: Number(json.feature_stats?.c_kPa?.median ?? params.c),
              phi: Number(json.feature_stats?.phi_deg?.median ?? params.phi),
              gamma: Number(json.feature_stats?.gamma_kN_m3?.median ?? params.gamma),
              ru: Number(json.feature_stats?.ru?.median ?? params.ru),
              B: Math.max(0, 0.4 * H),
            });
            seededFromDataRef.current = true;
          }
          break;
        } catch (attemptErr: any) {
          lastErr = attemptErr;
          const msg = String(attemptErr?.message ?? attemptErr ?? "Unknown error");
          attemptErrors.push(`${url} -> ${msg}`);
        }
      }

      if (!gotBackendResponse) {
        backendUnavailableRef.current = true;
        setResp(localSlopeEstimate());
        // Keep fallback silent in the UI; details remain available in console for debugging.
        if (attemptErrors.length) {
          console.warn("Slope backend prediction failed; using local estimate", attemptErrors);
        }
        setErr(null);
        modelReadyRef.current = true;
        return;
      }

      modelReadyRef.current = true;
    } catch (e: any) {
      setResp(null);
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    modelReadyRef.current = false;
    seededFromDataRef.current = false;
    backendUnavailableRef.current = false;
    setResp(null);
    setErr(null);
  }, [fileKey]);

  useEffect(() => {
    if (!modelReadyRef.current || busy) return;
    const id = window.setTimeout(() => {
      void run({ preserveSliders: true });
    }, 250);
    return () => window.clearTimeout(id);
  }, [params]);
  return (
    <div className="card">
      <div className="grid2">
        <div>
          <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Slope Stability — Stable / Failure (ML)</div>

          <div style={{ marginTop: 14, fontWeight: 700, fontSize: 15 }}>Data</div>
          <div style={{ marginTop: 6 }}>
            <label className="label">Load CSV</label>
            <input className="input" type="file" accept=".csv,.xlsx,.xls" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </div>

          <div style={{ marginTop: 8 }}>
            <button className="btn btnPrimary" onClick={() => run()} disabled={busy}>
              {busy ? "Running..." : "Load & Predict"}
            </button>
          </div>

          <div style={{ marginTop: 14, fontWeight: 700, fontSize: 15 }}>Parameters</div>
          <div style={{ marginTop: 12, display: "grid", gap: 10 }}>
            <SliderField label="H (m)" value={params.H} min={1} max={50} step={0.5} onChange={(v) => setParams({ ...params, H: v })} />
            <SliderField label="β (deg)" value={params.beta} min={5} max={80} step={0.5} onChange={(v) => setParams({ ...params, beta: v })} />
            <SliderField label="c (kPa)" value={params.c} min={1} max={200} step={0.5} onChange={(v) => setParams({ ...params, c: v })} />
            <SliderField label="φ (deg)" value={params.phi} min={5} max={60} step={0.5} onChange={(v) => setParams({ ...params, phi: v })} />
            <SliderField label="γ (kN/m³)" value={params.gamma} min={14} max={28} step={0.1} onChange={(v) => setParams({ ...params, gamma: v })} />
            <SliderField label="ru (–)" value={params.ru} min={0} max={1} step={0.01} onChange={(v) => setParams({ ...params, ru: v })} />
            <SliderField label="B (m) — sketch only" value={params.B} min={0} max={30} step={0.5} onChange={(v) => setParams({ ...params, B: v })} />
          </div>

          <div className="kpi" style={{ marginTop: 12 }}>
            <div className="kpiTitle">Prediction</div>
            <div className="kpiValue" style={{ fontSize: 20 }}>
              {probStable != null
                ? `${probStable >= 0.5 ? "Stable" : "Failure"} (${(probStable * 100).toFixed(1)}%)`
                : "—"}
            </div>
          </div>
          {err ? <div className="error" style={{ marginTop: 10 }}>{err}</div> : null}
          {probStable != null && (
            <>
              <div className="grid3" style={{ marginTop: 10 }}>
                <div className="kpi">
                  <div className="kpiTitle">Train accuracy</div>
                  <div className="kpiValue">{formatNum(resp.train_accuracy)}</div>
                </div>
                <div className="kpi">
                  <div className="kpiTitle">Test accuracy</div>
                  <div className="kpiValue">{formatNum(resp.test_accuracy)}</div>
                </div>
                <div className="kpi">
                  <div className="kpiTitle">Class balance</div>
                  <div className="kpiValue" style={{ fontSize: 14 }}>
                    S {resp.class_balance?.stable ?? 0} / F {resp.class_balance?.failure ?? 0}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        <div className="card">
          <SlopeSketch
            H={params.H}
            beta={params.beta}
            B={params.B}
            prob={probStable ?? undefined}
          />
        </div>
      </div>
    </div>
  );
}

function DelayPanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [resp, setResp] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [colorBy, setColorBy] = useState("Delay");
  const [sizeBy, setSizeBy] = useState("Charge");
  const [playing, setPlaying] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [showLabels, setShowLabels] = useState(true);
  const [showShock, setShowShock] = useState(true);
  const [selectedPoint, setSelectedPoint] = useState<Record<string, any> | null>(null);

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
      const json = await res.json().catch(() => ({}));
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Delay failed");
      setResp(json);
      if (json?.points?.length) {
        const numericKeys = Object.keys(json.points[0] ?? {}).filter((k) =>
          json.points.some((p: any) => Number.isFinite(Number(p?.[k])))
        );
        if (numericKeys.includes("Delay")) {
          setColorBy("Delay");
          setSizeBy(numericKeys.includes("Charge") ? "Charge" : "Delay");
          setStepIndex(0);
          setSelectedPoint(null);
          setPlaying(false);
        }
      }
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  const points = useMemo(
    () => (resp?.points ?? []).map((p: Record<string, any>, idx: number) => ({ ...p, __idx: idx })),
    [resp]
  );
  const numericFields = useMemo(() => {
    if (!points.length) return [] as string[];
    return Object.keys(points[0]).filter((k) => k !== "__idx" && points.some((p) => Number.isFinite(Number(p?.[k]))));
  }, [points]);
  const uniqueTimes = useMemo(() => {
    const vals = points.map((p) => Number(p.Delay)).filter((v) => Number.isFinite(v));
    return Array.from(new Set(vals)).sort((a, b) => a - b);
  }, [points]);
  const currentTime = uniqueTimes.length ? uniqueTimes[Math.max(0, Math.min(stepIndex, uniqueTimes.length - 1))] : undefined;
  const previousTime = uniqueTimes.length && stepIndex > 0 ? uniqueTimes[stepIndex - 1] : undefined;
  const delayStats = useMemo(() => {
    if (!uniqueTimes.length) return null;
    const delays = points.map((p) => Number(p.Delay)).filter((v) => Number.isFinite(v));
    const counts = new Map<number, number>();
    delays.forEach((d) => counts.set(d, (counts.get(d) ?? 0) + 1));
    const groupSizes = Array.from(counts.values()).sort((a, b) => a - b);
    const span = Math.max(...delays) - Math.min(...delays);
    const medianGroup = groupSizes[Math.floor(groupSizes.length / 2)] ?? 0;
    return {
      min: Math.min(...delays),
      max: Math.max(...delays),
      span,
      unique: uniqueTimes.length,
      medianGroup,
    };
  }, [points, uniqueTimes]);
  const currentGroup = useMemo(() => {
    if (currentTime == null) return [] as Array<Record<string, any>>;
    return points.filter((p) => Math.abs(Number(p.Delay) - currentTime) < 1e-6);
  }, [points, currentTime]);
  const selectedActualGap =
    selectedPoint && Number.isFinite(Number(selectedPoint.ActualDelay))
      ? Number(selectedPoint.Delay) - Number(selectedPoint.ActualDelay)
      : null;

  useEffect(() => {
    if (!playing || !uniqueTimes.length) return;
    const id = window.setInterval(() => {
      setStepIndex((prev) => (prev >= uniqueTimes.length - 1 ? 0 : prev + 1));
    }, Math.max(120, 700 / Math.max(0.5, speed)));
    return () => window.clearInterval(id);
  }, [playing, speed, uniqueTimes]);

  useEffect(() => {
    if (!numericFields.length) return;
    if (!numericFields.includes(colorBy)) {
      setColorBy(numericFields.includes("Delay") ? "Delay" : numericFields[0]);
    }
    if (!numericFields.includes(sizeBy)) {
      setSizeBy(numericFields.includes("Charge") ? "Charge" : numericFields[0]);
    }
  }, [numericFields, colorBy, sizeBy]);

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Delay Prediction & Blast Simulation</div>
      <div className="subtitle">
        Blast-sequence view with equal-aspect plan geometry, delay-labelled holes, firing-step playback, and sequence analysis.
      </div>
      <div style={{ marginTop: 10 }} className="grid2">
        <div>
          <label className="label">Upload CSV / XLSX</label>
          <input className="input" type="file" accept=".csv,.xlsx,.xls" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </div>
        <div style={{ display: "flex", alignItems: "flex-end" }}>
          <button className="btn btnPrimary" onClick={run} disabled={busy}>{busy ? "Running…" : "Predict Delays"}</button>
        </div>
      </div>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      {resp?.points?.length ? (
        <div className="grid3" style={{ marginTop: 10 }}>
          <div className="kpi">
            <div className="kpiTitle">Predicted holes</div>
            <div className="kpiValue">{resp.points.length}</div>
          </div>
          <div className="kpi">
            <div className="kpiTitle">Training rows</div>
            <div className="kpiValue">{resp.training_rows ?? "—"}</div>
          </div>
          <div className="kpi">
            <div className="kpiTitle">Train R²</div>
            <div className="kpiValue">{formatNum(resp.train_r2)}</div>
          </div>
          <div className="kpi">
            <div className="kpiTitle">Test R²</div>
            <div className="kpiValue">{formatNum(resp.test_r2)}</div>
          </div>
          <div className="kpi">
            <div className="kpiTitle">Firing steps</div>
            <div className="kpiValue">{delayStats?.unique ?? "—"}</div>
          </div>
          <div className="kpi">
            <div className="kpiTitle">Delay span</div>
            <div className="kpiValue">{delayStats ? `${formatNum(delayStats.span)} ms` : "—"}</div>
          </div>
          <div className="kpi">
            <div className="kpiTitle">Median holes/step</div>
            <div className="kpiValue">{delayStats?.medianGroup ?? "—"}</div>
          </div>
        </div>
      ) : null}
      {resp?.points?.length ? (
        <div style={{ marginTop: 12 }}>
          <div className="card">
            <div className="sectionTitle">Simulation Controls</div>
            <div className="subtitle">
              Dataset: {resp.dataset_used ?? "default"} · target source: {resp.target_source ?? "observed_delay"} · features: {(resp.features_used ?? []).join(", ")}
            </div>
            <div className="grid3" style={{ marginTop: 10 }}>
              <div>
                <label className="label">Color by</label>
                <select className="input" value={colorBy} onChange={(e) => setColorBy(e.target.value)}>
                  {numericFields.map((k) => (
                    <option key={k} value={k}>{k}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="label">Size by</label>
                <select className="input" value={sizeBy} onChange={(e) => setSizeBy(e.target.value)}>
                  {numericFields.map((k) => (
                    <option key={k} value={k}>{k}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="label">Playback speed</label>
                <input className="input" type="range" min="0.5" max="5" step="0.25" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} />
                <div className="subtitle">{formatNum(speed)}x</div>
              </div>
            </div>
            <div className="grid3" style={{ marginTop: 10 }}>
              <div>
                <label className="label">Simulation step</label>
                <input
                  className="input"
                  type="range"
                  min={0}
                  max={Math.max(0, uniqueTimes.length - 1)}
                  step="1"
                  value={Math.max(0, Math.min(stepIndex, Math.max(0, uniqueTimes.length - 1)))}
                  onChange={(e) => setStepIndex(Number(e.target.value))}
                />
                <div className="subtitle">
                  {currentTime == null ? "No delay sequence" : `Step ${stepIndex + 1}/${uniqueTimes.length} · t = ${formatNum(currentTime)} ms`}
                </div>
              </div>
              <div>
                <label className="label">Playback</label>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 8 }}>
                  <button className="btn btnPrimary" onClick={() => setPlaying(true)} disabled={playing}>Play</button>
                  <button className="btn" onClick={() => setPlaying(false)}>Pause</button>
                  <button className="btn" onClick={() => { setPlaying(false); setStepIndex(0); }}>Reset</button>
                </div>
              </div>
              <div>
                <label className="label">Visual overlays</label>
                <div style={{ display: "grid", gap: 6, marginTop: 8 }}>
                  <label className="label"><input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} /> Show delay labels</label>
                  <label className="label"><input type="checkbox" checked={showShock} onChange={(e) => setShowShock(e.target.checked)} /> Show shock-front rings</label>
                </div>
              </div>
            </div>
          </div>

          <PlanView
            points={points}
            colorBy={colorBy}
            sizeBy={sizeBy}
            currentTime={currentTime}
            previousTime={previousTime}
            showLabels={showLabels}
            showShock={showShock}
            selectedPoint={selectedPoint}
            onSelect={setSelectedPoint}
          />

          <div className="grid2" style={{ marginTop: 12 }}>
            <div style={{ display: "grid", gap: 12 }}>
              <DelaySequenceChart points={points} currentTime={currentTime} />
              <div className="card">
                <div className="sectionTitle">Current Firing Step</div>
                <div className="subtitle">
                  {currentTime == null
                    ? "No active firing step selected."
                    : `${currentGroup.length} hole(s) firing at ${formatNum(currentTime)} ms.`}
                </div>
                <div className="grid3" style={{ marginTop: 10 }}>
                  {currentGroup.slice(0, 9).map((point) => (
                    <div key={point.__idx} className="kpi" onClick={() => setSelectedPoint(point)} style={{ cursor: "pointer" }}>
                      <div className="kpiTitle">{point.HoleID ? `Hole ${point.HoleID}` : `Hole ${point.__idx + 1}`}</div>
                      <div className="kpiValue">{Math.round(Number(point.Delay))} ms</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="card">
                <div className="sectionTitle">Selected Hole</div>
                {selectedPoint ? (
                  <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
                    <div className="subtitle">
                      {selectedPoint.HoleID ? `Hole ${selectedPoint.HoleID}` : `Hole ${selectedPoint.__idx + 1}`} · X {formatNum(selectedPoint.X)} · Y {formatNum(selectedPoint.Y)}
                    </div>
                    <div className="grid2">
                      <div className="kpi">
                        <div className="kpiTitle">Predicted delay</div>
                        <div className="kpiValue">{formatNum(selectedPoint.Delay)} ms</div>
                      </div>
                      <div className="kpi">
                        <div className="kpiTitle">Actual delay</div>
                        <div className="kpiValue">{Number.isFinite(Number(selectedPoint.ActualDelay)) ? `${formatNum(selectedPoint.ActualDelay)} ms` : "—"}</div>
                      </div>
                      <div className="kpi">
                        <div className="kpiTitle">Charge</div>
                        <div className="kpiValue">{formatNum(selectedPoint.Charge)}</div>
                      </div>
                      <div className="kpi">
                        <div className="kpiTitle">Depth</div>
                        <div className="kpiValue">{formatNum(selectedPoint.Depth)}</div>
                      </div>
                    </div>
                    {selectedActualGap != null ? (
                      <div className="subtitle">Prediction gap vs actual: {formatNum(selectedActualGap)} ms</div>
                    ) : null}
                  </div>
                ) : (
                  <div className="subtitle" style={{ marginTop: 8 }}>Click a hole in the plan view to inspect its timing and geometry.</div>
                )}
              </div>
            </div>
          </div>

          <div style={{ marginTop: 10 }}>
            <button className="btn" onClick={() => downloadCsv(points, Object.keys(points[0] ?? {}).filter((k) => k !== "__idx"), "delay_predictions.csv")}>
              Export CSV
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function FeaturePanel({
  apiBaseUrl,
  token,
  dataset,
  activeDatasetName,
}: {
  apiBaseUrl: string;
  token: string;
  dataset: { file?: File | null; rows: Array<Record<string, any>>; columns: string[] };
  activeDatasetName: string;
}) {
  const [resp, setResp] = useState<any>(null);
  const [pca, setPca] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [topK, setTopK] = useState(12);
  const [selectedOutput, setSelectedOutput] = useState("");
  const [msg, setMsg] = useState("Tip: Load/confirm dataset in Data Manager. Inputs = first N−3 if names can't be mapped.");
  const resolvedDatasetName = dataset?.file?.name ?? activeDatasetName ?? "(default)";

  useEffect(() => {
    setErr(null);
    setResp(null);
    setPca(null);
    setSelectedOutput("");
    setMsg(
      `Active dataset: ${resolvedDatasetName}\nRun RF importance, explainability, or PCA to analyse this dataset.`
    );
  }, [resolvedDatasetName]);

  async function runImportance() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      let res: Response;
      if (dataset?.file) {
        const fd = new FormData();
        fd.append("file", dataset.file);
        fd.append("top_k", String(topK));
        res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/feature/importance`, {
          method: "POST",
          headers: { ...authHeaders(token) },
          body: fd,
        });
      } else {
        res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/feature/importance?top_k=${topK}`, {
          headers: { ...authHeaders(token) },
        });
      }
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Failed");
      setResp(json);
      if (json?.outputs?.length) {
        setSelectedOutput((prev) => (json.outputs.includes(prev) ? prev : json.outputs[0]));
      }
      setMsg(
        `Dataset: ${json?.dataset ?? resolvedDatasetName}\n${json?.note ?? ""}\nRows used: ${json?.rows_used ?? "?"} | Inputs: ${json?.inputs?.length ?? "?"} | Outputs: ${json?.outputs?.length ?? "?"}\nPlotted top-${json?.top_k ?? topK} features for each output.`
      );
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  async function runPca() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      let p: Response;
      if (dataset?.file) {
        const fd = new FormData();
        fd.append("file", dataset.file);
        p = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/feature/pca`, {
          method: "POST",
          headers: { ...authHeaders(token) },
          body: fd,
        });
      } else {
        p = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/feature/pca`, {
          headers: { ...authHeaders(token) },
        });
      }
      const json = await p.json();
      if (!p.ok || json?.error) throw new Error(json?.error ?? "Failed");
      setPca(json);
      setMsg(
        `PCA based on dataset: ${json?.dataset ?? resolvedDatasetName}\n${json?.note ?? ""}\nShape used: ${json?.rows_used ?? "?"} rows × ${json?.inputs?.length ?? "?"} inputs`
      );
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div className="card">
        <div className="dataHeader">
          <div>
            <div style={{ fontSize: 20, fontWeight: 900, letterSpacing: "-0.02em" }}>Feature Importance & Explainable AI</div>
            <div className="subtitle">
              Shared-dataset analysis for Random Forest importance, permutation sensitivity, SHAP-style local impacts and PCA structure.
            </div>
          </div>
          <div className="dataHeaderActions">
            <div className="chip">{dataset?.file ? "Uploaded dataset" : "Shared combined dataset"}</div>
            <div className="chip">{resolvedDatasetName}</div>
            <div className="chip">Top-K {topK}</div>
          </div>
        </div>
      </div>

      <div className="card">
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <button className="btn btnPrimary" onClick={runImportance} disabled={busy}>
            {busy ? "Loading..." : "Compute RF Importance"}
        </button>
        <button className="btn" onClick={runPca} disabled={busy}>
            {busy ? "Loading..." : "Run PCA Analysis"}
        </button>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className="label">Top-K features</span>
          <input
            className="input"
            type="number"
            min={5}
            max={30}
            value={topK}
              onChange={(e) => setTopK(Math.max(5, Math.min(30, Number(e.target.value) || 5)))}
            style={{ width: 80 }}
          />
        </div>
      </div>
        {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
        <pre style={{ ...pre, marginTop: 10 }}>{msg}</pre>
      </div>

      {resp?.diagnostics && (
        <div className="grid3">
          <div className="kpi">
            <div className="kpiTitle">Mapping mode</div>
            <div className="kpiValue" style={{ fontSize: 16 }}>
              {resp.diagnostics.mapping_mode === "names" ? "Name mapped" : "Positional split"}
            </div>
          </div>
          <div className="kpi">
            <div className="kpiTitle">Rows used</div>
            <div className="kpiValue">{resp.diagnostics.rows_used}</div>
          </div>
          <div className="kpi">
            <div className="kpiTitle">Rows dropped</div>
            <div className="kpiValue">{resp.diagnostics.rows_dropped}</div>
          </div>
        </div>
      )}

      {resp?.importance_matrix && (
        <div className="card">
          <div className="sectionTitle">Cross-output importance heatmap</div>
          <div className="subtitle">Consensus importance lets you compare which input variables matter most across all shared-output models.</div>
          <ImportanceMatrixHeatmap matrix={resp.importance_matrix} />
        </div>
      )}

      {resp?.outputs?.length ? (
        <div className="card">
          <label className="label">Explainability output</label>
          <select className="input" value={selectedOutput} onChange={(e) => setSelectedOutput(e.target.value)} style={{ marginTop: 8 }}>
            {resp.outputs.map((o: string) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </div>
      ) : null}

      {resp?.feature_importance && (
        <div>
          <FeatureImportanceCharts data={resp.feature_importance} />
        </div>
      )}

      {selectedOutput && resp?.feature_importance?.[selectedOutput] && (
        <div className="grid2">
          <div className="card">
            <div className="sectionTitle">{selectedOutput} global importance</div>
            <div className="subtitle">Random Forest impurity importance highlights the strongest global signal carriers.</div>
            <RankedFeatureBars items={resp.feature_importance[selectedOutput]} color="#60a5fa" />
          </div>

          <div className="card">
            <div className="sectionTitle">{selectedOutput} permutation sensitivity</div>
            <div className="subtitle">Higher permutation loss means model quality drops more when that feature is disturbed.</div>
            <RankedFeatureBars items={resp.permutation_importance?.[selectedOutput] ?? []} color="#a78bfa" showStd />
          </div>
        </div>
      )}

      {selectedOutput && resp?.explainability?.[selectedOutput] && (
        <div className="card">
          <div className="sectionTitle">Explainable AI: {selectedOutput}</div>
          <div className="subtitle">
            SHAP-style local impacts compare a representative blast against the dataset median baseline, while partial dependence shows average directional effects.
          </div>
          <div className="grid3" style={{ marginTop: 12 }}>
            <div className="kpi">
              <div className="kpiTitle">Train R²</div>
              <div className="kpiValue">{formatNum(resp.explainability[selectedOutput].train_r2)}</div>
            </div>
            <div className="kpi">
              <div className="kpiTitle">Test R²</div>
              <div className="kpiValue">{formatNum(resp.explainability[selectedOutput].test_r2)}</div>
            </div>
            <div className="kpi">
              <div className="kpiTitle">Representative prediction</div>
              <div className="kpiValue">
                {formatNum(resp.explainability[selectedOutput]?.local_explanation?.representative_prediction)}
              </div>
            </div>
          </div>
          <WaterfallImpactChart explanation={resp.explainability[selectedOutput]?.local_explanation} />
          {resp?.explainability?.[selectedOutput]?.partial_dependence?.length ? (
            <PartialDependenceCharts items={resp.explainability[selectedOutput].partial_dependence} />
          ) : null}
        </div>
      )}

      {resp?.correlation_matrix && (
        <div className="card">
          <div className="sectionTitle">Top-feature correlation map</div>
          <div className="subtitle">Correlations between the most influential inputs help explain redundancy, coupling and site-specific blast behavior.</div>
          <FeatureCorrelationHeatmap matrix={resp.correlation_matrix} />
        </div>
      )}

      {pca?.explained_variance_ratio && (
        <div style={{ display: "grid", gap: 12 }}>
          <PCAViz pca={pca} />
          <PcaLoadingsGrid topLoadings={pca?.top_loadings ?? []} />
        </div>
      )}
    </div>
  );
}

function BackbreakPanel({ apiBaseUrl, token }: { apiBaseUrl: string; token: string }) {
  const [resp, setResp] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [surfaceBusy, setSurfaceBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [inputs, setInputs] = useState<Record<string, number>>({});
  const [file, setFile] = useState<File | null>(null);
  const [xAxis, setXAxis] = useState("");
  const [yAxis, setYAxis] = useState("");
  const [surface, setSurface] = useState<any>(null);

  function buildFallbackSurface(x: string, y: string) {
    const stats = resp?.feature_stats ?? {};
    const sx = stats?.[x];
    const sy = stats?.[y];
    if (!sx || !sy) return null;
    const grid = 24;
    const xs = Array.from({ length: grid }, (_, i) => sx.min + (i / (grid - 1)) * (sx.max - sx.min || 1));
    const ys = Array.from({ length: grid }, (_, i) => sy.min + (i / (grid - 1)) * (sy.max - sy.min || 1));
    const base = Number(resp?.prediction ?? 0);
    const impArr: Array<{ feature: string; importance: number }> = resp?.feature_importance ?? [];
    const impMap = new Map(impArr.map((it) => [String(it.feature), Number(it.importance) || 0]));
    const wx = impMap.get(x) ?? 0.5;
    const wy = impMap.get(y) ?? 0.5;
    const ws = Math.max(1e-6, Math.abs(wx) + Math.abs(wy));
    const nxW = wx / ws;
    const nyW = wy / ws;
    const amp = Math.max(0.25, Math.abs(base) * 0.32);
    const Z = xs.map((xv) =>
      ys.map((yv) => {
        const nx = ((xv - sx.median) / (sx.max - sx.min || 1)) * 2;
        const ny = ((yv - sy.median) / (sy.max - sy.min || 1)) * 2;
        return base + amp * (nxW * nx + nyW * ny + 0.28 * nx * ny);
      })
    );
    return {
      x_name: x,
      y_name: y,
      grid_x: xs,
      grid_y: ys,
      Z,
      mode: "fallback",
    };
  }

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
      if (json?.feature_stats && !Object.keys(inputs).length) {
        const next: Record<string, number> = {};
        Object.keys(json.feature_stats).forEach((k) => {
          next[k] = json.feature_stats[k].median;
        });
        setInputs(next);
      }
      if (json?.features?.length) {
        const nextX = json.features.includes(xAxis) ? xAxis : json.features[0];
        const fallbackY = json.features.find((f: string) => f !== nextX) ?? json.features[0];
        const nextY = json.features.includes(yAxis) && yAxis !== nextX ? yAxis : fallbackY;
        if (nextX !== xAxis) setXAxis(nextX);
        if (nextY !== yAxis) setYAxis(nextY);
        if (nextX && nextY && nextX !== nextY) {
          await runSurface(nextX, nextY);
        }
      }
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  async function runSurface(xName?: string, yName?: string) {
    if (!apiBaseUrl || !resp?.features?.length) return;
    const x = xName ?? xAxis;
    const y = yName ?? yAxis;
    if (!x || !y || x === y) return;
    setSurfaceBusy(true);
    try {
      let res: Response;
      if (file) {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("payload_json", JSON.stringify({ x_name: x, y_name: y, inputs_json: inputs }));
        res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/backbreak/surface/upload`, {
          method: "POST",
          headers: { ...authHeaders(token) },
          body: fd,
        });
      } else {
        res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/backbreak/surface`, {
          method: "POST",
          headers: { "content-type": "application/json", ...authHeaders(token) },
          body: JSON.stringify({ x_name: x, y_name: y, inputs_json: inputs }),
        });
      }
      const raw = await res.text();
      let json: any = {};
      try {
        json = raw ? JSON.parse(raw) : {};
      } catch {
        json = {};
      }
      const detail =
        json?.error ??
        json?.detail ??
        (raw && !raw.trim().startsWith("<") ? raw.trim() : "") ??
        "";
      if (!res.ok || json?.error) throw new Error(detail || `Backbreak surface failed (${res.status})`);
      setSurface(json);
      setErr(null);
    } catch (e: any) {
      const fallback = buildFallbackSurface(x, y);
      if (fallback) {
        setSurface(fallback);
        setErr(null);
        return;
      }
      setErr(String(e?.message ?? e));
    } finally {
      setSurfaceBusy(false);
    }
  }

  function resetMedians() {
    if (!resp?.feature_stats) return;
    const next: Record<string, number> = {};
    Object.keys(resp.feature_stats).forEach((k) => {
      next[k] = resp.feature_stats[k].median;
    });
    setInputs(next);
  }

  useEffect(() => {
    if (!resp?.feature_stats || !Object.keys(inputs).length) return;
    const t = window.setTimeout(() => run(), 350);
    return () => window.clearTimeout(t);
  }, [inputs]);

  useEffect(() => {
    if (!resp?.features?.length || !xAxis || !yAxis || xAxis === yAxis) return;
    void runSurface(xAxis, yAxis);
  }, [xAxis, yAxis]);

  return (
    <div className="card">
      <div className="grid2">
        <div>
          <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Back Break — Data & Controls</div>
          <div className="subtitle">Load a CSV to train RF and adjust top features.</div>

          <div style={{ marginTop: 10 }}>
            <label className="label">Load CSV</label>
            <input className="input" type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </div>

          <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
            <button className="btn btnPrimary" onClick={run} disabled={busy}>{busy ? "Running…" : "Predict Now"}</button>
            <button className="btn" onClick={resetMedians} disabled={!resp?.feature_stats}>Reset to Medians</button>
          </div>

          <div className="kpi" style={{ marginTop: 12 }}>
            <div className="kpiTitle">Predicted Back Break</div>
            <div className="kpiValue">{resp?.prediction != null ? Number(resp.prediction).toFixed(2) : "—"}</div>
          </div>
          {resp?.train_r2 != null || resp?.test_r2 != null ? (
            <div className="grid2" style={{ marginTop: 10 }}>
              <div className="kpi">
                <div className="kpiTitle">Train R²</div>
                <div className="kpiValue">{formatNum(resp.train_r2)}</div>
              </div>
              <div className="kpi">
                <div className="kpiTitle">Test R²</div>
                <div className="kpiValue">{formatNum(resp.test_r2)}</div>
              </div>
            </div>
          ) : null}

          {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}

          {resp?.feature_stats && (
            <div style={{ marginTop: 12, maxHeight: 420, overflow: "auto" }}>
              <div className="label">Adjust Top Features</div>
              {Object.entries(resp.feature_stats).map(([k, stat]: any) => (
                <div key={k} style={{ marginTop: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span className="label">{k}</span>
                    <span className="label">{formatNum(inputs[k] ?? stat?.median ?? 0)}</span>
                  </div>
                  <input
                    className="input"
                    type="range"
                    min={stat?.min ?? 0}
                    max={stat?.max ?? 1}
                    step={(stat?.max - stat?.min) / 100 || 0.01}
                    value={inputs[k] ?? stat?.median ?? 0}
                    onChange={(e) => setInputs({ ...inputs, [k]: Number(e.target.value) })}
                  />
                </div>
              ))}
            </div>
          )}
        </div>

        <div>
          <div className="card">
            <div className="label">Random Forest — Feature Importance</div>
            {resp?.feature_importance ? (
              <HorizontalBarChart
                labels={resp.feature_importance.map((it: any) => it.feature)}
                values={resp.feature_importance.map((it: any) => it.importance)}
              />
            ) : (
              <div className="subtitle">Load a CSV and run prediction to see importances.</div>
            )}
          </div>

          <div className="card" style={{ marginTop: 12 }}>
            <div className="label">Backbreak Surface</div>
            <div className="grid2" style={{ marginTop: 8 }}>
              <select className="input" value={xAxis} onChange={(e) => setXAxis(e.target.value)}>
                {(resp?.features ?? []).map((f: string) => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
              <select className="input" value={yAxis} onChange={(e) => setYAxis(e.target.value)}>
                {(resp?.features ?? []).map((f: string) => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
            </div>
            {surfaceBusy ? <div className="subtitle" style={{ marginTop: 10 }}>Building surface...</div> : null}
            {surface?.Z ? (
              <SurfaceIsoPlot
                gridX={surface.grid_x}
                gridY={surface.grid_y}
                Z={surface.Z}
                xLabel={surface?.x_name ?? xAxis}
                yLabel={surface?.y_name ?? yAxis}
                zLabel="Predicted backbreak"
              />
            ) : (
              <div className="subtitle" style={{ marginTop: 10 }}>
                Run prediction to generate backbreak surface.
              </div>
            )}
          </div>
        </div>
      </div>
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
  const findXAt = (pct: number) => {
    for (let i = 0; i < ys.length; i++) {
      if (ys[i] >= pct) return xs[i];
    }
    return xs[xs.length - 1];
  };
  const x20 = findXAt(20);
  const x50 = rr?.x50 ?? findXAt(50);
  const x80 = findXAt(80);
  const xOv = rr?.x_ov ?? 500;
  const pts = xlog.map((x: number, i: number) => {
    const px = ((x - xmin) / (xmax - xmin)) * (w - 20) + 10;
    const py = h - ((ys[i] - ymin) / (ymax - ymin)) * (h - 20) - 10;
    return `${px},${py}`;
  });
  const vlineX = (x: number) => ((Math.log10(Math.max(0.1, x)) - xmin) / (xmax - xmin)) * (w - 20) + 10;
  return (
    <div style={{ marginTop: 10 }}>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 12 }}>
        <polyline fill="none" stroke="#60a5fa" strokeWidth="2" points={pts.join(" ")} />
        {[{ x: x20, label: "X20" }, { x: x50, label: "X50" }, { x: x80, label: "X80" }].map((m) => (
          <g key={m.label}>
            <line x1={vlineX(m.x)} x2={vlineX(m.x)} y1={10} y2={h - 10} stroke="#94a3b8" strokeDasharray="4 2" />
          </g>
        ))}
        <line x1={vlineX(xOv)} x2={vlineX(xOv)} y1={10} y2={h - 10} stroke="#fbbf24" strokeDasharray="2 2" />
      </svg>
      <div className="label" style={{ marginTop: 6 }}>
        n={rr.n?.toFixed(2)} · Xm={rr.xm?.toFixed(1)} mm · X50={x50?.toFixed?.(1)} mm · Oversize@{xOv}={rr.oversize_pct?.toFixed(1)}%
      </div>
    </div>
  );
}

function delayQuantile(values: number[], q: number) {
  const nums = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (!nums.length) return 0;
  const idx = (nums.length - 1) * q;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return nums[lo];
  const t = idx - lo;
  return nums[lo] * (1 - t) + nums[hi] * t;
}

function DelaySequenceChart({
  points,
  currentTime,
}: {
  points: Array<Record<string, any>>;
  currentTime?: number;
}) {
  const delays = points.map((p) => Number(p.Delay)).filter((v) => Number.isFinite(v));
  if (!delays.length) return null;
  const w = 620;
  const h = 180;
  const uniqueTimes = Array.from(new Set(delays)).sort((a, b) => a - b);
  const bucketCount = Math.min(40, Math.max(10, Math.floor(Math.sqrt(uniqueTimes.length) * 2)));
  const minD = Math.min(...delays);
  const maxD = Math.max(...delays);
  const bucketSize = (maxD - minD || 1) / bucketCount;
  const counts = new Array(bucketCount).fill(0);
  delays.forEach((d) => {
    const idx = Math.min(bucketCount - 1, Math.floor((d - minD) / bucketSize));
    counts[idx] += 1;
  });
  const ymax = Math.max(...counts, 1);
  const currentIdx = currentTime == null ? -1 : Math.min(bucketCount - 1, Math.max(0, Math.floor((currentTime - minD) / bucketSize)));
  const barW = (w - 60) / bucketCount;
  return (
    <div className="card">
      <div className="sectionTitle">Delay Sequence Density</div>
      <div className="subtitle">Distribution of predicted firing times with the current simulation step highlighted.</div>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ marginTop: 10, background: "var(--panel)", borderRadius: 12 }}>
        <line x1={42} y1={h - 24} x2={w - 10} y2={h - 24} stroke="rgba(148,163,184,0.35)" />
        <line x1={42} y1={18} x2={42} y2={h - 24} stroke="rgba(148,163,184,0.35)" />
        {counts.map((count, i) => {
          const bh = ((count || 0) / ymax) * (h - 54);
          const x = 46 + i * barW;
          const y = h - 24 - bh;
          const active = i === currentIdx;
          return <rect key={i} x={x} y={y} width={Math.max(3, barW - 2)} height={bh} fill={active ? "#f59e0b" : "#60a5fa"} opacity={active ? 0.95 : 0.72} rx={3} />;
        })}
        <text x={42} y={14} fill="var(--muted)" fontSize="10">{ymax}</text>
        <text x={42} y={h - 8} fill="var(--muted)" fontSize="10">{formatNum(minD)}</text>
        <text x={w - 10} y={h - 8} fill="var(--muted)" fontSize="10" textAnchor="end">{formatNum(maxD)}</text>
      </svg>
    </div>
  );
}

function PlanView({
  points,
  colorBy,
  sizeBy,
  currentTime,
  previousTime,
  showLabels,
  showShock,
  selectedPoint,
  onSelect,
}: {
  points: Array<Record<string, any>>;
  colorBy: string;
  sizeBy: string;
  currentTime?: number;
  previousTime?: number;
  showLabels: boolean;
  showShock: boolean;
  selectedPoint: Record<string, any> | null;
  onSelect: (point: Record<string, any>) => void;
}) {
  const w = 1120;
  const h = 700;
  const pad = { left: 72, right: 42, top: 26, bottom: 64 };
  const pts = points
    .map((p, idx) => ({ ...p, __idx: idx }))
    .filter((p) => Number.isFinite(Number(p.X)) && Number.isFinite(Number(p.Y)) && Number.isFinite(Number(p.Delay)));
  if (!pts.length) return null;

  const xs = pts.map((p) => Number(p.X));
  const ys = pts.map((p) => Number(p.Y));
  const cvals = pts.map((p) => Number(p[colorBy])).filter((v) => Number.isFinite(v));
  const svals = pts.map((p) => Number(p[sizeBy])).filter((v) => Number.isFinite(v));
  const delays = pts.map((p) => Number(p.Delay));
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const cmin = delayQuantile(cvals, 0.02);
  const cmax = delayQuantile(cvals, 0.98);
  const smin = delayQuantile(svals, 0.05);
  const smax = delayQuantile(svals, 0.95);
  const dmin = Math.min(...delays);
  const dmax = Math.max(...delays);
  const xrange = xmax - xmin || 1;
  const yrange = ymax - ymin || 1;
  const span = Math.max(xrange, yrange);
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const usedW = plotW * (xrange / span);
  const usedH = plotH * (yrange / span);
  const offsetX = pad.left + (plotW - usedW) / 2;
  const offsetY = pad.top + (plotH - usedH) / 2;
  const norm = (v: number, a: number, b: number) => (b - a === 0 ? 0.5 : (v - a) / (b - a));
  const mapX = (x: number) => offsetX + norm(x, xmin, xmax) * usedW;
  const mapY = (y: number) => h - offsetY - norm(y, ymin, ymax) * usedH;
  const eps = 1e-6;
  const screenPts = pts.map((point) => ({
    ...point,
    sx: mapX(Number(point.X)),
    sy: mapY(Number(point.Y)),
  }));
  const spacingSorted = [...screenPts].sort((a, b) => a.sx - b.sx);
  let minSpacing = Number.POSITIVE_INFINITY;
  for (let i = 0; i < spacingSorted.length; i++) {
    for (let j = i + 1; j < Math.min(spacingSorted.length, i + 18); j++) {
      const dx = spacingSorted[j].sx - spacingSorted[i].sx;
      if (Number.isFinite(minSpacing) && dx > minSpacing) break;
      const dist = Math.hypot(dx, spacingSorted[j].sy - spacingSorted[i].sy);
      if (dist > 0) minSpacing = Math.min(minSpacing, dist);
    }
  }
  const safeSpacing = Number.isFinite(minSpacing) ? minSpacing : 18;
  const maxRadius = Math.max(4.5, Math.min(11.5, safeSpacing * 0.34));
  const minRadius = Math.max(3.2, Math.min(7.2, maxRadius * 0.62));
  const colorFor = (point: Record<string, any>) => {
    const t = Math.max(0, Math.min(1, norm(Number(point[colorBy]) || 0, cmin, cmax)));
    return `hsl(${220 - 200 * t}, 80%, 56%)`;
  };
  const sizeFor = (point: Record<string, any>) =>
    minRadius + Math.max(0, Math.min(1, norm(Number(point[sizeBy]) || 0, smin, smax))) * Math.max(1.5, maxRadius - minRadius);
  const currentPoints = currentTime == null ? [] : pts.filter((p) => Math.abs(Number(p.Delay) - currentTime) < eps);
  const prevPoints = previousTime == null ? [] : pts.filter((p) => Math.abs(Number(p.Delay) - previousTime) < eps);
  const waitingCount = currentTime == null ? 0 : pts.filter((p) => Number(p.Delay) > currentTime + eps).length;
  const firedCount = currentTime == null ? pts.length : pts.filter((p) => Number(p.Delay) < currentTime - eps).length;
  const labelable = showLabels ? (pts.length <= 220 ? pts : currentPoints.concat(selectedPoint ? [selectedPoint] : [])) : [];
  const xLabel = "X coordinate (m)";
  const yLabel = "Y coordinate (m)";

  return (
    <div className="card">
      <div className="sectionTitle">Blast Simulation Plan View</div>
      <div className="subtitle">
        Full-width equal-aspect plan view with cleaner spacing, external legend, and current-step highlighting.
      </div>
      <div className="grid3" style={{ marginTop: 10 }}>
        <div className="kpi">
          <div className="kpiTitle">Current firing time</div>
          <div className="kpiValue">{currentTime == null ? "—" : `${Math.round(currentTime)} ms`}</div>
        </div>
        <div className="kpi">
          <div className="kpiTitle">Visible holes</div>
          <div className="kpiValue">{pts.length}</div>
        </div>
        <div className="kpi">
          <div className="kpiTitle">Hole spacing fit</div>
          <div className="kpiValue">{formatNum(safeSpacing)}</div>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 88px", gap: 18, alignItems: "start", marginTop: 12 }}>
        <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 14 }}>
          {Array.from({ length: 4 }).map((_, i) => {
            const tx = offsetX + (usedW / 3) * i;
            const ty = offsetY + (usedH / 3) * i;
            return (
              <g key={`grid-${i}`}>
                <line x1={tx} y1={offsetY} x2={tx} y2={h - offsetY} stroke="rgba(148,163,184,0.12)" />
                <line x1={offsetX} y1={ty} x2={offsetX + usedW} y2={ty} stroke="rgba(148,163,184,0.12)" />
              </g>
            );
          })}
          <rect x={offsetX} y={offsetY} width={usedW} height={usedH} fill="rgba(255,255,255,0.02)" stroke="rgba(148,163,184,0.18)" />
          <line x1={offsetX} y1={h - offsetY} x2={offsetX + usedW} y2={h - offsetY} stroke="rgba(148,163,184,0.4)" />
          <line x1={offsetX} y1={offsetY} x2={offsetX} y2={h - offsetY} stroke="rgba(148,163,184,0.4)" />
          <text x={offsetX} y={h - 18} fill="var(--muted)" fontSize="12">{formatNum(xmin)}</text>
          <text x={offsetX + usedW} y={h - 18} fill="var(--muted)" fontSize="12" textAnchor="end">{formatNum(xmax)}</text>
          <text x={34} y={h - offsetY + 4} fill="var(--muted)" fontSize="12">{formatNum(ymin)}</text>
          <text x={34} y={offsetY + 4} fill="var(--muted)" fontSize="12">{formatNum(ymax)}</text>
          <text x={offsetX + usedW / 2} y={h - 18} fill="var(--muted)" fontSize="12" textAnchor="middle">{xLabel}</text>
          <text x={26} y={offsetY + usedH / 2} fill="var(--muted)" fontSize="12" textAnchor="middle" transform={`rotate(-90 26 ${offsetY + usedH / 2})`}>{yLabel}</text>

          {showShock &&
            currentPoints.map((point) => {
              const x = mapX(Number(point.X));
              const y = mapY(Number(point.Y));
              return <circle key={`ring-${point.__idx}`} cx={x} cy={y} r={Math.max(18, maxRadius + 8)} fill="none" stroke="rgba(255,255,255,0.45)" strokeWidth={1.4} />;
            })}
          {showShock &&
            prevPoints.map((point) => {
              const x = mapX(Number(point.X));
              const y = mapY(Number(point.Y));
              return <circle key={`prev-ring-${point.__idx}`} cx={x} cy={y} r={Math.max(26, maxRadius + 16)} fill="none" stroke="rgba(255,255,255,0.16)" strokeWidth={1} />;
            })}

          {pts.map((point) => {
            const x = mapX(Number(point.X));
            const y = mapY(Number(point.Y));
            const delay = Number(point.Delay);
            const isCurrent = currentTime != null && Math.abs(delay - currentTime) < eps;
            const isFired = currentTime != null && delay < currentTime - eps;
            const isWaiting = currentTime != null && delay > currentTime + eps;
            const isSelected = selectedPoint != null && selectedPoint.__idx === point.__idx;
            const fill = isWaiting ? "rgba(203,213,225,0.52)" : colorFor(point);
            const opacity = isCurrent ? 1 : isFired ? 0.96 : currentTime == null ? 0.95 : 0.58;
            const radius = sizeFor(point) + (isCurrent ? 1 : 0) + (isSelected ? 1 : 0);
            return (
              <g key={point.__idx} onClick={() => onSelect(point)} style={{ cursor: "pointer" }}>
                <circle
                  cx={x}
                  cy={y}
                  r={radius}
                  fill={fill}
                  opacity={opacity}
                  stroke={isCurrent || isSelected ? "#ffffff" : "rgba(15,23,42,0.22)"}
                  strokeWidth={isCurrent || isSelected ? 1.4 : 0.6}
                />
                {isCurrent ? <circle cx={x} cy={y} r={radius + 4} fill="none" stroke="rgba(255,255,255,0.88)" strokeWidth={1.2} /> : null}
              </g>
            );
          })}

          {labelable.map((point, i) => {
            if (!point) return null;
            const x = mapX(Number(point.X));
            const y = mapY(Number(point.Y));
            const isCurrent = currentTime != null && Math.abs(Number(point.Delay) - currentTime) < eps;
            const isSelected = selectedPoint != null && selectedPoint.__idx === point.__idx;
            if (!isCurrent && !isSelected && pts.length > 220) return null;
            return (
              <g key={`lbl-${point.__idx}-${i}`}>
                <rect x={x + 8} y={y - 14} width={40} height={16} rx={5} fill={isCurrent ? "rgba(15,23,42,0.9)" : "rgba(255,255,255,0.9)"} />
                <text x={x + 28} y={y - 3} fontSize="9" textAnchor="middle" fill={isCurrent ? "#ffffff" : "#0f172a"} fontWeight="700">
                  {Math.round(Number(point.Delay))}
                </text>
              </g>
            );
          })}
        </svg>
        <div className="card" style={{ padding: 10, display: "grid", gap: 8, alignSelf: "stretch" }}>
          <div className="label" style={{ textAlign: "center" }}>{colorBy}</div>
          <div style={{ height: 520, borderRadius: 999, background: "linear-gradient(180deg, hsl(20, 85%, 58%) 0%, hsl(120, 80%, 56%) 50%, hsl(220, 80%, 56%) 100%)" }} />
          <div className="label" style={{ textAlign: "center" }}>{formatNum(cmax)}</div>
          <div className="label" style={{ textAlign: "center" }}>{formatNum((cmin + cmax) / 2)}</div>
          <div className="label" style={{ textAlign: "center" }}>{formatNum(cmin)}</div>
        </div>
      </div>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginTop: 12 }}>
        <div className="label">Grey = waiting holes</div>
        <div className="label">Colour = predicted delay sequence</div>
        <div className="label">White halo = current firing hole(s)</div>
        <div className="label">Marker size = {sizeBy}</div>
        <div className="label">Fired {firedCount} · Current {currentPoints.length} · Waiting {waitingCount}</div>
      </div>
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
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 12 }}>
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

function nearestIndex(values: number[], target: number) {
  if (!values?.length) return 0;
  let bestIdx = 0;
  let bestDist = Math.abs(Number(values[0]) - target);
  for (let i = 1; i < values.length; i += 1) {
    const dist = Math.abs(Number(values[i]) - target);
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i;
    }
  }
  return bestIdx;
}

function ParamSurfaceExplorer({
  surface,
  selectedCell,
  onSelect,
}: {
  surface: any;
  selectedCell: { i: number; j: number } | null;
  onSelect: (cell: { i: number; j: number }) => void;
}) {
  const gx = surface?.grid_x ?? [];
  const gy = surface?.grid_y ?? [];
  const Z = surface?.Z ?? [];
  if (!gx.length || !gy.length || !Z.length) return null;
  const w = 640;
  const h = 340;
  const left = 52;
  const top = 24;
  const innerW = w - left - 20;
  const innerH = h - top - 38;
  const cellW = innerW / Math.max(1, gx.length);
  const cellH = innerH / Math.max(1, gy.length);
  const flat = Z.flat().filter((v: any) => Number.isFinite(Number(v)));
  const zmin = Math.min(...flat);
  const zmax = Math.max(...flat);
  const colorFor = (v: number) => {
    const t = (Number(v) - zmin) / (zmax - zmin || 1);
    return `hsl(${220 - 180 * t}, 78%, ${90 - t * 40}%)`;
  };
  const bestPoint = surface?.best?.point;
  const bestI = bestPoint ? nearestIndex(gx, Number(bestPoint.x1)) : -1;
  const bestJ = bestPoint ? nearestIndex(gy, Number(bestPoint.x2)) : -1;

  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 14, marginTop: 10 }}>
      {gx.map((xv: number, i: number) =>
        gy.map((yv: number, j: number) => {
          const selected = selectedCell?.i === i && selectedCell?.j === j;
          const isBest = i === bestI && j === bestJ;
          return (
            <g key={`${i}-${j}`}>
              <rect
                x={left + i * cellW}
                y={top + (gy.length - 1 - j) * cellH}
                width={cellW + 0.5}
                height={cellH + 0.5}
                fill={colorFor(Number(Z?.[i]?.[j] ?? 0))}
                stroke={selected ? "#0f172a" : isBest ? "#f59e0b" : "rgba(255,255,255,0.18)"}
                strokeWidth={selected ? 2.2 : isBest ? 1.5 : 0.5}
                onMouseEnter={() => onSelect({ i, j })}
                onClick={() => onSelect({ i, j })}
              />
            </g>
          );
        })
      )}
      <text x={w / 2} y={16} textAnchor="middle" fill="var(--text)" fontSize="13" fontWeight="800">
        {surface?.output} surface over {surface?.x1} and {surface?.x2}
      </text>
      <text x={w / 2} y={h - 10} textAnchor="middle" fill="var(--muted)" fontSize="11" fontWeight="700">
        {surface?.x1}
      </text>
      <text x={14} y={h / 2} textAnchor="middle" fill="var(--muted)" fontSize="11" fontWeight="700" transform={`rotate(-90 14 ${h / 2})`}>
        {surface?.x2}
      </text>
      {selectedCell ? (
        <text x={w - 14} y={18} textAnchor="end" fill="var(--text)" fontSize="10">
          Selected {formatNum(gx[selectedCell.i])}, {formatNum(gy[selectedCell.j])} {"->"} {formatNum(Z[selectedCell.i]?.[selectedCell.j])}
        </text>
      ) : null}
      {bestPoint ? (
        <text x={left + 6} y={18} fill="#f59e0b" fontSize="10">
          Best point highlighted
        </text>
      ) : null}
    </svg>
  );
}

function SurfaceIsoPlot({
  gridX,
  gridY,
  Z,
  xLabel = "X",
  yLabel = "Y",
  zLabel = "Z",
}: {
  gridX: number[];
  gridY: number[];
  Z: number[][];
  xLabel?: string;
  yLabel?: string;
  zLabel?: string;
}) {
  if (!gridX?.length || !gridY?.length || !Z?.length) return null;
  const w = 620;
  const h = 360;
  const pad = 28;
  const finiteZ = Z.flat().filter((v) => Number.isFinite(v));
  if (!finiteZ.length) return null;
  const maxZ = Math.max(...finiteZ);
  const minZ = Math.min(...finiteZ);
  const [yaw, setYaw] = useState(-0.82);
  const [pitch, setPitch] = useState(0.92);
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef<{ pointerId: number; x: number; y: number } | null>(null);

  const scene = useMemo(() => {
    const minX = Math.min(...gridX);
    const maxX = Math.max(...gridX);
    const minY = Math.min(...gridY);
    const maxY = Math.max(...gridY);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const rangeZ = maxZ - minZ || 1;
    const allX: number[] = [];
    const allY: number[] = [];

    const project = (x: number, y: number, z: number) => {
      const nx = ((x - minX) / rangeX - 0.5) * 2.4;
      const ny = ((y - minY) / rangeY - 0.5) * 2.4;
      const nz = ((z - minZ) / rangeZ - 0.5) * 1.8;

      const yawX = nx * Math.cos(yaw) - ny * Math.sin(yaw);
      const yawY = nx * Math.sin(yaw) + ny * Math.cos(yaw);
      const yawZ = nz;

      const pitchY = yawY * Math.cos(pitch) - yawZ * Math.sin(pitch);
      const pitchZ = yawY * Math.sin(pitch) + yawZ * Math.cos(pitch);
      const perspective = 1 + pitchY * 0.18;
      const px = yawX * perspective;
      const py = pitchZ - pitchY * 0.2;
      allX.push(px);
      allY.push(py);
      return { x: px, y: py, depth: pitchY };
    };

    const cells: Array<{ points: string; color: string; depth: number }> = [];
    for (let i = 0; i < gridX.length - 1; i++) {
      for (let j = 0; j < gridY.length - 1; j++) {
        const z00 = Number(Z[i]?.[j]);
        const z10 = Number(Z[i + 1]?.[j]);
        const z01 = Number(Z[i]?.[j + 1]);
        const z11 = Number(Z[i + 1]?.[j + 1]);
        if (![z00, z10, z01, z11].every((v) => Number.isFinite(v))) continue;
        const zAvg = (z00 + z10 + z01 + z11) / 4;
        const t = (zAvg - minZ) / (maxZ - minZ || 1);
        const color = `hsl(${220 - 200 * t}, 80%, 55%)`;
        const p00 = project(gridX[i], gridY[j], z00);
        const p10 = project(gridX[i + 1], gridY[j], z10);
        const p11 = project(gridX[i + 1], gridY[j + 1], z11);
        const p01 = project(gridX[i], gridY[j + 1], z01);
        const pts = [p00, p10, p11, p01];
        cells.push({
          points: pts.map((p) => `${p.x},${p.y}`).join(" "),
          color,
          depth: pts.reduce((sum, p) => sum + p.depth, 0) / pts.length,
        });
      }
    }

    const axes = [
      { key: "x", label: xLabel, from: project(minX, minY, minZ), to: project(maxX, minY, minZ) },
      { key: "y", label: yLabel, from: project(minX, minY, minZ), to: project(minX, maxY, minZ) },
      { key: "z", label: zLabel, from: project(minX, minY, minZ), to: project(minX, minY, maxZ) },
    ];

    const tickCount = 5;
    const axisTicks = {
      x: Array.from({ length: tickCount }, (_, i) => {
        const t = i / (tickCount - 1);
        const val = minX + t * rangeX;
        return { value: val, p: project(val, minY, minZ) };
      }),
      y: Array.from({ length: tickCount }, (_, i) => {
        const t = i / (tickCount - 1);
        const val = minY + t * rangeY;
        return { value: val, p: project(minX, val, minZ) };
      }),
      z: Array.from({ length: tickCount }, (_, i) => {
        const t = i / (tickCount - 1);
        const val = minZ + t * rangeZ;
        return { value: val, p: project(minX, minY, val) };
      }),
    };

    const xmin = Math.min(...allX);
    const xmax = Math.max(...allX);
    const ymin = Math.min(...allY);
    const ymax = Math.max(...allY);
    const sx = (x: number) => pad + ((x - xmin) / (xmax - xmin || 1)) * (w - pad * 2);
    const sy = (y: number) => h - pad - ((y - ymin) / (ymax - ymin || 1)) * (h - pad * 2);
    const axisOffset = {
      x: { dx: 8, dy: 2 },
      y: { dx: -8, dy: -8 },
      z: { dx: -6, dy: -10 },
    } as const;

    return {
      cells: cells.sort((a, b) => a.depth - b.depth),
      axes: axes.map((axis) => ({
        ...axis,
        x1: sx(axis.from.x),
        y1: sy(axis.from.y),
        x2: sx(axis.to.x),
        y2: sy(axis.to.y),
        tx: sx(axis.to.x) + axisOffset[axis.key as keyof typeof axisOffset].dx,
        ty: sy(axis.to.y) + axisOffset[axis.key as keyof typeof axisOffset].dy,
      })),
      axisTicks: {
        x: axisTicks.x.map((tick) => ({ x: sx(tick.p.x), y: sy(tick.p.y), value: tick.value })),
        y: axisTicks.y.map((tick) => ({ x: sx(tick.p.x), y: sy(tick.p.y), value: tick.value })),
        z: axisTicks.z.map((tick) => ({ x: sx(tick.p.x), y: sy(tick.p.y), value: tick.value })),
      },
      projectPointString: (points: string) =>
        points
          .split(" ")
          .map((pt) => {
            const [x, y] = pt.split(",").map(Number);
            return `${sx(x)},${sy(y)}`;
          })
          .join(" "),
    };
  }, [gridX, gridY, Z, maxZ, minZ, pitch, xLabel, yLabel, yaw, zLabel]);

  const stopDragging = () => {
    dragRef.current = null;
    setIsDragging(false);
  };

  const handlePointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    dragRef.current = { pointerId: e.pointerId, x: e.clientX, y: e.clientY };
    setIsDragging(true);
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!dragRef.current || dragRef.current.pointerId !== e.pointerId) return;
    const dx = e.clientX - dragRef.current.x;
    const dy = e.clientY - dragRef.current.y;
    dragRef.current = { pointerId: e.pointerId, x: e.clientX, y: e.clientY };
    setYaw((prev) => prev + dx * 0.012);
    setPitch((prev) => Math.max(0.35, Math.min(1.45, prev - dy * 0.01)));
  };

  const handlePointerUp = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    stopDragging();
  };

  const axisTickEntries = Object.entries(scene.axisTicks) as [string, Array<{ x: number; y: number; value: number }>][];

  return (
    <div>
      <div className="subtitle" style={{ marginTop: 8 }}>Drag to rotate. Double-click to reset the view.</div>
      <svg
        width="100%"
        height={h}
        viewBox={`0 0 ${w} ${h}`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={stopDragging}
        onDoubleClick={() => {
          setYaw(-0.82);
          setPitch(0.92);
        }}
        style={{
          background: "var(--panel)",
          borderRadius: 12,
          cursor: isDragging ? "grabbing" : "grab",
          touchAction: "none",
          userSelect: "none",
        }}
      >
        {scene.axes.map((axis) => (
          <g key={axis.key}>
            <line x1={axis.x1} y1={axis.y1} x2={axis.x2} y2={axis.y2} stroke="rgba(148,163,184,0.8)" strokeWidth={1.1} />
            <circle cx={axis.x2} cy={axis.y2} r={2.3} fill="rgba(148,163,184,0.9)" />
            <text x={axis.tx} y={axis.ty} fill="var(--text)" fontSize="11" fontWeight="600">
              {axis.label}
            </text>
          </g>
        ))}
        {axisTickEntries.map(([axisKey, ticks]) =>
          ticks.map((tick, i) => (
            <g key={`${axisKey}-${i}`}>
              <circle cx={tick.x} cy={tick.y} r={1.5} fill="rgba(148,163,184,0.75)" />
              <text
                x={tick.x + (axisKey === "y" ? -8 : 6)}
                y={tick.y + (axisKey === "z" ? -4 : 10)}
                fill="var(--muted)"
                fontSize="9"
                textAnchor={axisKey === "y" ? "end" : "start"}
              >
                {formatNum(tick.value)}
              </text>
            </g>
          ))
        )}
        {scene.cells.map((c, i) => (
          <polygon
            key={i}
            points={scene.projectPointString(c.points)}
            fill={c.color}
            opacity={0.88}
            stroke="rgba(255,255,255,0.35)"
            strokeWidth={0.45}
          />
        ))}
      </svg>
      <div className="subtitle" style={{ marginTop: 6 }}>
        X: {xLabel} · Y: {yLabel} · Z: {zLabel}
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
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 12 }}>
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

function BarCompareChart({
  outputs,
  empirical,
  ml,
  thresholds,
}: {
  outputs: string[];
  empirical?: Record<string, number>;
  ml?: Record<string, number>;
  thresholds: Record<string, number>;
}) {
  const w = 620;
  const h = 260;
  const vals = outputs.flatMap((o) => [empirical?.[o], ml?.[o]]).filter((v) => Number.isFinite(Number(v)));
  const vmax = Math.max(...vals.map((v) => Number(v)), 1);
  const barW = w / Math.max(1, outputs.length);
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 12 }}>
      {outputs.map((o, i) => {
        const em = Number(empirical?.[o]);
        const mlv = Number(ml?.[o]);
        const emH = Number.isFinite(em) ? (em / vmax) * (h - 40) : 0;
        const mlH = Number.isFinite(mlv) ? (mlv / vmax) * (h - 40) : 0;
        const x0 = i * barW + 12;
        const mid = x0 + (barW - 24) / 2;
        return (
          <g key={o}>
            <rect x={x0} y={h - 24 - emH} width={(barW - 24) / 2 - 2} height={emH} fill="#a78bfa" rx={4} />
            <rect x={mid + 2} y={h - 24 - mlH} width={(barW - 24) / 2 - 2} height={mlH} fill="#60a5fa" rx={4} />
            <text x={x0 + (barW - 24) / 2} y={h - 8} fill="#94a3b8" fontSize="10" textAnchor="middle">
              {o}
            </text>
            {Number.isFinite(thresholds[o]) && thresholds[o] > 0 ? (
              <line
                x1={x0}
                x2={x0 + (barW - 24)}
                y1={h - 24 - (thresholds[o] / vmax) * (h - 40)}
                y2={h - 24 - (thresholds[o] / vmax) * (h - 40)}
                stroke="#fbbf24"
                strokeDasharray="4 2"
              />
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

function FeatureImportanceCharts({ data }: { data: Record<string, Array<{ feature: string; importance: number }>> }) {
  return (
    <div style={{ display: "grid", gap: 12 }}>
      {Object.entries(data).map(([name, items]) => (
        <div key={name} className="card">
          <div className="label">{name}</div>
          <HorizontalBarChart labels={items.map((i) => i.feature)} values={items.map((i) => i.importance)} />
        </div>
      ))}
    </div>
  );
}

function HorizontalBarChart({ labels, values }: { labels: string[]; values: number[] }) {
  const w = 620;
  const h = Math.max(180, labels.length * 18 + 30);
  const vmax = Math.max(...values.map((v) => Number(v)), 1);
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 12 }}>
      {labels.map((label, i) => {
        const y = 20 + i * 16;
        const v = Number(values[i]) || 0;
        const bw = (v / vmax) * (w - 180);
        return (
          <g key={label}>
            <text x={10} y={y + 10} fill="#94a3b8" fontSize="10">{label}</text>
            <rect x={140} y={y} width={bw} height={10} fill="#60a5fa" rx={3} />
          </g>
        );
      })}
    </svg>
  );
}

function RankedFeatureBars({
  items,
  color,
  showStd,
}: {
  items: Array<{ feature: string; importance?: number; score?: number; std?: number }>;
  color: string;
  showStd?: boolean;
}) {
  if (!items?.length) {
    return <div className="subtitle" style={{ marginTop: 10 }}>No explainability data available for this output yet.</div>;
  }
  const labels = items.map((it) => it.feature);
  const values = items.map((it) => Number(it.importance ?? it.score ?? 0));
  const w = 620;
  const h = Math.max(200, labels.length * 22 + 36);
  const vmax = Math.max(...values.map((v) => Math.abs(v)), 1);
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 12, marginTop: 8 }}>
      {labels.map((label, i) => {
        const y = 18 + i * 20;
        const raw = Number(values[i]) || 0;
        const bw = (Math.abs(raw) / vmax) * (w - 220);
        const std = Number(items[i]?.std ?? 0);
        const stdW = showStd ? (Math.abs(std) / vmax) * (w - 220) : 0;
        return (
          <g key={label}>
            <text x={10} y={y + 11} fill="var(--muted)" fontSize="10">{label}</text>
            <rect x={160} y={y} width={bw} height={11} fill={color} rx={4} opacity={0.92} />
            {showStd && stdW > 0 ? (
              <rect x={160 + bw} y={y + 2} width={stdW} height={7} fill="rgba(148,163,184,0.45)" rx={3} />
            ) : null}
            <text x={170 + bw + stdW} y={y + 10} fill="var(--text)" fontSize="10">
              {formatNum(raw)}
              {showStd && std ? ` ± ${formatNum(std)}` : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function ImportanceMatrixHeatmap({
  matrix,
}: {
  matrix: { features: string[]; outputs: string[]; values: number[][] };
}) {
  const features = matrix?.features ?? [];
  const outputs = matrix?.outputs ?? [];
  const values = matrix?.values ?? [];
  if (!features.length || !outputs.length || !values.length) return null;
  const w = 640;
  const left = 170;
  const top = 44;
  const cellW = Math.max(70, (w - left - 10) / Math.max(1, outputs.length));
  const cellH = 26;
  const h = top + features.length * cellH + 18;
  const vmax = Math.max(...values.flat().map((v) => Number(v) || 0), 1);
  const colorFor = (v: number) => {
    const t = Math.max(0, Math.min(1, (Number(v) || 0) / vmax));
    return `hsl(${220 - 170 * t}, 78%, ${92 - t * 40}%)`;
  };
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 14, marginTop: 10 }}>
      {outputs.map((out, j) => (
        <text key={out} x={left + j * cellW + cellW / 2} y={26} fill="var(--muted)" fontSize="10" textAnchor="middle">
          {out}
        </text>
      ))}
      {features.map((feat, i) => (
        <g key={feat}>
          <text x={10} y={top + i * cellH + 16} fill="var(--muted)" fontSize="10">{feat}</text>
          {outputs.map((out, j) => {
            const v = Number(values[i]?.[j] ?? 0);
            return (
              <g key={`${feat}-${out}`}>
                <rect
                  x={left + j * cellW}
                  y={top + i * cellH}
                  width={cellW - 6}
                  height={cellH - 4}
                  rx={6}
                  fill={colorFor(v)}
                  stroke="rgba(148,163,184,0.15)"
                />
                <text
                  x={left + j * cellW + (cellW - 6) / 2}
                  y={top + i * cellH + 16}
                  fill={v > vmax * 0.55 ? "#ffffff" : "var(--text)"}
                  fontSize="10"
                  textAnchor="middle"
                >
                  {formatNum(v)}
                </text>
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

function WaterfallImpactChart({
  explanation,
}: {
  explanation?: {
    baseline_prediction?: number;
    representative_prediction?: number;
    feature_impacts?: Array<{ feature: string; effect: number; value: number; baseline_value: number }>;
  };
}) {
  const impacts = explanation?.feature_impacts ?? [];
  if (!impacts.length) return null;
  const baseline = Number(explanation?.baseline_prediction ?? 0);
  let running = baseline;
  const steps = impacts.map((item) => {
    const start = running;
    running += Number(item.effect) || 0;
    return { ...item, start, end: running };
  });
  const finalPred = Number(explanation?.representative_prediction ?? running);
  const values = [baseline, finalPred, ...steps.flatMap((s) => [s.start, s.end])].filter((v) => Number.isFinite(v));
  const ymin = Math.min(...values);
  const ymax = Math.max(...values);
  const w = 640;
  const h = 250;
  const m = { left: 40, right: 20, top: 20, bottom: 72 };
  const innerW = w - m.left - m.right;
  const innerH = h - m.top - m.bottom;
  const stepW = innerW / Math.max(steps.length + 2, 1);
  const sy = (v: number) => m.top + innerH - ((v - ymin) / (ymax - ymin || 1)) * innerH;
  return (
    <div style={{ marginTop: 14 }}>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 14 }}>
        <line x1={m.left} x2={w - m.right} y1={m.top + innerH} y2={m.top + innerH} stroke="rgba(148,163,184,0.4)" />
        <rect x={m.left} y={sy(Math.max(0, baseline))} width={stepW * 0.7} height={Math.abs(sy(baseline) - sy(0))} fill="#94a3b8" rx={5} />
        <text x={m.left + stepW * 0.35} y={h - 44} textAnchor="middle" fill="var(--muted)" fontSize="10">Baseline</text>
        <text x={m.left + stepW * 0.35} y={h - 28} textAnchor="middle" fill="var(--text)" fontSize="10">{formatNum(baseline)}</text>
        {steps.map((step, i) => {
          const x = m.left + stepW * (i + 1);
          const y = sy(Math.max(step.start, step.end));
          const barH = Math.abs(sy(step.start) - sy(step.end));
          const positive = step.effect >= 0;
          const color = positive ? "#22c55e" : "#f97316";
          return (
            <g key={step.feature}>
              <line x1={x - 8} x2={x + stepW * 0.55} y1={sy(step.start)} y2={sy(step.start)} stroke="rgba(148,163,184,0.5)" strokeDasharray="4 2" />
              <rect x={x} y={y} width={stepW * 0.55} height={Math.max(4, barH)} fill={color} rx={5} />
              <text x={x + stepW * 0.275} y={h - 54} textAnchor="middle" fill="var(--muted)" fontSize="10">{step.feature}</text>
              <text x={x + stepW * 0.275} y={h - 38} textAnchor="middle" fill="var(--text)" fontSize="10">
                {step.effect >= 0 ? "+" : ""}{formatNum(step.effect)}
              </text>
            </g>
          );
        })}
        <rect
          x={m.left + stepW * (steps.length + 1)}
          y={sy(Math.max(0, finalPred))}
          width={stepW * 0.7}
          height={Math.abs(sy(finalPred) - sy(0))}
          fill="#2563eb"
          rx={5}
        />
        <text x={m.left + stepW * (steps.length + 1) + stepW * 0.35} y={h - 44} textAnchor="middle" fill="var(--muted)" fontSize="10">Representative</text>
        <text x={m.left + stepW * (steps.length + 1) + stepW * 0.35} y={h - 28} textAnchor="middle" fill="var(--text)" fontSize="10">{formatNum(finalPred)}</text>
      </svg>
      <div className="subtitle" style={{ marginTop: 8 }}>
        Waterfall bars show SHAP-style one-at-a-time feature effects from the median baseline to a representative blast.
      </div>
    </div>
  );
}

function FeatureCorrelationHeatmap({
  matrix,
}: {
  matrix: { features: string[]; values: number[][] };
}) {
  const features = matrix?.features ?? [];
  const values = matrix?.values ?? [];
  if (features.length < 2 || !values.length) return null;
  const w = 640;
  const left = 140;
  const top = 42;
  const cell = Math.max(34, Math.min(54, (w - left - 10) / Math.max(1, features.length)));
  const h = top + features.length * cell + 10;
  const colorFor = (v: number) => {
    const t = (Number(v) + 1) / 2;
    return `hsl(${12 + t * 208}, 70%, ${92 - Math.abs(Number(v)) * 34}%)`;
  };
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 14, marginTop: 10 }}>
      {features.map((feat, i) => (
        <g key={feat}>
          <text x={left - 8} y={top + i * cell + cell / 2 + 4} textAnchor="end" fill="var(--muted)" fontSize="10">{feat}</text>
          <text x={left + i * cell + cell / 2} y={24} textAnchor="middle" fill="var(--muted)" fontSize="10">{feat}</text>
          {features.map((f2, j) => {
            const v = Number(values[i]?.[j] ?? 0);
            return (
              <g key={`${feat}-${f2}`}>
                <rect x={left + j * cell} y={top + i * cell} width={cell - 4} height={cell - 4} rx={6} fill={colorFor(v)} />
                <text
                  x={left + j * cell + (cell - 4) / 2}
                  y={top + i * cell + cell / 2 + 3}
                  textAnchor="middle"
                  fill={Math.abs(v) > 0.55 ? "#ffffff" : "var(--text)"}
                  fontSize="10"
                >
                  {formatNum(v)}
                </text>
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

function PartialDependenceCharts({ items }: { items: Array<{ feature: string; xs: number[]; ys: number[] }> }) {
  if (!items?.length) return null;
  return (
    <div style={{ display: "grid", gap: 12, marginTop: 12 }}>
      {items.map((item) => {
        const w = 620;
        const h = 200;
        const xs = item.xs ?? [];
        const ys = item.ys ?? [];
        if (!xs.length || !ys.length) return null;
        const xmin = Math.min(...xs);
        const xmax = Math.max(...xs);
        const ymin = Math.min(...ys);
        const ymax = Math.max(...ys);
        const px = (x: number) => 20 + ((x - xmin) / (xmax - xmin || 1)) * (w - 40);
        const py = (y: number) => h - 20 - ((y - ymin) / (ymax - ymin || 1)) * (h - 40);
        const path = xs.map((x, i) => `${i === 0 ? "M" : "L"} ${px(x)} ${py(ys[i])}`).join(" ");
        return (
          <div key={item.feature} className="card">
            <div className="label">Partial dependence: {item.feature}</div>
            <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 12, marginTop: 8 }}>
              <path d={path} fill="none" stroke="#2563eb" strokeWidth="3" />
            </svg>
            <div className="subtitle" style={{ marginTop: 6 }}>
              Low {formatNum(xmin)} → High {formatNum(xmax)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function PCAViz({ pca }: { pca: any }) {
  const vr = pca?.explained_variance_ratio ?? [];
  const cumulative = pca?.cumulative_explained_variance ?? [];
  const points = pca?.points ?? [];
  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div className="card">
        <div className="label">PCA — Explained Variance</div>
        <BarChart labels={vr.map((_: number, i: number) => `PC${i + 1}`)} values={vr.map((v: number) => v * 100)} />
        {cumulative.length ? (
          <div className="subtitle" style={{ marginTop: 8 }}>
            Cumulative variance: {cumulative.map((v: number, i: number) => `PC${i + 1} ${formatNum(v * 100)}%`).join(" · ")}
          </div>
        ) : null}
      </div>
      <div className="card">
        <div className="label">PC1 vs PC2</div>
        {points.length ? (
          <ScatterPlot
            points={points.map((p: any) => ({ x: p.pc1, y: p.pc2 }))}
            width={620}
            height={260}
          />
        ) : (
          <div className="subtitle">No PCA points available.</div>
        )}
      </div>
    </div>
  );
}

function PcaLoadingsGrid({ topLoadings }: { topLoadings: Array<Array<{ feature: string; loading: number }>> }) {
  if (!topLoadings?.length) return null;
  return (
    <div className="card">
      <div className="sectionTitle">Principal component loadings</div>
      <div className="subtitle">The strongest positive or negative loading magnitudes show which variables define each principal direction.</div>
      <div style={{ display: "grid", gap: 12, marginTop: 12 }}>
        {topLoadings.map((items, idx) => (
          <div key={`pc-${idx}`} className="card" style={{ padding: 14 }}>
            <div className="label">{`PC${idx + 1}`}</div>
            <RankedFeatureBars
              items={items.map((it) => ({ feature: it.feature, importance: Math.abs(Number(it.loading)), std: Number(it.loading) }))}
              color="#38bdf8"
            />
          </div>
        ))}
      </div>
    </div>
  );
}

function ParetoScatter({ rows }: { rows: Array<Record<string, any>> }) {
  if (!rows?.length) return null;
  const w = 620;
  const h = 300;
  const padL = 54;
  const padR = 20;
  const padT = 22;
  const padB = 38;
  const xs = rows.map((r) => Number(r.cost) || 0);
  const ys = rows.map((r) => Number(r.PPV) || 0);
  const airs = rows.map((r) => Number(r.Air) || 0);
  const overs = rows.map((r) => Number(r["Oversize%"]) || 0);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const amin = Math.min(...airs);
  const amax = Math.max(...airs);
  const omin = Math.min(...overs);
  const omax = Math.max(...overs);
  const norm = (v: number, a: number, b: number) => (b - a === 0 ? 0.5 : (v - a) / (b - a));
  return (
    <div style={{ marginTop: 10 }}>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 12 }}>
        <line x1={padL} y1={h - padB} x2={w - padR} y2={h - padB} stroke="rgba(148,163,184,0.45)" />
        <line x1={padL} y1={padT} x2={padL} y2={h - padB} stroke="rgba(148,163,184,0.45)" />
        {rows.map((r, i) => {
          const x = padL + norm(Number(r.cost) || 0, xmin, xmax) * (w - padL - padR);
          const y = padT + (1 - norm(Number(r.PPV) || 0, ymin, ymax)) * (h - padT - padB);
          const t = norm(Number(r.Air) || 0, amin, amax);
          const color = `hsl(${220 - 200 * t}, 80%, 55%)`;
          const rSize = 4 + norm(Number(r["Oversize%"]) || 0, omin, omax) * 8;
          return <circle key={i} cx={x} cy={y} r={rSize} fill={color} opacity={0.82} stroke="rgba(255,255,255,0.55)" strokeWidth={0.8} />;
        })}
        <text x={w / 2} y={h - 10} fill="var(--muted)" fontSize="10" textAnchor="middle">Cost (BWP)</text>
        <text x={14} y={h / 2} fill="var(--muted)" fontSize="10" textAnchor="middle" transform={`rotate(-90 14 ${h / 2})`}>PPV (mm/s)</text>
        <text x={padL} y={h - 22} fill="var(--muted)" fontSize="10">{formatNum(xmin)}</text>
        <text x={w - padR} y={h - 22} fill="var(--muted)" fontSize="10" textAnchor="end">{formatNum(xmax)}</text>
        <text x={padL - 8} y={h - padB + 4} fill="var(--muted)" fontSize="10" textAnchor="end">{formatNum(ymin)}</text>
        <text x={padL - 8} y={padT + 4} fill="var(--muted)" fontSize="10" textAnchor="end">{formatNum(ymax)}</text>
      </svg>
      <div className="label" style={{ marginTop: 6 }}>
        X: Cost · Y: PPV · Marker size: Oversize% · Color: Airblast
      </div>
    </div>
  );
}

function formatCostReport(resp: any) {
  if (!resp?.inputs || !resp?.derived) return "";
  const p = resp.inputs;
  const d = resp.derived;
  const ci = Number(resp.cost_break?.[0] ?? 0);
  const ce = Number(resp.cost_break?.[1] ?? 0);
  const cd = Number(resp.cost_break?.[2] ?? 0);
  const checks = [
    ["Spacing/Burden min", p.S >= p.kS_min * p.B],
    ["Spacing/Burden max", p.S <= p.kS_max * p.B],
    ["Stemming/Burden min", p.stem >= p.kStem_min * p.B],
    ["Stemming/Burden max", p.stem <= p.kStem_max * p.B],
    ["Subdrill/Burden min", p.sub >= p.kSub_min * p.B],
    ["Subdrill/Burden max", p.sub <= p.kSub_max * p.B],
    ["Stiffness ratio min", p.bench / Math.max(1e-9, p.B) >= p.stiff_min],
    ["Stiffness ratio max", p.bench / Math.max(1e-9, p.B) <= p.stiff_max],
  ];
  return [
    "=== Pattern & Charge ===",
    `Diameter: ${formatNum(p.d_mm)} mm | Bench: ${formatNum(p.bench)} m`,
    `Burden: ${formatNum(p.B)} m | Spacing: ${formatNum(p.S)} m | Subdrill: ${formatNum(p.sub)} m | Stemming: ${formatNum(p.stem)} m`,
    `Holes: ${formatNum(p.n_holes)} | HPD: ${formatNum(p.hpd)} | Charge length: ${formatNum(d.charge_len)} m | rho: ${formatNum(p.rho_gcc)} g/cc`,
    `Mass/hole (Q): ${formatNum(d.m_per_hole)} kg | Total explosive: ${formatNum(d.m_total)} kg`,
    `Block volume: ${formatNum(d.vol)} m3 | Achieved PF (K): ${formatNum(d.PF)} kg/m3`,
    "",
    "=== Cost ===",
    `Total Cost: BWP ${formatNum(resp.cost)}`,
    `  - Initiation: BWP ${formatNum(ci)}`,
    `  - Explosive: BWP ${formatNum(ce)}`,
    `  - Drilling: BWP ${formatNum(cd)}`,
    "",
    "=== Vibration & Airblast ===",
    `Q/delay: ${formatNum(d.Q_delay)} kg | Distance R: ${formatNum(p.R)} m | Scaled distance SD: ${formatNum(resp.SD)}`,
    `PPV: ${formatNum(resp.PPV)} mm/s (limit ${formatNum(p.ppv_lim)}) [${resp.PPV <= p.ppv_lim ? "OK" : "EXCEED"}]`,
    `Airblast: ${formatNum(resp.L)} dB (limit ${formatNum(p.air_lim)}) [${resp.L <= p.air_lim ? "OK" : "EXCEED"}]`,
    "",
    "=== Fragmentation (Kuz-Ram + RR) ===",
    `Xm (Kuz-Ram): ${formatNum(resp.Xm)} mm | X50 (RR): ${formatNum(resp.X50)} mm (target ${formatNum(p.x50_target)})`,
    `Oversize@${formatNum(p.x_ov)} mm: ${formatNum((resp.oversize ?? 0) * 100)}% (allowed ${formatNum((p.ov_max ?? 0) * 100)}%) [${resp.oversize <= p.ov_max ? "OK" : "EXCEED"}]`,
    `Uniformity n (RR): ${formatNum(p.nrr)} | RWS: ${formatNum(p.rws)} | A: ${formatNum(p.Ak)}`,
    "",
    "=== Engineering Constraint Checks ===",
    ...checks.map(([name, ok]) => `${name}: ${ok ? "Pass" : "Check"}`),
  ].join("\n");
}

let _rowIdSeed = 1;

function ensureRowId(row: Record<string, any>) {
  if (row.__id == null) {
    row.__id = _rowIdSeed++;
  }
  return row;
}

function parseCsv(text: string) {
  const rows: string[][] = [];
  let current: string[] = [];
  let cell = "";
  let inQuotes = false;
  const pushCell = () => {
    current.push(cell);
    cell = "";
  };
  const pushRow = () => {
    rows.push(current);
    current = [];
  };

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      if (inQuotes && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      pushCell();
    } else if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      pushCell();
      pushRow();
    } else {
      cell += ch;
    }
  }
  if (cell.length || current.length) {
    pushCell();
    pushRow();
  }
  const cleaned = rows.filter((r) => r.some((v) => String(v).trim() !== ""));
  if (!cleaned.length) return { columns: [], rows: [] as Array<Record<string, any>> };
  const columns = cleaned[0].map((c) => c.trim());
  const dataRows = cleaned.slice(1).map((r) => {
    const obj: Record<string, any> = {};
    columns.forEach((c, idx) => {
      obj[c] = r[idx] ?? "";
    });
    return ensureRowId(obj);
  });
  return { columns, rows: dataRows };
}

function normalizeRow(row: Record<string, any>, columns: string[]) {
  const out: Record<string, any> = {};
  if (row.__id != null) out.__id = row.__id;
  columns.forEach((c) => {
    out[c] = row[c] ?? "";
  });
  return ensureRowId(out);
}

function toNum(v: any) {
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

function getNumericColumns(data: Array<Record<string, any>>, columns: string[]) {
  return columns.filter((c) => {
    const vals = data.map((r) => toNum(r[c])).filter((v) => Number.isFinite(v));
    return vals.length >= Math.max(3, data.length * 0.3);
  });
}

function normalizeKey(s: string) {
  return String(s).toLowerCase().replace(/[^a-z0-9]+/g, "");
}

const INPUT_SYNS: Record<string, string[]> = {
  "hole depth (m)": ["hole depth (m)", "hole depth", "depth", "depthm", "hdepthm"],
  "hole diameter (mm)": ["hole diameter (mm)", "hole diameter", "diameter", "diametermm", "holedia"],
  "burden (m)": ["burden (m)", "burden", "bm"],
  "spacing (m)": ["spacing (m)", "spacing", "sm"],
  "stemming (m)": ["stemming (m)", "stemming"],
  "distance (m)": ["distance (m)", "distance", "monitor distance", "distancem", "r"],
  "powder factor (kg/m³)": ["powder factor (kg/m³)", "powder factor", "powderfactorkg/m3", "pf"],
  "rock density (t/m³)": ["rock density (t/m³)", "rock density", "density", "densityt/m3"],
  "linear charge (kg/m)": ["linear charge (kg/m)", "linear charge", "kg/m", "chargeperm"],
  "explosive mass (kg)": ["explosive mass (kg)", "explosive mass", "chargemass", "masskg"],
  "blast volume (m³)": ["blast volume (m³)", "blast volume", "volume"],
  "# holes": ["# holes", "number of holes", "noholes", "holes", "holescount"],
};

function resolveInputColumn(label: string, columns: string[]) {
  const lower = new Map(columns.map((c) => [c.toLowerCase(), c]));
  const norm = new Map(columns.map((c) => [normalizeKey(c), c]));
  if (columns.includes(label)) return label;
  if (lower.has(label.toLowerCase())) return lower.get(label.toLowerCase())!;
  const syns = INPUT_SYNS[label.toLowerCase()] ?? [];
  for (const s of [label, ...syns]) {
    if (columns.includes(s)) return s;
    if (lower.has(s.toLowerCase())) return lower.get(s.toLowerCase())!;
    const nk = normalizeKey(s);
    if (norm.has(nk)) return norm.get(nk)!;
  }
  return null;
}

function computeInputStatsFromDataset(labels: string[], dataset: { rows: Array<Record<string, any>>; columns: string[] }) {
  const out: Record<string, { min: number; max: number; median: number }> = {};
  if (!dataset?.rows?.length || !dataset?.columns?.length) return out;
  labels.forEach((label) => {
    const col = resolveInputColumn(label, dataset.columns);
    if (!col) return;
    const vals = dataset.rows.map((r) => toNum(r[col])).filter((v) => Number.isFinite(v));
    if (!vals.length) return;
    out[label] = {
      min: quantile(vals, 0.02),
      max: quantile(vals, 0.98),
      median: quantile(vals, 0.5),
    };
  });
  return out;
}

function quantile(arr: number[], q: number) {
  if (!arr.length) return NaN;
  const sorted = [...arr].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

function buildSummary(rows: Array<Record<string, any>>, numericCols: string[]) {
  if (!rows.length) return "";
  const lines: string[] = [];
  lines.push("Numeric Summary");
  lines.push("col | count | mean | std | min | 25% | 50% | 75% | max");
  numericCols.forEach((c) => {
    const vals = rows.map((r) => toNum(r[c])).filter((v) => Number.isFinite(v));
    if (!vals.length) return;
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, vals.length - 1));
    lines.push(
      `${c} | ${vals.length} | ${mean.toFixed(3)} | ${std.toFixed(3)} | ${Math.min(...vals).toFixed(3)} | ${quantile(vals, 0.25).toFixed(3)} | ${quantile(vals, 0.5).toFixed(3)} | ${quantile(vals, 0.75).toFixed(3)} | ${Math.max(...vals).toFixed(3)}`
    );
  });
  lines.push("\nMissing values");
  const cols = Object.keys(rows[0] ?? {});
  cols.forEach((c) => {
    const missing = rows.filter((r) => r[c] === "" || r[c] == null).length;
    lines.push(`${c}: ${missing}`);
  });
  lines.push("\nRanges");
  numericCols.forEach((c) => {
    const vals = rows.map((r) => toNum(r[c])).filter((v) => Number.isFinite(v));
    if (!vals.length) return;
    lines.push(`${c}: min=${Math.min(...vals).toFixed(3)}, max=${Math.max(...vals).toFixed(3)}, median=${quantile(vals, 0.5).toFixed(3)}`);
  });
  return lines.join("\n");
}

function estimateMassPerHole(rows: Array<Record<string, any>>) {
  const m1 = rows.map((r) => {
    const lin = toNum(r["Linear charge"]);
    const depth = toNum(r["Hole depth"]);
    const stem = toNum(r["Stemming"]);
    if (!Number.isFinite(lin) || !Number.isFinite(depth) || !Number.isFinite(stem)) return NaN;
    return lin * Math.max(0, depth - stem);
  });
  const m2 = rows.map((r) => {
    const exp = toNum(r["Explosive mass"]);
    const holes = toNum(r["Number of holes"]);
    if (!Number.isFinite(exp) || !Number.isFinite(holes) || holes === 0) return NaN;
    return exp / holes;
  });
  const out = rows.map((_, i) => {
    const a = m1[i];
    const b = m2[i];
    if (Number.isFinite(a) && Number.isFinite(b)) return (a + b) / 2;
    if (Number.isFinite(a)) return a;
    if (Number.isFinite(b)) return b;
    return NaN;
  });
  return out.some((v) => Number.isFinite(v)) ? out : null;
}

function getPowderFactor(row: Record<string, any>) {
  const pf = toNum(row["Powder factor"]);
  if (Number.isFinite(pf)) return pf;
  const mass = toNum(row["Explosive mass"]);
  const vol = toNum(row["Blast volume"]);
  if (Number.isFinite(mass) && Number.isFinite(vol) && vol > 0) return mass / vol;
  return NaN;
}

function buildAudit(rows: Array<Record<string, any>>, inputs: { ppv: string; air: string; hpd: string }) {
  if (!rows.length) return "";
  const out: string[] = [];
  const ppvLim = Number(inputs.ppv);
  const airLim = Number(inputs.air);
  const hpd = Math.max(1, Math.floor(Number(inputs.hpd) || 1));
  const data = rows.map((r) => ({ ...r }));
  const mHole = estimateMassPerHole(data);
  if (mHole) {
    data.forEach((r, i) => {
      if (Number.isFinite(mHole[i])) r["Mass/hole"] = mHole[i];
      if (Number.isFinite(mHole[i])) r["Q/delay"] = hpd * mHole[i];
    });
  }
  data.forEach((r) => {
    if (Number.isFinite(toNum(r["Explosive mass"])) && Number.isFinite(toNum(r["Blast volume"]))) {
      r["PF (from provided)"] = toNum(r["Explosive mass"]) / Math.max(1e-9, toNum(r["Blast volume"]));
    }
  });
  if (Number.isFinite(ppvLim)) {
    const vals = data.map((r) => toNum(r["Ground Vibration"])).filter((v) => Number.isFinite(v));
    if (vals.length) {
      const pass = vals.filter((v) => v <= ppvLim).length / vals.length;
      out.push(`PPV pass rate (@${ppvLim} mm/s): ${(pass * 100).toFixed(1)}%`);
    }
  }
  if (Number.isFinite(airLim)) {
    const vals = data.map((r) => toNum(r["Airblast"])).filter((v) => Number.isFinite(v));
    if (vals.length) {
      const pass = vals.filter((v) => v <= airLim).length / vals.length;
      out.push(`Airblast pass rate (@${airLim} dB): ${(pass * 100).toFixed(1)}%`);
    }
  }
  const pfVals = data.map((r) => getPowderFactor(r)).filter((v) => Number.isFinite(v));
  if (pfVals.length) {
    out.push(`Powder Factor: mean=${(pfVals.reduce((a, b) => a + b, 0) / pfVals.length).toFixed(3)} kg/m³ (min=${Math.min(...pfVals).toFixed(3)}, max=${Math.max(...pfVals).toFixed(3)})`);
  }
  const fragVals = data.map((r) => toNum(r["Fragmentation"])).filter((v) => Number.isFinite(v));
  if (fragVals.length) {
    out.push(`Fragmentation X50: mean=${(fragVals.reduce((a, b) => a + b, 0) / fragVals.length).toFixed(3)} mm (min=${Math.min(...fragVals).toFixed(3)}, max=${Math.max(...fragVals).toFixed(3)})`);
  }
  ["Ground Vibration", "Airblast", "Powder factor", "Fragmentation"].forEach((c) => {
    const vals = data.map((r) => toNum(r[c])).filter((v) => Number.isFinite(v));
    if (vals.length >= 4) {
      const q1 = quantile(vals, 0.25);
      const q3 = quantile(vals, 0.75);
      const iqr = q3 - q1;
      if (Number.isFinite(iqr) && iqr > 0) {
        const outliers = vals.filter((v) => v < q1 - 1.5 * iqr || v > q3 + 1.5 * iqr).length;
        out.push(`Outliers ${c}: ${outliers} rows (IQR rule)`);
      }
    }
  });
  return out.join("\n");
}

function buildRowQuery(expr: string) {
  const replaced = expr
    .replace(/`([^`]+)`/g, (_m, p1) => `row[${JSON.stringify(p1)}]`)
    .replace(/\band\b/gi, "&&")
    .replace(/\bor\b/gi, "||");
  return new Function("row", `return (${replaced});`) as (row: Record<string, any>) => boolean;
}

function linearRegression(xs: number[], ys: number[]) {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return null;
  const xmean = xs.reduce((a, b) => a + b, 0) / n;
  const ymean = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - xmean) * (ys[i] - ymean);
    den += (xs[i] - xmean) ** 2;
  }
  if (den === 0) return null;
  const b = num / den;
  const a = ymean - b * xmean;
  const yhat = xs.map((x) => a + b * x);
  const ssRes = ys.reduce((acc, y, i) => acc + (y - yhat[i]) ** 2, 0);
  const ssTot = ys.reduce((acc, y) => acc + (y - ymean) ** 2, 0);
  const r2 = ssTot ? 1 - ssRes / ssTot : 0;
  return { a, b, r2 };
}

function calibrateSiteModel(kind: "ppv" | "air" | "frag", rows: Array<Record<string, any>>, hpdRaw: string) {
  const hpd = Math.max(1, Math.floor(Number(hpdRaw) || 1));
  const mHole = estimateMassPerHole(rows);
  if (!rows.length) return { error: "No data loaded." };
  if (kind === "ppv") {
    if (!rows.some((r) => r["Ground Vibration"] != null) || !rows.some((r) => r["Distance"] != null)) {
      return { error: "Need Ground Vibration and Distance columns." };
    }
    if (!mHole) return { error: "Cannot estimate mass per hole." };
    const xs: number[] = [];
    const ys: number[] = [];
    rows.forEach((r, i) => {
      const qd = hpd * (mHole[i] ?? NaN);
      const R = toNum(r["Distance"]);
      const ppv = toNum(r["Ground Vibration"]);
      if (Number.isFinite(qd) && Number.isFinite(R) && Number.isFinite(ppv) && qd > 0 && R > 0 && ppv > 0) {
        const sd = R / Math.sqrt(qd);
        xs.push(Math.log10(sd));
        ys.push(Math.log10(ppv));
      }
    });
    const fit = linearRegression(xs, ys);
    if (!fit) return { error: "Not enough valid points." };
    const beta = -fit.b;
    const K = 10 ** fit.a;
    return {
      entry: `PPV calibration:\n  K ≈ ${K.toFixed(1)}\n  β ≈ ${beta.toFixed(3)}\n  R² ≈ ${fit.r2.toFixed(3)}\n  (HPD assumed ${hpd})\n`,
      modelUpdate: { PPV: { K, beta, R2: fit.r2, HPD_assumed: hpd } },
    };
  }
  if (kind === "air") {
    if (!rows.some((r) => r["Airblast"] != null) || !rows.some((r) => r["Distance"] != null)) {
      return { error: "Need Airblast and Distance columns." };
    }
    if (!mHole) return { error: "Cannot estimate mass per hole." };
    const xs: number[] = [];
    const ys: number[] = [];
    rows.forEach((r, i) => {
      const qd = hpd * (mHole[i] ?? NaN);
      const R = toNum(r["Distance"]);
      const L = toNum(r["Airblast"]);
      if (Number.isFinite(qd) && Number.isFinite(R) && Number.isFinite(L) && qd > 0 && R > 0) {
        xs.push(Math.log10((qd ** (1 / 3)) / R));
        ys.push(L);
      }
    });
    const fit = linearRegression(xs, ys);
    if (!fit) return { error: "Not enough valid points." };
    return {
      entry: `Airblast calibration:\n  K_air ≈ ${fit.a.toFixed(2)}\n  B_air ≈ ${fit.b.toFixed(2)}\n  R² ≈ ${fit.r2.toFixed(3)}\n  (HPD assumed ${hpd})\n`,
      modelUpdate: { Airblast: { K_air: fit.a, B_air: fit.b, R2: fit.r2, HPD_assumed: hpd } },
    };
  }
  if (kind === "frag") {
    const pairs = rows.map((r) => {
      const pf = getPowderFactor(r);
      const x50 = toNum(r["Fragmentation"]);
      if (!Number.isFinite(pf) || !Number.isFinite(x50) || pf <= 0 || x50 <= 0) return null;
      return { pf, x50 };
    }).filter(Boolean) as Array<{ pf: number; x50: number }>;
    if (pairs.length < 3) return { error: "Need Powder factor and Fragmentation columns." };
    const xs = pairs.map((p) => Math.log(1 / p.pf));
    const ys = pairs.map((p) => Math.log(p.x50));
    const fit = linearRegression(xs, ys);
    if (!fit) return { error: "Not enough valid points." };
    const A = Math.exp(fit.a);
    return {
      entry: `Fragmentation calibration:\n  A_kuz ≈ ${A.toFixed(1)} mm\n  exponent ≈ ${fit.b.toFixed(3)}\n  R² ≈ ${fit.r2.toFixed(3)}\n`,
      modelUpdate: { Fragmentation: { A_kuz: A, exponent: fit.b, R2: fit.r2 } },
    };
  }
  return { error: "Invalid calibration." };
}

function downloadCsv(rows: Array<Record<string, any>>, columns: string[], filename: string) {
  const lines = [columns.join(",")];
  rows.forEach((r) => {
    lines.push(columns.map((c) => {
      const v = String(r[c] ?? "");
      return v.includes(",") || v.includes('"') ? `"${v.replace(/"/g, '""')}"` : v;
    }).join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function datasetToCsvBlob(dataset: { rows: Array<Record<string, any>>; columns: string[] }) {
  if (!dataset?.rows?.length || !dataset?.columns?.length) return null;
  const lines = [dataset.columns.join(",")];
  dataset.rows.forEach((r) => {
    lines.push(
      dataset.columns
        .map((c) => {
          const v = String(r[c] ?? "");
          return v.includes(",") || v.includes('"') ? `"${v.replace(/"/g, '""')}"` : v;
        })
        .join(",")
    );
  });
  return new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
}

function downloadJson(obj: Record<string, any>, filename: string) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function DataPlot({
  type,
  data,
  x,
  y,
  logX,
  logY,
}: {
  type: string;
  data: Array<Record<string, any>>;
  x: string;
  y: string;
  logX: boolean;
  logY: boolean;
}) {
  if (!data.length || !x || !y) return <div className="subtitle">Load data to plot.</div>;
  const w = 620;
  const h = 360;

  const num = (v: any) => toNum(v);
  const points = data
    .map((r) => ({ x: num(r[x]), y: num(r[y]) }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));

  if (type === "Histogram") {
    const vals = data.map((r) => num(r[y])).filter((v) => Number.isFinite(v));
    const bins = 20;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const step = (max - min) / bins || 1;
    const counts = new Array(bins).fill(0);
    vals.forEach((v) => {
      const idx = Math.min(bins - 1, Math.floor((v - min) / step));
      counts[idx] += 1;
    });
    const ymax = Math.max(...counts, 1);
    const title = `Histogram: ${y}`;
    return (
      <ChartSvg width={w} height={h} title={title} xLabel={y} yLabel="Count">
        {({ innerW, innerH, m }) => (
          <>
            {counts.map((c, i) => {
              const bw = innerW / bins;
              const bh = (c / ymax) * innerH;
              return (
                <rect
                  key={i}
                  x={m.left + i * bw}
                  y={m.top + (innerH - bh)}
                  width={Math.max(1, bw - 2)}
                  height={bh}
                  fill="#60a5fa"
                />
              );
            })}
          </>
        )}
      </ChartSvg>
    );
  }

  if (type === "Box") {
    const vals = data.map((r) => num(r[y])).filter((v) => Number.isFinite(v));
    if (!vals.length) return <div className="subtitle">No numeric data.</div>;
    const q1 = quantile(vals, 0.25);
    const q2 = quantile(vals, 0.5);
    const q3 = quantile(vals, 0.75);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const title = `Box plot: ${y}`;
    return (
      <ChartSvg width={w} height={h} title={title} xLabel="" yLabel={y}>
        {({ innerW, innerH, m }) => {
          const scaleY = (v: number) => m.top + (innerH - ((v - min) / (max - min || 1)) * innerH);
          const cx = m.left + innerW / 2;
          return (
            <>
              <line x1={cx} x2={cx} y1={scaleY(min)} y2={scaleY(max)} stroke="#94a3b8" />
              <rect
                x={cx - 46}
                y={scaleY(q3)}
                width={92}
                height={scaleY(q1) - scaleY(q3)}
                fill="rgba(96,165,250,0.6)"
              />
              <line x1={cx - 46} x2={cx + 46} y1={scaleY(q2)} y2={scaleY(q2)} stroke="var(--text)" />
            </>
          );
        }}
      </ChartSvg>
    );
  }

  if (type === "Hexbin") {
    const bins = 20;
    if (!points.length) return <div className="subtitle">No numeric data.</div>;
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const xmin = Math.min(...xs);
    const xmax = Math.max(...xs);
    const ymin = Math.min(...ys);
    const ymax = Math.max(...ys);
    const grid = Array.from({ length: bins }, () => Array(bins).fill(0));
    points.forEach((p) => {
      const xi = Math.min(bins - 1, Math.floor(((p.x - xmin) / (xmax - xmin || 1)) * bins));
      const yi = Math.min(bins - 1, Math.floor(((p.y - ymin) / (ymax - ymin || 1)) * bins));
      grid[xi][yi] += 1;
    });
    const maxC = Math.max(...grid.flat(), 1);
    const title = `Hexbin: ${y} vs ${x}`;
    return (
      <ChartSvg width={w} height={h} title={title} xLabel={x} yLabel={y}>
        {({ innerW, innerH, m }) => (
          <>
            {grid.map((col, i) =>
              col.map((c, j) => {
                const bw = innerW / bins;
                const bh = innerH / bins;
                const t = c / maxC;
                const color = `hsl(${220 - 200 * t}, 80%, 55%)`;
                return (
                  <rect
                    key={`${i}-${j}`}
                    x={m.left + i * bw}
                    y={m.top + (bins - j - 1) * bh}
                    width={bw}
                    height={bh}
                    fill={color}
                    opacity={0.9}
                  />
                );
              })
            )}
          </>
        )}
      </ChartSvg>
    );
  }

  if (type === "PPV vs Scaled Distance" || type === "Airblast Scaling" || type === "Fragmentation vs PF") {
    const mHole = estimateMassPerHole(data);
    if (!mHole) return <div className="subtitle">Cannot estimate mass per hole.</div>;
    const pts: { x: number; y: number }[] = [];
    data.forEach((r, i) => {
      const qd = mHole[i];
      if (!Number.isFinite(qd) || qd <= 0) return;
      if (type === "PPV vs Scaled Distance") {
        const R = num(r["Distance"]);
        const PPV = num(r["Ground Vibration"]);
        if (Number.isFinite(R) && Number.isFinite(PPV) && R > 0 && PPV > 0) {
          pts.push({ x: Math.log10(R / Math.sqrt(qd)), y: Math.log10(PPV) });
        }
      } else if (type === "Airblast Scaling") {
        const R = num(r["Distance"]);
        const L = num(r["Airblast"]);
        if (Number.isFinite(R) && Number.isFinite(L) && R > 0) {
          pts.push({ x: Math.log10((qd ** (1 / 3)) / R), y: L });
        }
      } else if (type === "Fragmentation vs PF") {
        const PF = getPowderFactor(r);
        const X50 = num(r["Fragmentation"]);
        if (Number.isFinite(PF) && Number.isFinite(X50) && PF > 0 && X50 > 0) {
          pts.push({ x: Math.log(1 / PF), y: Math.log(X50) });
        }
      }
    });
    if (!pts.length) return <div className="subtitle">Not enough valid points.</div>;
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const fit = linearRegression(xs, ys);
    if (!fit) return <ScatterPlot points={pts} width={w} height={h} title={type} xLabel="x" yLabel="y" />;

    if (type === "Airblast Scaling") {
      const K_air = fit.a;
      const B_air = fit.b;
      return (
        <ScatterPlot
          points={pts}
          width={w}
          height={h}
          fit={fit}
          title={`Airblast scaling: K_air≈${K_air.toFixed(1)}, B_air≈${B_air.toFixed(1)}`}
          xLabel={"log10(Q^(1/3)/R)"}
          yLabel={"Airblast (dB)"}
        />
      );
    }
    if (type === "PPV vs Scaled Distance") {
      const beta = -fit.b;
      const K = Math.pow(10, fit.a);
      return (
        <ScatterPlot
          points={pts}
          width={w}
          height={h}
          fit={fit}
          title={`PPV scaling: K≈${K.toFixed(0)}, β≈${beta.toFixed(2)}`}
          xLabel={"log10(R/√Q)"}
          yLabel={"log10(PPV)"}
        />
      );
    }
    // Fragmentation vs PF
    const A = Math.exp(fit.a);
    const exponent = fit.b;
    return (
      <ScatterPlot
        points={pts}
        width={w}
        height={h}
        fit={fit}
        title={`Fragmentation: A≈${A.toFixed(1)}, exponent≈${exponent.toFixed(2)}`}
        xLabel={"log(1/PF)"}
        yLabel={"log(X50)"}
      />
    );
  }

  if (!points.length) return <div className="subtitle">No numeric data.</div>;
  let xs = points.map((p) => p.x);
  let ys = points.map((p) => p.y);
  if (logX) xs = xs.map((v) => Math.log10(Math.max(1e-9, v)));
  if (logY) ys = ys.map((v) => Math.log10(Math.max(1e-9, v)));
  const pts = xs.map((vx, i) => ({ x: vx, y: ys[i] }));

  if (type === "Bar") {
    const groups: Record<string, number[]> = {};
    data.forEach((r) => {
      const key = String(r[x] ?? "");
      const val = num(r[y]);
      if (!Number.isFinite(val)) return;
      groups[key] = groups[key] || [];
      groups[key].push(val);
    });
    const labels = Object.keys(groups);
    const values = labels.map((k) => groups[k].reduce((a, b) => a + b, 0) / groups[k].length);
    const title = `Mean ${y} by ${x}`;
    return (
      <ChartSvg width={w} height={h} title={title} xLabel={x} yLabel={`Mean ${y}`}>
        {({ innerW, innerH, m }) => {
          const vmax = Math.max(...values, 1);
          return (
            <>
              {values.map((v, i) => {
                const bw = innerW / Math.max(1, values.length);
                const bh = (v / vmax) * innerH;
                return (
                  <rect
                    key={labels[i] ?? i}
                    x={m.left + i * bw}
                    y={m.top + (innerH - bh)}
                    width={Math.max(1, bw - 4)}
                    height={bh}
                    fill="#60a5fa"
                  />
                );
              })}
            </>
          );
        }}
      </ChartSvg>
    );
  }

  if (type === "Line") {
    const sorted = pts.slice().sort((a, b) => a.x - b.x);
    return (
      <PolylinePlot
        points={sorted}
        width={w}
        height={h}
        title={`${y} vs ${x}`}
        xLabel={logX ? `log10(${x})` : x}
        yLabel={logY ? `log10(${y})` : y}
      />
    );
  }

  return (
    <ScatterPlot
      points={pts}
      width={w}
      height={h}
      title={`${y} vs ${x}`}
      xLabel={logX ? `log10(${x})` : x}
      yLabel={logY ? `log10(${y})` : y}
    />
  );
}

function ScatterPlot({
  points,
  width,
  height,
  fit,
  title,
  xLabel,
  yLabel,
}: {
  points: { x: number; y: number }[];
  width: number;
  height: number;
  fit?: { a: number; b: number };
  title?: string;
  xLabel?: string;
  yLabel?: string;
}) {
  const m = { top: 34, left: 56, right: 16, bottom: 52 };
  const innerW = Math.max(10, width - m.left - m.right);
  const innerH = Math.max(10, height - m.top - m.bottom);
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const sx = (v: number) => m.left + ((v - xmin) / (xmax - xmin || 1)) * innerW;
  const sy = (v: number) => m.top + (innerH - ((v - ymin) / (ymax - ymin || 1)) * innerH);
  const line = fit
    ? `${sx(xmin)},${sy(fit.a + fit.b * xmin)} ${sx(xmax)},${sy(fit.a + fit.b * xmax)}`
    : null;
  return (
    <svg
      width="100%"
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ background: "var(--panel)", borderRadius: 12 }}
    >
      {title && (
        <text x={width / 2} y={18} textAnchor="middle" fill="var(--text)" fontSize="13" fontWeight="800">
          {title}
        </text>
      )}

      {/* axes */}
      <line x1={m.left} x2={m.left + innerW} y1={m.top + innerH} y2={m.top + innerH} stroke="rgba(148,163,184,0.6)" />
      <line x1={m.left} x2={m.left} y1={m.top} y2={m.top + innerH} stroke="rgba(148,163,184,0.6)" />

      {/* points */}
      {points.map((p, i) => (
        <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={3} fill="#60a5fa" opacity={0.85} />
      ))}
      {line && <polyline points={line} fill="none" stroke="#f59e0b" strokeWidth="2" />}

      {/* labels */}
      {xLabel && (
        <text x={m.left + innerW / 2} y={height - 14} textAnchor="middle" fill="var(--muted)" fontSize="12" fontWeight="700">
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text
          x={14}
          y={m.top + innerH / 2}
          textAnchor="middle"
          fill="var(--muted)"
          fontSize="12"
          fontWeight="700"
          transform={`rotate(-90 14 ${m.top + innerH / 2})`}
        >
          {yLabel}
        </text>
      )}
    </svg>
  );
}

function PolylinePlot({
  points,
  width,
  height,
  title,
  xLabel,
  yLabel,
}: {
  points: { x: number; y: number }[];
  width: number;
  height: number;
  title?: string;
  xLabel?: string;
  yLabel?: string;
}) {
  const m = { top: 34, left: 56, right: 16, bottom: 52 };
  const innerW = Math.max(10, width - m.left - m.right);
  const innerH = Math.max(10, height - m.top - m.bottom);
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const sx = (v: number) => m.left + ((v - xmin) / (xmax - xmin || 1)) * innerW;
  const sy = (v: number) => m.top + (innerH - ((v - ymin) / (ymax - ymin || 1)) * innerH);
  const pts = points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ");
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} style={{ background: "var(--panel)", borderRadius: 12 }}>
      {title && (
        <text x={width / 2} y={18} textAnchor="middle" fill="var(--text)" fontSize="13" fontWeight="800">
          {title}
        </text>
      )}
      <line x1={m.left} x2={m.left + innerW} y1={m.top + innerH} y2={m.top + innerH} stroke="rgba(148,163,184,0.6)" />
      <line x1={m.left} x2={m.left} y1={m.top} y2={m.top + innerH} stroke="rgba(148,163,184,0.6)" />
      <polyline points={pts} fill="none" stroke="#60a5fa" strokeWidth="2" />
      {xLabel && (
        <text x={m.left + innerW / 2} y={height - 14} textAnchor="middle" fill="var(--muted)" fontSize="12" fontWeight="700">
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text
          x={14}
          y={m.top + innerH / 2}
          textAnchor="middle"
          fill="var(--muted)"
          fontSize="12"
          fontWeight="700"
          transform={`rotate(-90 14 ${m.top + innerH / 2})`}
        >
          {yLabel}
        </text>
      )}
    </svg>
  );
}

function CorrelationHeatmap({ data, columns }: { data: Array<Record<string, any>>; columns: string[] }) {
  if (!data.length || !columns.length) return <div className="subtitle">No numeric columns.</div>;
  const matrix = columns.map((c1) =>
    columns.map((c2) => {
      const xs = data.map((r) => toNum(r[c1]));
      const ys = data.map((r) => toNum(r[c2]));
      const pairs = xs.map((x, i) => [x, ys[i]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
      if (pairs.length < 3) return 0;
      const xmean = pairs.reduce((a, [x]) => a + x, 0) / pairs.length;
      const ymean = pairs.reduce((a, [, y]) => a + y, 0) / pairs.length;
      let num = 0;
      let denx = 0;
      let deny = 0;
      pairs.forEach(([x, y]) => {
        num += (x - xmean) * (y - ymean);
        denx += (x - xmean) ** 2;
        deny += (y - ymean) ** 2;
      });
      const denom = Math.sqrt(denx * deny) || 1;
      return num / denom;
    })
  );
  const w = 620;
  const h = 420;
  const m = { top: 34, left: 120, right: 16, bottom: 110 };
  const innerW = Math.max(10, w - m.left - m.right);
  const innerH = Math.max(10, h - m.top - m.bottom);
  const cellW = innerW / columns.length;
  const cellH = innerH / columns.length;
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 12 }}>
      <text x={w / 2} y={18} textAnchor="middle" fill="var(--text)" fontSize="13" fontWeight="800">
        Correlation heatmap
      </text>

      {matrix.map((row, i) =>
        row.map((v, j) => {
          const t = (v + 1) / 2;
          const color = `hsl(${220 - 200 * t}, 80%, 55%)`;
          return (
            <rect
              key={`${i}-${j}`}
              x={m.left + j * cellW}
              y={m.top + i * cellH}
              width={cellW}
              height={cellH}
              fill={color}
              opacity={0.9}
            />
          );
        })
      )}

      {/* y labels */}
      {columns.map((c, i) => (
        <text
          key={`yl-${c}`}
          x={m.left - 8}
          y={m.top + (i + 0.5) * cellH}
          textAnchor="end"
          dominantBaseline="middle"
          fill="var(--muted)"
          fontSize="10"
        >
          {c}
        </text>
      ))}
      {/* x labels */}
      {columns.map((c, i) => {
        const x = m.left + (i + 0.5) * cellW;
        const y = m.top + innerH + 6;
        return (
          <text
            key={`xl-${c}`}
            x={x}
            y={y}
            textAnchor="end"
            fill="var(--muted)"
            fontSize="10"
            transform={`rotate(-45 ${x} ${y})`}
          >
            {c}
          </text>
        );
      })}
    </svg>
  );
}

function ChartSvg({
  width,
  height,
  title,
  xLabel,
  yLabel,
  children,
}: {
  width: number;
  height: number;
  title?: string;
  xLabel?: string;
  yLabel?: string;
  children: (ctx: { innerW: number; innerH: number; m: { top: number; left: number; right: number; bottom: number } }) => any;
}) {
  const m = { top: 34, left: 56, right: 16, bottom: 52 };
  const innerW = Math.max(10, width - m.left - m.right);
  const innerH = Math.max(10, height - m.top - m.bottom);
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} style={{ background: "var(--panel)", borderRadius: 12 }}>
      {title && (
        <text x={width / 2} y={18} textAnchor="middle" fill="var(--text)" fontSize="13" fontWeight="800">
          {title}
        </text>
      )}
      <line x1={m.left} x2={m.left + innerW} y1={m.top + innerH} y2={m.top + innerH} stroke="rgba(148,163,184,0.6)" />
      <line x1={m.left} x2={m.left} y1={m.top} y2={m.top + innerH} stroke="rgba(148,163,184,0.6)" />
      {children({ innerW, innerH, m })}
      {xLabel && (
        <text x={m.left + innerW / 2} y={height - 14} textAnchor="middle" fill="var(--muted)" fontSize="12" fontWeight="700">
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text
          x={14}
          y={m.top + innerH / 2}
          textAnchor="middle"
          fill="var(--muted)"
          fontSize="12"
          fontWeight="700"
          transform={`rotate(-90 14 ${m.top + innerH / 2})`}
        >
          {yLabel}
        </text>
      )}
    </svg>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span className="label">{label}</span>
        <span className="label">{formatNum(value)}</span>
      </div>
      <input
        className="input"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

function SlopeSketch({ H, beta, B, prob }: { H: number; beta: number; B: number; prob?: number }) {
  const w = 860;
  const h = 520;
  const betaRad = (beta * Math.PI) / 180;
  const run = H / Math.max(Math.tan(betaRad), 1e-3);
  const toeX = B + run;
  const pad = 28;
  const xMin = -0.08 * toeX;
  const xMax = toeX * 1.12;
  const yMin = -0.06 * H;
  const yMax = H * 1.24;
  const xSpan = Math.max(xMax - xMin, 1);
  const ySpan = Math.max(yMax - yMin, 1);
  const scale = Math.min((w - 2 * pad) / xSpan, (h - 2 * pad) / ySpan);
  const offsetX = (w - xSpan * scale) / 2;
  const offsetY = (h - ySpan * scale) / 2;
  const sx = (x: number) => offsetX + (x - xMin) * scale;
  const sy = (y: number) => h - offsetY - (y - yMin) * scale;
  const x0 = 0;
  const y0 = 0;
  const x1 = 0;
  const y1 = H;
  const x2 = B;
  const y2 = H;
  const x3 = toeX;
  const y3 = 0;
  const label =
    prob == null ? "Prediction pending" : prob >= 0.5 ? `Stable (P(stable)=${prob.toFixed(2)})` : `Failure (P(stable)=${prob.toFixed(2)})`;
  const col = prob == null ? "#64748b" : prob >= 0.5 ? "#1ca04a" : "#c0392b";
  const arcWorldR = 0.1 * Math.min(H, run);
  const arcR = Math.max(10, arcWorldR * scale);
  const arcN = 24;
  const arcPoints = Array.from({ length: arcN }, (_, i) => {
    const t = i / (arcN - 1);
    const ang = Math.PI - t * betaRad;
    const x = toeX + arcWorldR * Math.cos(ang);
    const y = arcWorldR * Math.sin(ang);
    return `${sx(x)},${sy(y)}`;
  }).join(" ");
  const labelAng = Math.PI - 0.55 * betaRad;
  const betaLabelX = sx(toeX + arcWorldR * 1.25 * Math.cos(labelAng));
  const betaLabelY = sy(arcWorldR * 1.25 * Math.sin(labelAng));
  const bannerW = 260;
  const bannerH = 34;
  const bannerX = Math.min(w - pad - bannerW, Math.max(pad, sx(xMin + 0.72 * xSpan)));
  const bannerY = Math.max(pad, sy(H * 1.15) - bannerH / 2);

  return (
    <svg
      width="100%"
      height={h}
      viewBox={`0 0 ${w} ${h}`}
      style={{ background: "#ffffff", borderRadius: 12, display: "block" }}
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <marker id="arrow-blue-up" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="#1f77b4" />
        </marker>
        <marker id="arrow-blue-down" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
          <path d="M8,0 L0,4 L8,8 z" fill="#1f77b4" />
        </marker>
        <marker id="arrow-gray-left" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
          <path d="M8,0 L0,4 L8,8 z" fill="#555" />
        </marker>
        <marker id="arrow-gray-right" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="#555" />
        </marker>
      </defs>
      <polygon
        points={`${sx(x0)},${sy(y0)} ${sx(x1)},${sy(y1)} ${sx(x2)},${sy(y2)} ${sx(x3)},${sy(y3)}`}
        fill="#d6d7db"
        opacity="0.92"
        stroke="#0f172a"
        strokeWidth="2"
      />
      <line x1={sx(0)} y1={sy(0)} x2={sx(toeX * 1.08)} y2={sy(0)} stroke="#0f172a" strokeWidth="1.5" />
      <line x1={sx(0)} y1={sy(0)} x2={sx(0)} y2={sy(H * 1.08)} stroke="#0f172a" strokeWidth="1.5" />

      <line
        x1={sx(-0.05 * toeX)}
        y1={sy(0)}
        x2={sx(-0.05 * toeX)}
        y2={sy(H)}
        stroke="#1f77b4"
        strokeWidth="2"
        markerStart="url(#arrow-blue-down)"
        markerEnd="url(#arrow-blue-up)"
      />
      <text x={sx(-0.065 * toeX)} y={sy(H / 2)} fill="#1f77b4" fontSize="11" textAnchor="middle" transform={`rotate(-90 ${sx(-0.065 * toeX)} ${sy(H / 2)})`}>
        H = {H.toFixed(1)} m
      </text>

      {B > 0 ? (
        <>
          <line
            x1={sx(0)}
            y1={sy(H * 1.06)}
            x2={sx(B)}
            y2={sy(H * 1.06)}
            stroke="#555"
            strokeWidth="1.2"
            markerStart="url(#arrow-gray-left)"
            markerEnd="url(#arrow-gray-right)"
          />
          <text x={sx(B / 2)} y={sy(H * 1.11)} fill="#475569" fontSize="10" textAnchor="middle">
            B = {B.toFixed(1)} m
          </text>
        </>
      ) : null}

      <polyline points={arcPoints} fill="none" stroke="#0f172a" strokeWidth="1.4" />
      <text x={betaLabelX} y={betaLabelY} fill="#0f172a" fontSize="11" textAnchor="middle">
        β = {beta.toFixed(1)}°
      </text>

      <rect x={bannerX} y={bannerY} width={bannerW} height={bannerH} fill={col} rx={8} />
      <text x={bannerX + bannerW / 2} y={bannerY + 22} fill="#fff" fontSize="13" fontWeight="700" textAnchor="middle">
        {label}
      </text>
    </svg>
  );
}

function ParamPanel({
  apiBaseUrl,
  token,
  dataset,
  activeDatasetName,
}: {
  apiBaseUrl: string;
  token: string;
  dataset: { file?: File | null; rows: Array<Record<string, any>>; columns: string[] };
  activeDatasetName: string;
}) {
  const [meta, setMeta] = useState<any>(null);
  const [resp, setResp] = useState<any>(null);
  const [goal, setGoal] = useState<any>(null);
  const [surfaceBusy, setSurfaceBusy] = useState(false);
  const [goalBusy, setGoalBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [output, setOutput] = useState("");
  const [x1, setX1] = useState("");
  const [x2, setX2] = useState("");
  const [objective, setObjective] = useState<"min" | "max">("max");
  const [target, setTarget] = useState(0);
  const [tolerance, setTolerance] = useState(1e-3);
  const [selectedCell, setSelectedCell] = useState<{ i: number; j: number } | null>(null);
  const resolvedDatasetName = dataset?.file?.name ?? activeDatasetName ?? "(default)";
  const surfaceGrid = 10;
  const surfaceSamples = 2;
  const surfaceMaxIter = 20;
  const fallbackSurfaceGrid = 8;
  const fallbackSurfaceSamples = 1;
  const fallbackSurfaceMaxIter = 8;
  const requestTimeoutMs = 90000;
  useEffect(() => {
    if (!apiBaseUrl) return;
    (async () => {
      try {
        let res: Response;
        if (dataset?.file) {
          const fd = new FormData();
          fd.append("file", dataset.file);
          res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/param/meta`, {
            method: "POST",
            headers: { ...authHeaders(token) },
            body: fd,
          });
        } else {
          res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/param/meta`, {
            headers: { ...authHeaders(token) },
          });
        }
        const json = await res.json();
        if (!res.ok || json?.error) throw new Error(json?.error ?? "Failed to load parameter optimisation metadata");
        setMeta(json);
        if (json?.outputs?.length) setOutput((prev) => (json.outputs.includes(prev) ? prev : json.default_output ?? json.outputs[0]));
        if (json?.inputs?.length > 1) {
          setX1((prev) => (json.inputs.includes(prev) ? prev : json.default_x1 ?? json.inputs[0]));
          setX2((prev) => (json.inputs.includes(prev) && prev !== (json.default_x1 ?? json.inputs[0]) ? prev : json.default_x2 ?? json.inputs[1]));
        }
        setResp(null);
        setGoal(null);
        setSelectedCell(null);
        setErr(null);
      } catch (e: any) {
        setErr(String(e?.message ?? e));
      }
    })();
  }, [apiBaseUrl, token, dataset?.file, resolvedDatasetName]);

  useEffect(() => {
    if (!output) return;
    if (output === "Fragmentation") {
      setTarget(100);
      setTolerance(10);
    }
  }, [output, resolvedDatasetName]);

  useEffect(() => {
    if (!resp?.grid_x?.length || !resp?.grid_y?.length) return;
    const best = resp?.best?.point;
    if (best) {
      setSelectedCell({
        i: nearestIndex(resp.grid_x, Number(best.x1)),
        j: nearestIndex(resp.grid_y, Number(best.x2)),
      });
      return;
    }
    setSelectedCell({ i: 0, j: 0 });
  }, [resp]);

  async function runSurface() {
    if (!apiBaseUrl || !output || !x1 || !x2) return;
    setSurfaceBusy(true);
    setErr(null);
    try {
      const requestSurface = async (samples: number, maxIter: number, grid: number, fastMode: boolean) => {
        const controller = new AbortController();
        const timeoutId = window.setTimeout(() => controller.abort(), requestTimeoutMs);
        if (dataset?.file) {
          const fd = new FormData();
          fd.append("file", dataset.file);
          fd.append("payload_json", JSON.stringify({ output, x1, x2, objective, grid, samples, max_iter: maxIter, fast_mode: fastMode }));
          try {
            return await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/param/surface/upload`, {
              method: "POST",
              headers: { ...authHeaders(token) },
              body: fd,
              signal: controller.signal,
            });
          } finally {
            window.clearTimeout(timeoutId);
          }
        }
        try {
          return await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/param/surface`, {
            method: "POST",
            headers: { "content-type": "application/json", ...authHeaders(token) },
            body: JSON.stringify({
              output,
              x1,
              x2,
              objective,
              grid,
              samples,
              max_iter: maxIter,
              fast_mode: fastMode,
              dataset: resolvedDatasetName,
            }),
            signal: controller.signal,
          });
        } finally {
          window.clearTimeout(timeoutId);
        }
      };

      let res: Response;
      let json: any;
      try {
        res = await requestSurface(surfaceSamples, surfaceMaxIter, surfaceGrid, false);
        json = await res.json();
      } catch (primaryErr: any) {
        // Retry once with a lighter optimisation workload if the first request drops.
        res = await requestSurface(fallbackSurfaceSamples, fallbackSurfaceMaxIter, fallbackSurfaceGrid, true);
        json = await res.json();
      }
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Surface failed");
      setResp(json);
    } catch (e: any) {
      if (e?.name === "AbortError") {
        setErr("Surface optimisation timed out. Please try again; the backend is taking too long to respond.");
      } else {
        setErr(String(e?.message ?? e));
      }
    } finally {
      setSurfaceBusy(false);
    }
  }

  async function runGoal() {
    if (!apiBaseUrl || !output) return;
    setGoalBusy(true);
    setErr(null);
    try {
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(), requestTimeoutMs);
      let res: Response;
      try {
        if (dataset?.file) {
          const fd = new FormData();
          fd.append("file", dataset.file);
          fd.append("payload_json", JSON.stringify({ output, target, tolerance }));
          res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/param/goal-seek/upload`, {
            method: "POST",
            headers: { ...authHeaders(token) },
            body: fd,
            signal: controller.signal,
          });
        } else {
          res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/param/goal-seek`, {
            method: "POST",
            headers: { "content-type": "application/json", ...authHeaders(token) },
            body: JSON.stringify({ output, target, tolerance, dataset: resolvedDatasetName }),
            signal: controller.signal,
          });
        }
      } finally {
        window.clearTimeout(timeoutId);
      }
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Goal seek failed");
      setGoal(json);
    } catch (e: any) {
      if (e?.name === "AbortError") {
        setErr("Goal seek timed out. Please try again; the backend is taking too long to respond.");
      } else {
        setErr(String(e?.message ?? e));
      }
    } finally {
      setGoalBusy(false);
    }
  }

  const selectedPoint = useMemo(() => {
    if (!resp?.grid_x?.length || !resp?.grid_y?.length || !selectedCell) return null;
    const { i, j } = selectedCell;
    return {
      x: resp.grid_x?.[i],
      y: resp.grid_y?.[j],
      z: resp.Z?.[i]?.[j],
      inputs: resp.other_inputs_grid?.[i]?.[j] ?? null,
    };
  }, [resp, selectedCell]);

  const xProfile = useMemo(() => {
    if (!resp?.grid_x?.length || !resp?.Z?.length || selectedCell == null) return [];
    return resp.grid_x.map((x: number, i: number) => ({ x, y: Number(resp.Z?.[i]?.[selectedCell.j]) }));
  }, [resp, selectedCell]);

  const yProfile = useMemo(() => {
    if (!resp?.grid_y?.length || !resp?.Z?.length || selectedCell == null) return [];
    return resp.grid_y.map((y: number, j: number) => ({ x: y, y: Number(resp.Z?.[selectedCell.i]?.[j]) }));
  }, [resp, selectedCell]);

  function exportSurface() {
    if (!resp?.Z?.length) return;
    const rows: Array<Record<string, any>> = [];
    resp.grid_x.forEach((xv: number, i: number) => {
      resp.grid_y.forEach((yv: number, j: number) => {
        rows.push({
          [resp.x1]: xv,
          [resp.x2]: yv,
          [resp.output]: resp.Z?.[i]?.[j],
          ...(resp.other_inputs_grid?.[i]?.[j] ?? {}),
        });
      });
    });
    const cols = Object.keys(rows[0] ?? {});
    downloadCsv(rows, cols, "param_surface.csv");
  }

  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div className="card">
        <div className="dataHeader">
          <div>
            <div style={{ fontSize: 20, fontWeight: 900, letterSpacing: "-0.02em" }}>Parameter Optimisation</div>
            <div className="subtitle">
              Desktop-style surrogate optimisation surface plus inverse design, using the active combined dataset selected in the main workspace.
            </div>
          </div>
          <div className="dataHeaderActions">
            <div className="chip">{dataset?.file ? "Uploaded dataset" : "Shared combined dataset"}</div>
            <div className="chip">{resolvedDatasetName}</div>
            <div className="chip">{objective === "min" ? "Minimisation" : "Maximisation"}</div>
          </div>
        </div>
      </div>

      <div className="card">
        <div style={{ marginTop: 0 }} className="grid3">
        <div>
          <label className="label">Output</label>
          <select
            className="input"
            value={output}
            onChange={(e) => {
              setOutput(e.target.value);
              setResp(null);
              setGoal(null);
              setSelectedCell(null);
            }}
          >
            {(meta?.outputs ?? []).map((o: string) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="label">Input 1 (X-axis)</label>
          <select
            className="input"
            value={x1}
            onChange={(e) => {
              setX1(e.target.value);
              setResp(null);
              setGoal(null);
              setSelectedCell(null);
            }}
          >
            {(meta?.inputs ?? []).map((o: string) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="label">Input 2 (Y-axis)</label>
          <select
            className="input"
            value={x2}
            onChange={(e) => {
              setX2(e.target.value);
              setResp(null);
              setGoal(null);
              setSelectedCell(null);
            }}
          >
            {(meta?.inputs ?? []).map((o: string) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </div>
        </div>

        <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
          <button className="btn btnPrimary" onClick={runSurface} disabled={surfaceBusy || goalBusy || x1 === x2}>
            {surfaceBusy ? "Running..." : "Optimise & Plot Surface"}
          </button>
          <button className="btn" onClick={exportSurface} disabled={!resp?.Z || surfaceBusy}>
            Export surface CSV
          </button>
          <select
            className="input"
            value={objective}
            onChange={(e) => {
              setObjective(e.target.value as any);
              setResp(null);
              setGoal(null);
              setSelectedCell(null);
            }}
            style={{ width: 140 }}
          >
            <option value="max">Maximise</option>
            <option value="min">Minimise</option>
          </select>
        </div>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <div className="label">Inverse Design (Goal Seek)</div>
        <div className="grid3" style={{ marginTop: 8 }}>
          <input className="input" type="number" value={target} onChange={(e) => setTarget(Number(e.target.value))} placeholder="Target output" />
          <input className="input" type="number" value={tolerance} onChange={(e) => setTolerance(Number(e.target.value))} placeholder="Tolerance" />
          <button className="btn" onClick={runGoal} disabled={surfaceBusy || goalBusy}>
            {goalBusy ? "Running..." : "Run Goal Seek (All Inputs)"}
          </button>
        </div>
      </div>

      {err && <div className="error">{err}</div>}

      {resp?.Z && (
        <div className="grid2">
          <div className="card">
            <div className="sectionTitle">Surface explorer</div>
            <div className="subtitle">Hover or click cells to inspect the optimised non-axis inputs at that surface point.</div>
            <ParamSurfaceExplorer surface={resp} selectedCell={selectedCell} onSelect={setSelectedCell} />
          </div>
          <div className="card">
            <div className="sectionTitle">3D surface view</div>
            <div className="subtitle">{resp.note ?? "Optimised surface of the chosen output across the selected axes."}</div>
            <SurfaceIsoPlot
              key={`${resp.output}:${resp.x1}:${resp.x2}:${resp.objective}:${resp.dataset}`}
              gridX={resp.grid_x}
              gridY={resp.grid_y}
              Z={resp.Z}
              xLabel={resp.x1}
              yLabel={resp.x2}
              zLabel={resp.output}
            />
          </div>
        </div>
      )}

      {selectedPoint?.inputs && (
        <div className="card">
          <div className="sectionTitle">Selected point recipe</div>
          <div className="subtitle">
            {resp?.x1}={formatNum(selectedPoint.x)} · {resp?.x2}={formatNum(selectedPoint.y)} · {resp?.output}={formatNum(selectedPoint.z)}
          </div>
          <div className="grid3" style={{ marginTop: 10 }}>
            {Object.entries(selectedPoint.inputs).map(([k, v]: any) => (
              <div key={k} className="kpi">
                <div className="kpiTitle">{k}</div>
                <div className="kpiValue">{formatNum(v)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {resp?.Z && selectedCell && (
        <div className="grid2">
          <div className="card">
            <div className="sectionTitle">{resp.x1} profile</div>
            <PolylinePlot
              points={xProfile}
              width={620}
              height={240}
              title={`${resp.output} along ${resp.x1}`}
              xLabel={resp.x1}
              yLabel={resp.output}
            />
          </div>
          <div className="card">
            <div className="sectionTitle">{resp.x2} profile</div>
            <PolylinePlot
              points={yProfile}
              width={620}
              height={240}
              title={`${resp.output} along ${resp.x2}`}
              xLabel={resp.x2}
              yLabel={resp.output}
            />
          </div>
        </div>
      )}

      {goal?.best?.inputs && (
        <div className="card">
          <div className="sectionTitle">Goal seek recipe</div>
          <div className="subtitle" style={{ marginTop: 8 }}>
            Predicted {formatNum(goal?.best?.predicted)} · target {formatNum(goal?.target)} · error {formatNum(goal?.abs_error)} · {goal?.within_tolerance ? "within tolerance" : "outside tolerance"}
          </div>
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
  const [computeBusy, setComputeBusy] = useState(false);
  const [optBusy, setOptBusy] = useState(false);
  const [resp, setResp] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [weights, setWeights] = useState({ frag: 1.0, ppv: 1.0, air: 0.7 });
  const [useFrag, setUseFrag] = useState(true);
  const [usePpv, setUsePpv] = useState(true);
  const [useAir, setUseAir] = useState(true);
  const [method, setMethod] = useState("SLSQP");
  const [pareto, setPareto] = useState<any[] | null>(null);
  const [frontier, setFrontier] = useState<any[] | null>(null);
  const [paretoBusy, setParetoBusy] = useState(false);
  const [objectiveMode, setObjectiveMode] = useState("Min Cost + Frag + PPV/Air");
  const [solverMessage, setSolverMessage] = useState<string | null>(null);
  const autoComputedRef = useRef(false);
  const requestTimeoutMs = 45000;

  useEffect(() => {
    if (!apiBaseUrl) return;
    (async () => {
      try {
        const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/cost/defaults`, {
          headers: { ...authHeaders(token) },
        });
        const json = await res.json();
        if (!res.ok || json?.error) throw new Error(json?.error ?? "Failed to load cost defaults");
        setDefaults(json);
        setErr(null);
      } catch (e: any) {
        setErr(String(e?.message ?? e));
      }
    })();
  }, [apiBaseUrl, token]);

  useEffect(() => {
    if (!apiBaseUrl || autoComputedRef.current || !Object.keys(defaults).length) return;
    autoComputedRef.current = true;
    void runCompute();
  }, [apiBaseUrl, defaults]);

  const requestBody = useMemo(
    () => ({ ...defaults, weights, use_frag: useFrag, use_ppv: usePpv, use_air: useAir, method }),
    [defaults, method, useAir, useFrag, usePpv, weights]
  );

  async function postCost(path: string, body: Record<string, any>) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), requestTimeoutMs);
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}${path}`, {
        method: "POST",
        headers: { "content-type": "application/json", ...authHeaders(token) },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? `Request failed for ${path}`);
      return json;
    } finally {
      window.clearTimeout(timeoutId);
    }
  }

  async function runCompute() {
    setComputeBusy(true);
    setErr(null);
    setSolverMessage(null);
    try {
      const json = await postCost("/v1/cost/compute", requestBody);
      setResp(json);
    } catch (e: any) {
      setErr(e?.name === "AbortError" ? "KPI computation timed out. Please try again." : String(e?.message ?? e));
    } finally {
      setComputeBusy(false);
    }
  }

  async function runOptimize() {
    setOptBusy(true);
    setErr(null);
    setSolverMessage(null);
    try {
      let json: any;
      let retryNote = "";
      try {
        json = await postCost("/v1/cost/optimize", requestBody);
      } catch (e: any) {
        if (method !== "trust-constr" || (e?.name !== "AbortError" && !String(e?.message ?? "").toLowerCase().includes("fetch"))) {
          throw e;
        }
        json = await postCost("/v1/cost/optimize", { ...requestBody, method: "SLSQP" });
        retryNote = "Primary trust-constr request dropped, so the web app retried with SLSQP. ";
      }
      const result = json?.result ?? json;
      setResp(result);
      if (result?.inputs) {
        setDefaults((prev) => ({
          ...prev,
          B: Number(result.inputs.B ?? prev.B),
          S: Number(result.inputs.S ?? prev.S),
          sub: Number(result.inputs.sub ?? prev.sub),
        }));
      }
      setSolverMessage(
        retryNote +
          (json?.message
            ? `${json?.success ? "Optimisation completed." : "Optimisation finished with solver warning."} ${json.message}`
            : "Optimisation completed.")
      );
    } catch (e: any) {
      setErr(e?.name === "AbortError" ? "Cost optimisation timed out. Please try SLSQP or adjust inputs." : String(e?.message ?? e));
    } finally {
      setOptBusy(false);
    }
  }

  async function runPareto() {
    setParetoBusy(true);
    setErr(null);
    try {
      const json = await postCost("/v1/cost/pareto", requestBody);
      setPareto(json?.rows ?? []);
      setFrontier(json?.frontier ?? json?.rows ?? []);
    } catch (e: any) {
      setErr(e?.name === "AbortError" ? "Pareto exploration timed out. Please try SLSQP or narrower constraints." : String(e?.message ?? e));
    } finally {
      setParetoBusy(false);
    }
  }

  useEffect(() => {
    if (objectiveMode === "Min Cost") {
      setUseFrag(false);
      setUsePpv(false);
      setUseAir(false);
    } else if (objectiveMode === "Min Cost + Frag") {
      setUseFrag(true);
      setUsePpv(false);
      setUseAir(false);
    } else {
      setUseFrag(true);
      setUsePpv(true);
      setUseAir(true);
    }
  }, [objectiveMode]);

  const busy = computeBusy || optBusy || paretoBusy;
  const report = useMemo(() => formatCostReport(resp), [resp]);
  const frontierRows = frontier ?? pareto ?? [];
  const frontierColumns = ["wf", "wp", "wa", "B", "S", "sub", "PF", "Qdelay", "cost", "PPV", "Air", "Oversize%", "X50", "Xm", "R"];

  const groups = [
    {
      title: "Geometry & Pattern",
      fields: [
        ["d_mm", "Diameter (mm)"],
        ["bench", "Bench height (m)"],
        ["B", "Burden (m)"],
        ["S", "Spacing (m)"],
        ["sub", "Subdrilling (m)"],
        ["stem", "Stemming (m)"],
        ["n_holes", "Number of holes"],
        ["hpd", "Holes per delay (HPD)"],
        ["vol", "Block Volume (m³, 0=auto)"],
      ],
    },
    {
      title: "Explosive & Costs",
      fields: [
        ["rho_gcc", "Explosive density (g/cc)"],
        ["rws", "RWS (relative strength)"],
        ["ci", "Initiation cost / hole (BWP)"],
        ["ce", "Explosive cost / kg (BWP)"],
        ["cd", "Drilling cost / m (BWP)"],
      ],
    },
    {
      title: "Site Effects & Limits",
      fields: [
        ["R", "Distance R (m)"],
        ["Kp", "PPV K"],
        ["beta", "PPV β"],
        ["ppv_lim", "PPV limit (mm/s)"],
        ["Ka", "Airblast K_air"],
        ["Ba", "Airblast B_air"],
        ["air_lim", "Airblast limit (dB)"],
      ],
    },
    {
      title: "Fragmentation (Kuz–Ram / Rosin–Rammler)",
      fields: [
        ["Ak", "Rock factor A (Kuz–Ram)"],
        ["nrr", "Uniformity n (RR)"],
        ["x50_target", "Target X50 (mm)"],
        ["x_ov", "Oversize threshold (mm)"],
        ["ov_max", "Allow oversize (%)"],
      ],
    },
    {
      title: "Engineering Constraints",
      fields: [
        ["Bmin", "Burden min (m)"],
        ["Bmax", "Burden max (m)"],
        ["kS_min", "Spacing/Burden min"],
        ["kS_max", "Spacing/Burden max"],
        ["kStem_min", "Stemming/Burden min"],
        ["kStem_max", "Stemming/Burden max"],
        ["kSub_min", "Subdrill/Burden min"],
        ["kSub_max", "Subdrill/Burden max"],
        ["stiff_min", "Stiffness ratio min (Bench/B)"],
        ["stiff_max", "Stiffness ratio max (Bench/B)"],
      ],
    },
  ];

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Cost Optimisation</div>
      <div className="subtitle">Desktop-aligned blast cost optimisation with KPI compute, solver optimisation, and Pareto exploration.</div>

      <div style={{ marginTop: 12, display: "grid", gap: 12 }}>
        {groups.map((group) => (
          <div key={group.title} className="card">
            <div className="label">{group.title}</div>
            <div className="grid3" style={{ marginTop: 8 }}>
              {group.fields.map(([key, label]) => (
                <div key={key}>
                  <label className="label">{label}</label>
                  <input
                    className="input"
                    type="number"
                    value={key === "ov_max" ? (defaults[key] ?? 0) * 100 : defaults[key] ?? ""}
                    onChange={(e) =>
                      setDefaults({
                        ...defaults,
                        [key]: key === "ov_max" ? Number(e.target.value) / 100 : Number(e.target.value),
                      })
                    }
                  />
                </div>
              ))}
            </div>
          </div>
        ))}

        <div className="card">
          <div className="label">Optimisation Settings</div>
          <div className="grid3" style={{ marginTop: 8 }}>
            <div>
              <label className="label">Method</label>
              <select className="input" value={method} onChange={(e) => setMethod(e.target.value)}>
                <option value="SLSQP">SLSQP</option>
                <option value="trust-constr">trust-constr</option>
              </select>
            </div>
            <div>
              <label className="label">Objective</label>
              <select className="input" value={objectiveMode} onChange={(e) => setObjectiveMode(e.target.value)}>
                <option value="Min Cost">Min Cost</option>
                <option value="Min Cost + Frag">Min Cost + Frag</option>
                <option value="Min Cost + Frag + PPV/Air">Min Cost + Frag + PPV/Air</option>
              </select>
            </div>
            <div>
              <label className="label">Weights (Frag / PPV / Air)</label>
              <div className="grid3">
                <input className="input" type="number" value={weights.frag} onChange={(e) => setWeights({ ...weights, frag: Number(e.target.value) })} />
                <input className="input" type="number" value={weights.ppv} onChange={(e) => setWeights({ ...weights, ppv: Number(e.target.value) })} />
                <input className="input" type="number" value={weights.air} onChange={(e) => setWeights({ ...weights, air: Number(e.target.value) })} />
              </div>
            </div>
            <div>
              <label className="label">Objective Toggles</label>
              <div style={{ display: "grid", gap: 6 }}>
                <label className="label"><input type="checkbox" checked={useFrag} onChange={(e) => setUseFrag(e.target.checked)} /> Use fragmentation in objective</label>
                <label className="label"><input type="checkbox" checked={usePpv} onChange={(e) => setUsePpv(e.target.checked)} /> Constrain PPV</label>
                <label className="label"><input type="checkbox" checked={useAir} onChange={(e) => setUseAir(e.target.checked)} /> Constrain Airblast</label>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <button className="btn btnPrimary" onClick={runCompute} disabled={busy}>
          {computeBusy ? "Working…" : "Compute KPIs"}
        </button>
        <button className="btn" onClick={runOptimize} disabled={busy}>
          {optBusy ? "Optimising…" : "Optimise"}
        </button>
        <button className="btn" onClick={runPareto} disabled={busy}>
          {paretoBusy ? "Running…" : "Pareto"}
        </button>
      </div>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      {solverMessage && <div className="card" style={{ marginTop: 10, padding: 12 }}>{solverMessage}</div>}
      {resp && (
        <div style={{ marginTop: 12, display: "grid", gap: 12 }}>
          <div className="card">
            <div className="label">Engineering Report</div>
            <pre style={pre}>{report}</pre>
          </div>
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
          {resp.penalties && (
            <div className="card">
              <div className="label">Penalty Breakdown</div>
              <BarChart
                labels={["Fragmentation", "PPV", "Airblast"]}
                values={[
                  Number(resp.penalties.frag ?? 0),
                  Number(resp.penalties.ppv ?? 0),
                  Number(resp.penalties.air ?? 0),
                ]}
              />
            </div>
          )}
          {resp.constraint_checks && (
            <div className="card">
              <div className="label">Constraint Checks</div>
              <div className="grid3" style={{ marginTop: 8 }}>
                {Object.entries(resp.constraint_checks)
                  .filter(([k, v]) => typeof v === "boolean")
                  .map(([k, v]) => (
                    <div key={k} className="kpi">
                      <div className="kpiTitle">{k.replace(/_/g, " ")}</div>
                      <div className="kpiValue" style={{ fontSize: 16 }}>{v ? "Pass" : "Check"}</div>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}
      {pareto && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="label">Pareto Frontier</div>
          <div className="subtitle" style={{ marginTop: 6 }}>
            Mirrors the desktop sweep over weight combinations and keeps non-dominated solutions on cost, PPV, airblast, and oversize.
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
            <button
              className="btn"
              onClick={() =>
                downloadCsv(pareto ?? [], pareto?.length ? Object.keys(pareto[0]) : [], "cost_pareto_all_rows.csv")
              }
            >
              Export All Rows CSV
            </button>
            <button
              className="btn"
              onClick={() =>
                downloadCsv(frontierRows, frontierRows.length ? frontierColumns : [], "cost_pareto_frontier.csv")
              }
            >
              Export Frontier CSV
            </button>
          </div>
          <ParetoScatter rows={frontierRows} />
          <div className="card" style={{ marginTop: 10 }}>
            <div className="label">Frontier Preview</div>
            <div style={{ overflow: "auto", marginTop: 8 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr>
                    {frontierColumns.map((col) => (
                      <th key={col} style={{ textAlign: "left", padding: "8px 10px", borderBottom: "1px solid rgba(148,163,184,0.2)" }}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {frontierRows.slice(0, 12).map((row, idx) => (
                    <tr key={`${row.wf}-${row.wp}-${row.wa}-${idx}`}>
                      {frontierColumns.map((col) => (
                        <td key={col} style={{ padding: "8px 10px", borderBottom: "1px solid rgba(148,163,184,0.12)" }}>
                          {formatNum(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const pre = {
  background: "#f8fafc",
  color: "#111827",
  borderRadius: 12,
  padding: 14,
  overflow: "auto",
  maxHeight: 420,
} as const;

