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

const TAB_META: Record<TabKey, { title: string; desc: string }> = {
  home: { title: "Welcome", desc: "Overview and quick access cards" },
  data: { title: "Data Manager", desc: "Upload / preview datasets (GCS-backed later)" },
  predict: { title: "Prediction", desc: "Empirical + ML outputs + RR" },
  feature: { title: "Feature Importance", desc: "RF importance + PCA" },
  param: { title: "Parameter Optimisation", desc: "Surface + goal seek" },
  cost: { title: "Cost Optimisation", desc: "KPIs, optimise, Pareto" },
  backbreak: { title: "Back Break", desc: "RF model from CSV" },
  flyrock: { title: "Flyrock (ML + Empirical)", desc: "ML + empirical lines" },
  slope: { title: "Slope Stability", desc: "Stable/Failure classifier" },
  delay: { title: "Delay Prediction", desc: "Delay prediction & plan view" },
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

  const datasetLabel = useMemo(() => {
    if (dataset.file?.name) return `Dataset: ${dataset.file.name}`;
    if (dataset.rows?.length) return "Dataset: (custom)";
    if (meta?.default_dataset) return `Dataset: ${meta.default_dataset}`;
    return "Dataset: (none)";
  }, [dataset.file, dataset.rows?.length, meta?.default_dataset]);

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
          <button className="headerTitle" onClick={() => setTab("home")} aria-label="Go to home">
            Blasting Optimization Suite
          </button>
        </div>

        <div className="headerControls">
          <span className="chip">{datasetLabel}</span>
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
                      <div className="sidebarButtonLabel">{meta.title}</div>
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

        <main style={{ minHeight: 600 }}>
          {metaErr && <div className="error">{metaErr}</div>}
          {tab === "home" ? (
            <HomePanel onOpen={setTab} />
          ) : tab === "predict" ? (
            <PredictPanel apiBaseUrl={apiBaseUrl} token={session.token} meta={meta} dataset={dataset} />
          ) : tab === "data" ? (
            <DataPanel apiBaseUrl={apiBaseUrl} token={session.token} dataset={dataset} onDatasetChange={setDataset} />
          ) : tab === "feature" ? (
            <FeaturePanel apiBaseUrl={apiBaseUrl} token={session.token} dataset={dataset} />
          ) : tab === "param" ? (
            <ParamPanel apiBaseUrl={apiBaseUrl} token={session.token} dataset={dataset} />
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
      <div className="homeGrid">
        <div className="homeCard">
          <div className="homeCardTitle">📊 Prediction</div>
          <div className="homeCardDesc">
            Run ML &amp; empirical predictions (USBM PPV/Air, Kuz–Ram Xm + RR curve).
          </div>
          <button className="btn btnPrimary" onClick={() => onOpen("predict")}>
            Open
          </button>
        </div>
        <div className="homeCard">
          <div className="homeCardTitle">💥 Cost Optimisation</div>
          <div className="homeCardDesc">
            Minimise cost with penalties for PPV, airblast, fragmentation (Xm→RR X50).
          </div>
          <button className="btn btnPrimary" onClick={() => onOpen("cost")}>
            Open
          </button>
        </div>
        <div className="homeCard">
          <div className="homeCardTitle">🪨 Flyrock</div>
          <div className="homeCardDesc">
            Predict flyrock (ML + empirical lines); check limits and distances.
          </div>
          <button className="btn btnPrimary" onClick={() => onOpen("flyrock")}>
            Open
          </button>
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
    try {
      let res: Response;
      if (dataset?.file || dataset?.rows?.length) {
        const fd = new FormData();
        if (dataset.file) {
          fd.append("file", dataset.file);
        } else {
          const blob = datasetToCsvBlob(dataset);
          if (blob) fd.append("file", blob, "dataset.csv");
        }
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
      setOut({ status: res.status, json });
    } catch (e: any) {
      setOut({ error: String(e?.message ?? e) });
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

  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div className="grid2">
        <div className="card">
          <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Inputs</div>
          <div className="subtitle">Dataset: combinedv2Orapa.csv</div>

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
      if (json?.feature_stats) {
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
    if (!apiBaseUrl || !resp?.features?.length) return;
    const x = xName ?? xAxis;
    const y = yName ?? yAxis;
    if (!x || !y || x === y) return;
    try {
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/flyrock/surface`, {
        method: "POST",
        headers: { "content-type": "application/json", ...authHeaders(token) },
        body: JSON.stringify({ x_name: x, y_name: y, inputs_json: inputs }),
      });
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
          </div>

          <div className="kpi" style={{ marginTop: 10 }}>
            <div className="kpiTitle">Empirical estimate</div>
            <div className="kpiValue">{empiricalAuto ? formatNum(empiricalAuto.value) : "—"} m</div>
            {empiricalAuto?.method && <div className="label">{empiricalAuto.method}</div>}
          </div>

          {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}

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
              <SurfaceIsoPlot gridX={surface.grid_x} gridY={surface.grid_y} Z={surface.Z} />
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

  function resolveFeature(features: string[], synonyms: string[]) {
    const lower = features.map((f) => f.toLowerCase());
    for (const s of synonyms) {
      const idx = lower.findIndex((f) => f === s || f.includes(s));
      if (idx >= 0) return features[idx];
    }
    return null;
  }

  function buildInputsFromParams(features: string[]) {
    const mapping = {
      gamma: resolveFeature(features, ["gamma", "unit weight"]),
      c: resolveFeature(features, ["c", "cohesion"]),
      phi: resolveFeature(features, ["phi", "friction angle"]),
      beta: resolveFeature(features, ["beta", "slope angle"]),
      H: resolveFeature(features, ["h", "height"]),
      ru: resolveFeature(features, ["ru", "pore pressure ratio"]),
    };
    const inputs: Record<string, number> = {};
    if (mapping.gamma) inputs[mapping.gamma] = params.gamma;
    if (mapping.c) inputs[mapping.c] = params.c;
    if (mapping.phi) inputs[mapping.phi] = params.phi;
    if (mapping.beta) inputs[mapping.beta] = params.beta;
    if (mapping.H) inputs[mapping.H] = params.H;
    if (mapping.ru) inputs[mapping.ru] = params.ru;
    return inputs;
  }

  async function run() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      if (file) fd.append("file", file);
      if (resp?.features?.length) {
        fd.append("inputs_json", JSON.stringify(buildInputsFromParams(resp.features)));
      }
      const res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/slope/predict`, {
        method: "POST",
        headers: { ...authHeaders(token) },
        body: fd,
      });
      const json = await res.json().catch(() => ({}));
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
      <div className="grid2">
        <div>
          <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Slope Stability — Stable / Failure (ML)</div>
          <div className="subtitle">Load CSV to train, then adjust parameters.</div>

          <div style={{ marginTop: 10 }}>
            <label className="label">Load CSV</label>
            <input className="input" type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </div>

          <div style={{ marginTop: 10 }}>
            <button className="btn btnPrimary" onClick={run} disabled={busy}>
              {busy ? "Running…" : "Predict"}
            </button>
          </div>

          <div style={{ marginTop: 12, display: "grid", gap: 10 }}>
            <SliderField label="H (m)" value={params.H} min={1} max={50} step={0.5} onChange={(v) => setParams({ ...params, H: v })} />
            <SliderField label="β (deg)" value={params.beta} min={5} max={80} step={0.5} onChange={(v) => setParams({ ...params, beta: v })} />
            <SliderField label="c (kPa)" value={params.c} min={1} max={200} step={0.5} onChange={(v) => setParams({ ...params, c: v })} />
            <SliderField label="φ (deg)" value={params.phi} min={5} max={60} step={0.5} onChange={(v) => setParams({ ...params, phi: v })} />
            <SliderField label="γ (kN/m³)" value={params.gamma} min={14} max={28} step={0.1} onChange={(v) => setParams({ ...params, gamma: v })} />
            <SliderField label="ru (–)" value={params.ru} min={0} max={1} step={0.01} onChange={(v) => setParams({ ...params, ru: v })} />
            <SliderField label="B (m) — sketch only" value={params.B} min={0} max={30} step={0.5} onChange={(v) => setParams({ ...params, B: v })} />
          </div>

          {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
          {resp?.prob_stable != null && (
            <div className="kpi" style={{ marginTop: 12 }}>
              <div className="kpiTitle">Prediction</div>
              <div className="kpiValue">
                {resp.prob_stable >= 0.5 ? "🟢 Stable" : "🔴 Failure"} (P(stable)={(Number(resp.prob_stable) * 100).toFixed(1)}%)
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <SlopeSketch
            H={params.H}
            beta={params.beta}
            B={params.B}
            prob={resp?.prob_stable}
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
  const [sizeBy, setSizeBy] = useState("Delay");
  const [animate, setAnimate] = useState(false);
  const [timeCutoff, setTimeCutoff] = useState<number | undefined>(undefined);

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
        const keys = Object.keys(json.points[0] ?? {});
        if (keys.includes("Delay")) {
          setColorBy("Delay");
          setSizeBy("Delay");
          const delays = json.points.map((p: any) => Number(p.Delay)).filter((v: number) => Number.isFinite(v));
          if (delays.length) {
            setTimeCutoff(Math.min(...delays));
          }
        }
      }
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!animate || !resp?.points?.length) return;
    const delays = resp.points.map((p: any) => Number(p.Delay)).filter((v: number) => Number.isFinite(v));
    if (!delays.length) return;
    const minD = Math.min(...delays);
    const maxD = Math.max(...delays);
    let t = timeCutoff ?? minD;
    const step = (maxD - minD) / 50;
    const id = setInterval(() => {
      t += step;
      if (t > maxD) t = minD;
      setTimeCutoff(t);
    }, 200);
    return () => clearInterval(id);
  }, [animate, resp, timeCutoff]);

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
          <button className="btn btnPrimary" onClick={run} disabled={busy}>{busy ? "Running…" : "Predict Delays"}</button>
        </div>
      </div>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      {resp?.points?.length ? (
        <div className="kpi" style={{ marginTop: 12 }}>
          <div className="kpiTitle">Predicted points</div>
          <div className="kpiValue">{resp.points.length}</div>
        </div>
      ) : null}
      {resp?.points?.length ? (
        <div style={{ marginTop: 12 }}>
          {(() => {
            const delays = resp.points.map((p: any) => Number(p.Delay)).filter((v: number) => Number.isFinite(v));
            const minDelay = delays.length ? Math.min(...delays) : 0;
            const maxDelay = delays.length ? Math.max(...delays) : 100;
            return (
              <>
          <div className="grid3">
            <div>
              <label className="label">Color by</label>
              <select className="input" value={colorBy} onChange={(e) => setColorBy(e.target.value)}>
                {Object.keys(resp.points[0] ?? {}).map((k) => (
                  <option key={k} value={k}>{k}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Size by</label>
              <select className="input" value={sizeBy} onChange={(e) => setSizeBy(e.target.value)}>
                {Object.keys(resp.points[0] ?? {}).map((k) => (
                  <option key={k} value={k}>{k}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Simulate blast</label>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <input type="checkbox" checked={animate} onChange={(e) => setAnimate(e.target.checked)} />
                <input
                  className="input"
                  type="range"
                  min={minDelay}
                  max={maxDelay}
                  step="1"
                  value={timeCutoff ?? minDelay}
                  onChange={(e) => setTimeCutoff(Number(e.target.value))}
                />
              </div>
            </div>
          </div>
          <PlanView points={resp.points} colorBy={colorBy} sizeBy={sizeBy} timeCutoff={animate ? timeCutoff : undefined} />
          <div style={{ marginTop: 10 }}>
            <button className="btn" onClick={() => downloadCsv(resp.points, Object.keys(resp.points[0] ?? {}), "delay_predictions.csv")}>
              Export CSV
            </button>
          </div>
              </>
            );
          })()}
        </div>
      ) : null}
    </div>
  );
}

function FeaturePanel({
  apiBaseUrl,
  token,
  dataset,
}: {
  apiBaseUrl: string;
  token: string;
  dataset: { file?: File | null; rows: Array<Record<string, any>>; columns: string[] };
}) {
  const [resp, setResp] = useState<any>(null);
  const [pca, setPca] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [topK, setTopK] = useState(12);
  const [msg, setMsg] = useState("Tip: Load/confirm dataset in Data Manager. Inputs = first N−3 if names can't be mapped.");

  async function runImportance() {
    if (!apiBaseUrl) return;
    setBusy(true);
    setErr(null);
    try {
      let res: Response;
      if (dataset?.file || dataset?.rows?.length) {
        const fd = new FormData();
        if (dataset.file) {
          fd.append("file", dataset.file);
        } else {
          const blob = datasetToCsvBlob(dataset);
          if (blob) fd.append("file", blob, "dataset.csv");
        }
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
      setMsg(
        `Dataset: ${json?.note ?? ""}\nRows used: ${json?.rows_used ?? "?"} | Inputs: ${json?.inputs?.length ?? "?"} | Outputs: ${json?.outputs?.length ?? "?"}\nPlotted top-${json?.top_k ?? topK} features for each output.`
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
      if (dataset?.file || dataset?.rows?.length) {
        const fd = new FormData();
        if (dataset.file) {
          fd.append("file", dataset.file);
        } else {
          const blob = datasetToCsvBlob(dataset);
          if (blob) fd.append("file", blob, "dataset.csv");
        }
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
        `PCA based on dataset: ${json?.note ?? ""}\nShape used: ${json?.rows_used ?? "?"} rows × ${json?.inputs?.length ?? "?"} inputs`
      );
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Feature Importance & PCA</div>
      <div className="subtitle">Mirror of the local RF importance and PCA analysis.</div>
      <div style={{ marginTop: 10, display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <button className="btn btnPrimary" onClick={runImportance} disabled={busy}>
          {busy ? "Loading…" : "Compute RF Importance"}
        </button>
        <button className="btn" onClick={runPca} disabled={busy}>
          {busy ? "Loading…" : "Run PCA Analysis"}
        </button>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className="label">Top-K features</span>
          <input
            className="input"
            type="number"
            min={5}
            max={30}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            style={{ width: 80 }}
          />
        </div>
      </div>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      <pre style={{ ...pre, marginTop: 10 }}>{msg}</pre>
      {resp?.feature_importance && (
        <div style={{ marginTop: 12 }}>
          <FeatureImportanceCharts data={resp.feature_importance} />
        </div>
      )}
      {pca?.explained_variance_ratio && (
        <div style={{ marginTop: 12 }}>
          <PCAViz pca={pca} />
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

  function resetMedians() {
    if (!resp?.feature_stats) return;
    const next: Record<string, number> = {};
    Object.keys(resp.feature_stats).forEach((k) => {
      next[k] = resp.feature_stats[k].median;
    });
    setInputs(next);
  }

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

function PlanView({
  points,
  colorBy,
  sizeBy,
  timeCutoff,
}: {
  points: Array<Record<string, any>>;
  colorBy: string;
  sizeBy: string;
  timeCutoff?: number;
}) {
  const w = 620;
  const h = 360;
  const xs = points.map((p) => Number(p.X));
  const ys = points.map((p) => Number(p.Y));
  const cs = points.map((p) => Number(p[colorBy]));
  const ss = points.map((p) => Number(p[sizeBy]));
  const ds = points.map((p) => Number(p.Delay));
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const cmin = Math.min(...cs.filter((v) => Number.isFinite(v)));
  const cmax = Math.max(...cs.filter((v) => Number.isFinite(v)));
  const smin = Math.min(...ss.filter((v) => Number.isFinite(v)));
  const smax = Math.max(...ss.filter((v) => Number.isFinite(v)));
  const norm = (v: number, a: number, b: number) => (b - a === 0 ? 0.5 : (v - a) / (b - a));
  const cutoff = timeCutoff ?? Number.POSITIVE_INFINITY;
  return (
    <div style={{ marginTop: 12 }}>
      <div className="label">Plan View (color by {colorBy}, size by {sizeBy})</div>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 12 }}>
        {points.slice(0, 800).map((p, i) => {
          const delay = Number(p.Delay);
          if (Number.isFinite(delay) && delay > cutoff) return null;
          const x = 10 + norm(p.X, xmin, xmax) * (w - 20);
          const y = 10 + (1 - norm(p.Y, ymin, ymax)) * (h - 20);
          const t = norm(Number(p[colorBy]) || 0, cmin, cmax);
          const s = norm(Number(p[sizeBy]) || 0, smin, smax);
          const color = `hsl(${(1 - t) * 220}, 80%, 60%)`;
          return <circle key={i} cx={x} cy={y} r={2 + s * 4} fill={color} />;
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

function SurfaceIsoPlot({ gridX, gridY, Z }: { gridX: number[]; gridY: number[]; Z: number[][] }) {
  if (!gridX?.length || !gridY?.length || !Z?.length) return null;
  const w = 620;
  const h = 360;
  const maxZ = Math.max(...Z.flat().filter((v) => Number.isFinite(v)));
  const minZ = Math.min(...Z.flat().filter((v) => Number.isFinite(v)));
  const scale = 0.6;
  const proj = (x: number, y: number, z: number) => {
    const px = (x - y) * scale;
    const py = (x + y) * scale * 0.5 - z * scale * 0.4;
    return { x: px, y: py };
  };
  const cells: Array<{ points: string; color: string }> = [];
  const allX: number[] = [];
  const allY: number[] = [];
  for (let i = 0; i < gridX.length - 1; i++) {
    for (let j = 0; j < gridY.length - 1; j++) {
      const z00 = Z[j]?.[i];
      const z10 = Z[j]?.[i + 1];
      const z01 = Z[j + 1]?.[i];
      const z11 = Z[j + 1]?.[i + 1];
      if (![z00, z10, z01, z11].every((v) => Number.isFinite(v))) continue;
      const zAvg = (z00 + z10 + z01 + z11) / 4;
      const t = (zAvg - minZ) / (maxZ - minZ || 1);
      const color = `hsl(${220 - 200 * t}, 80%, 55%)`;
      const p00 = proj(gridX[i], gridY[j], z00);
      const p10 = proj(gridX[i + 1], gridY[j], z10);
      const p11 = proj(gridX[i + 1], gridY[j + 1], z11);
      const p01 = proj(gridX[i], gridY[j + 1], z01);
      const pts = [p00, p10, p11, p01];
      pts.forEach((p) => {
        allX.push(p.x);
        allY.push(p.y);
      });
      cells.push({ points: pts.map((p) => `${p.x},${p.y}`).join(" "), color });
    }
  }
  const xmin = Math.min(...allX);
  const xmax = Math.max(...allX);
  const ymin = Math.min(...allY);
  const ymax = Math.max(...allY);
  const sx = (x: number) => 20 + ((x - xmin) / (xmax - xmin || 1)) * (w - 40);
  const sy = (y: number) => h - 20 - ((y - ymin) / (ymax - ymin || 1)) * (h - 40);
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 12 }}>
      {cells.map((c, i) => (
        <polygon
          key={i}
          points={c.points.split(" ").map((pt) => {
            const [x, y] = pt.split(",").map(Number);
            return `${sx(x)},${sy(y)}`;
          }).join(" ")}
          fill={c.color}
          opacity={0.85}
          stroke="#ffffff"
          strokeWidth={0.2}
        />
      ))}
    </svg>
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

function PCAViz({ pca }: { pca: any }) {
  const vr = pca?.explained_variance_ratio ?? [];
  const points = pca?.points ?? [];
  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div className="card">
        <div className="label">PCA — Explained Variance</div>
        <BarChart labels={vr.map((_: number, i: number) => `PC${i + 1}`)} values={vr.map((v: number) => v * 100)} />
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
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 12 }}>
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
  const w = 620;
  const h = 360;
  const betaRad = (beta * Math.PI) / 180;
  const run = H / Math.max(Math.tan(betaRad), 1e-3);
  const toeX = B + run;
  const pad = 20;
  const scale = Math.min((w - 2 * pad) / (toeX * 1.2), (h - 2 * pad) / (H * 1.2));
  const sx = (x: number) => pad + x * scale;
  const sy = (y: number) => h - pad - y * scale;
  const x0 = 0;
  const y0 = 0;
  const x1 = 0;
  const y1 = H;
  const x2 = B;
  const y2 = H;
  const x3 = toeX;
  const y3 = 0;
  const label = prob == null ? "—" : prob >= 0.5 ? `Stable (P=${prob.toFixed(2)})` : `Failure (P=${prob.toFixed(2)})`;
  const col = prob == null ? "#64748b" : prob >= 0.5 ? "#22c55e" : "#ef4444";

  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} style={{ background: "var(--panel)", borderRadius: 12 }}>
      <polygon
        points={`${sx(x0)},${sy(y0)} ${sx(x1)},${sy(y1)} ${sx(x2)},${sy(y2)} ${sx(x3)},${sy(y3)}`}
        fill="#d6d7db"
        opacity="0.7"
        stroke="#0f172a"
        strokeWidth="2"
      />
      <line x1={sx(0)} y1={sy(0)} x2={sx(toeX * 1.06)} y2={sy(0)} stroke="#0f172a" strokeWidth="1.5" />
      <line x1={sx(0)} y1={sy(0)} x2={sx(0)} y2={sy(H * 1.05)} stroke="#0f172a" strokeWidth="1.5" />
      <text x={sx(0) + 10} y={sy(H / 2)} fill="#1f77b4" fontSize="10">
        H = {H.toFixed(1)} m
      </text>
      <text x={sx(toeX - run / 2)} y={sy(H) - 6} fill="#555" fontSize="10">
        B = {B.toFixed(1)} m
      </text>
      <rect x={sx(toeX * 0.55)} y={sy(H * 1.15)} width={160} height={24} fill={col} rx={6} />
      <text x={sx(toeX * 0.55) + 8} y={sy(H * 1.15) + 16} fill="#fff" fontSize="12">
        {label}
      </text>
    </svg>
  );
}

function ParamPanel({
  apiBaseUrl,
  token,
  dataset,
}: {
  apiBaseUrl: string;
  token: string;
  dataset: { file?: File | null; rows: Array<Record<string, any>>; columns: string[] };
}) {
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
  const [msg, setMsg] = useState(
    "Creates an optimisation surface by minimising/maximising the chosen output.\n" +
      "Other inputs are optimised via random sampling within observed bounds.\n\n" +
      "Use 'Goal Seek' to set a target output and search for an input recipe."
  );

  useEffect(() => {
    if (!apiBaseUrl) return;
    (async () => {
      let res: Response;
      if (dataset?.file || dataset?.rows?.length) {
        const fd = new FormData();
        if (dataset.file) {
          fd.append("file", dataset.file);
        } else {
          const blob = datasetToCsvBlob(dataset);
          if (blob) fd.append("file", blob, "dataset.csv");
        }
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
      let res: Response;
      if (dataset?.file || dataset?.rows?.length) {
        const fd = new FormData();
        if (dataset.file) {
          fd.append("file", dataset.file);
        } else {
          const blob = datasetToCsvBlob(dataset);
          if (blob) fd.append("file", blob, "dataset.csv");
        }
        fd.append("payload_json", JSON.stringify({ output, x1, x2, objective, grid, samples }));
        res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/param/surface/upload`, {
          method: "POST",
          headers: { ...authHeaders(token) },
          body: fd,
        });
      } else {
        res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/param/surface`, {
          method: "POST",
          headers: { "content-type": "application/json", ...authHeaders(token) },
          body: JSON.stringify({ output, x1, x2, objective, grid, samples }),
        });
      }
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Surface failed");
      setResp(json);
      setMsg(
        `Optimised surface built.\nOutput: ${json.output}\nAxes: ${json.x1}, ${json.x2}\nBest value: ${formatNum(json?.best?.value)}`
      );
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
      let res: Response;
      if (dataset?.file || dataset?.rows?.length) {
        const fd = new FormData();
        if (dataset.file) {
          fd.append("file", dataset.file);
        } else {
          const blob = datasetToCsvBlob(dataset);
          if (blob) fd.append("file", blob, "dataset.csv");
        }
        fd.append("payload_json", JSON.stringify({ output, target }));
        res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/param/goal-seek/upload`, {
          method: "POST",
          headers: { ...authHeaders(token) },
          body: fd,
        });
      } else {
        res = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/param/goal-seek`, {
          method: "POST",
          headers: { "content-type": "application/json", ...authHeaders(token) },
          body: JSON.stringify({ output, target }),
        });
      }
      const json = await res.json();
      if (!res.ok || json?.error) throw new Error(json?.error ?? "Goal seek failed");
      setGoal(json);
      setMsg(
        `Goal seek target: ${formatNum(json?.target)}\nPredicted: ${formatNum(json?.best?.predicted)}`
      );
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Parameter Optimisation</div>
      <div className="subtitle">Mirror of the local optimisation surface + goal seek.</div>

      <div style={{ marginTop: 10 }} className="grid3">
        <div>
          <label className="label">Output</label>
          <select className="input" value={output} onChange={(e) => setOutput(e.target.value)}>
            {(meta?.outputs ?? []).map((o: string) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="label">Input 1 (X-axis)</label>
          <select className="input" value={x1} onChange={(e) => setX1(e.target.value)}>
            {(meta?.inputs ?? []).map((o: string) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="label">Input 2 (Y-axis)</label>
          <select className="input" value={x2} onChange={(e) => setX2(e.target.value)}>
            {(meta?.inputs ?? []).map((o: string) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </div>
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
        <button className="btn btnPrimary" onClick={runSurface} disabled={busy}>
          {busy ? "Running…" : "Optimise & Plot Surface"}
        </button>
        <button className="btn" onClick={() => downloadCsv(resp?.Z?.flatMap((row: any, i: number) => row.map((z: number, j: number) => ({
          [resp?.x1]: resp?.grid_x?.[i],
          [resp?.x2]: resp?.grid_y?.[j],
          [resp?.output]: z,
        }))) ?? [], resp?.x1 ? [resp.x1, resp.x2, resp.output] : [], "param_surface.csv")} disabled={!resp?.Z}>
          Export surface CSV
        </button>
        <select className="input" value={objective} onChange={(e) => setObjective(e.target.value as any)} style={{ width: 120 }}>
          <option value="max">Maximise</option>
          <option value="min">Minimise</option>
        </select>
        <input className="input" type="number" value={grid} onChange={(e) => setGrid(Number(e.target.value))} style={{ width: 120 }} />
        <input className="input" type="number" value={samples} onChange={(e) => setSamples(Number(e.target.value))} style={{ width: 140 }} />
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <div className="label">Inverse Design (Goal Seek)</div>
        <div className="grid3" style={{ marginTop: 8 }}>
          <input className="input" type="number" value={target} onChange={(e) => setTarget(Number(e.target.value))} placeholder="Target output" />
          <button className="btn" onClick={runGoal} disabled={busy}>
            {busy ? "Running…" : "Run Goal Seek (All Inputs)"}
          </button>
        </div>
      </div>

      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}

      <div className="card" style={{ marginTop: 12 }}>
        <div className="label">Message</div>
        <pre style={pre}>{msg}</pre>
      </div>

      {resp?.Z && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="label">Optimisation Surface</div>
          <SurfaceIsoPlot gridX={resp.grid_x} gridY={resp.grid_y} Z={resp.Z} />
        </div>
      )}

      {resp?.best?.inputs && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="label">Optimised Other Inputs</div>
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
  const [objectiveMode, setObjectiveMode] = useState("Min Cost + Frag + PPV/Air");

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
      <div className="subtitle">Matches the local CTk layout and naming.</div>

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
  background: "#f8fafc",
  color: "#111827",
  borderRadius: 12,
  padding: 14,
  overflow: "auto",
  maxHeight: 420,
} as const;

