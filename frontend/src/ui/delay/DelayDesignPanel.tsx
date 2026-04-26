import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import sampleCsv from "../../data/sample.csv?raw";
import { parseBlastCsv, rowsToBlastHoles } from "../../lib/csvParser";
import { buildDelayAssignmentCsv, downloadTextFile, exportWarnings } from "../../lib/exportCsv";
import { buildProjectJson } from "../../lib/exportJson";
import { estimatePerformance, summarizePerformance } from "../../lib/performanceIndicators";
import type { PerformanceSummary } from "../../lib/performanceIndicators";
import { buildPrintableReport, openPrintableReport } from "../../lib/reportGenerator";
import { defaultRowTolerance } from "../../lib/rowDetection";
import { simulationState, uniqueDelayTimes } from "../../lib/simulation";
import { assignTiming } from "../../lib/timingAlgorithms";
import { designCompleteness, validateBlastDesign } from "../../lib/validation";
import type { BlastHole, BlastProject, ColorMode, ColumnMapping, TimingLine, TimingPattern, TimingSettings, ValidationIssue } from "../../types/blast";
import { DEFAULT_TIMING_SETTINGS, TIMING_PATTERN_LABELS } from "../../types/blast";

const STORAGE_KEY = "blast_timing_studio_project_v1";
const SAFETY_DISCLAIMER =
  "This tool provides planning and simulation support only. Real blast performance depends on geology, burden, spacing, explosive type, charge distribution, confinement, stemming, timing accuracy, initiation system, and site-specific conditions. All designs must be reviewed and approved by qualified blasting personnel.";

function formatNum(value: unknown, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return Math.abs(n) >= 1000 ? n.toFixed(0) : n.toFixed(digits);
}

function safeName(name: string) {
  return (name || "blast_timing_design").trim().replace(/[^a-z0-9_-]+/gi, "_").replace(/^_+|_+$/g, "") || "blast_timing_design";
}

function createProject(overrides: Partial<BlastProject> = {}): BlastProject {
  const now = new Date().toISOString();
  return {
    projectName: "Blast Timing Design",
    importedFileName: "",
    holes: [],
    timingPattern: "rowByRow",
    settings: DEFAULT_TIMING_SETTINGS,
    createdAt: now,
    updatedAt: now,
    ...overrides,
  };
}

function kpi(title: string, value: string | number) {
  return (
    <div className="kpi">
      <div className="kpiTitle">{title}</div>
      <div className="kpiValue">{value}</div>
    </div>
  );
}

function CollapsibleSection({
  title,
  collapsed,
  onToggle,
  children,
}: {
  title: string;
  collapsed: boolean;
  onToggle: () => void;
  children: ReactNode;
}) {
  return (
    <div className="card" style={{ padding: 12 }}>
      <button
        type="button"
        className="btn"
        onClick={onToggle}
        style={{ width: "100%", display: "flex", justifyContent: "space-between", alignItems: "center" }}
      >
        <span>{title}</span>
        <span>{collapsed ? "Show" : "Minimise"}</span>
      </button>
      {!collapsed ? <div style={{ marginTop: 10 }}>{children}</div> : null}
    </div>
  );
}

export function DelayDesignPanel() {
  const [project, setProject] = useState<BlastProject>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) return { ...createProject(), ...JSON.parse(saved) };
    } catch {
      // Ignore corrupt local drafts and start fresh.
    }
    return createProject();
  });
  const [columns, setColumns] = useState<string[]>([]);
  const [rawRows, setRawRows] = useState<Array<Record<string, string>>>([]);
  const [mapping, setMapping] = useState<ColumnMapping>({});
  const [parseIssues, setParseIssues] = useState<ValidationIssue[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [selectedHoleId, setSelectedHoleId] = useState<string | null>(null);
  const [colorMode, setColorMode] = useState<ColorMode>("delay");
  const [showLabels, setShowLabels] = useState(true);
  const [showOrder, setShowOrder] = useState(false);
  const [showFired, setShowFired] = useState(true);
  const [showUnfired, setShowUnfired] = useState(true);
  const [showWavefront, setShowWavefront] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [exportMessage, setExportMessage] = useState("");
  const [focusCanvas, setFocusCanvas] = useState(false);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({
    setup: false,
    timing: false,
    simulation: true,
    export: true,
    analysis: true,
    details: false,
    validation: false,
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...project, updatedAt: new Date().toISOString() }));
  }, [project]);

  const displayHoles = useMemo(() => estimatePerformance(project.holes), [project.holes]);
  const delayTimes = useMemo(() => uniqueDelayTimes(displayHoles), [displayHoles]);
  const currentTime = delayTimes.length ? delayTimes[Math.max(0, Math.min(stepIndex, delayTimes.length - 1))] : undefined;
  const sim = useMemo(() => simulationState(displayHoles, currentTime), [displayHoles, currentTime]);
  const performanceSummary = useMemo(() => summarizePerformance(displayHoles), [displayHoles]);
  const issues = useMemo(
    () => [...parseIssues, ...validateBlastDesign(project.holes, project.settings, project.holes.some((hole) => Number.isFinite(hole.delayMs)))],
    [parseIssues, project.holes, project.settings]
  );
  const selectedHole = displayHoles.find((hole) => hole.id === selectedHoleId) ?? null;
  const assignedCount = project.holes.filter((hole) => Number.isFinite(hole.delayMs)).length;
  const delays = project.holes.map((hole) => hole.delayMs).filter((delay): delay is number => Number.isFinite(delay));
  const totalCharge = project.holes.reduce((sum, hole) => sum + (Number.isFinite(hole.charge) ? (hole.charge as number) : 0), 0);
  const errorCount = issues.filter((issue) => issue.severity === "error").length;
  const warningCount = issues.filter((issue) => issue.severity === "warning").length;
  const selectedForLine = selectedIds
    .slice(-2)
    .map((id) => project.holes.find((hole) => hole.id === id))
    .filter(Boolean) as BlastHole[];
  const selectedLine: TimingLine | undefined =
    selectedForLine.length >= 2 ? { start: selectedForLine[0], end: selectedForLine[1] } : undefined;

  useEffect(() => {
    if (!playing || delayTimes.length <= 1) return;
    const id = window.setInterval(() => {
      setStepIndex((prev) => (prev >= delayTimes.length - 1 ? 0 : prev + 1));
    }, Math.max(90, 650 / Math.max(0.25, speed)));
    return () => window.clearInterval(id);
  }, [playing, speed, delayTimes.length]);

  function updateProject(patch: Partial<BlastProject>) {
    setProject((prev) => ({ ...prev, ...patch, updatedAt: new Date().toISOString() }));
  }

  function updateSettings(patch: Partial<TimingSettings>) {
    setProject((prev) => ({ ...prev, settings: { ...prev.settings, ...patch }, updatedAt: new Date().toISOString() }));
  }

  function importCsvText(text: string, filename: string) {
    const parsed = parseBlastCsv(text);
    const rowTolerance = defaultRowTolerance(parsed.holes);
    setColumns(parsed.columns);
    setRawRows(parsed.rows);
    setMapping(parsed.mapping);
    setParseIssues(parsed.issues);
    setSelectedIds([]);
    setSelectedHoleId(null);
    setStepIndex(0);
    updateProject({
      importedFileName: filename,
      holes: parsed.holes,
      settings: { ...project.settings, rowTolerance },
    });
  }

  async function handleFile(file: File | null) {
    if (!file) return;
    const text = await file.text();
    importCsvText(text, file.name);
  }

  function applyMapping(nextMapping = mapping) {
    const converted = rowsToBlastHoles(rawRows, nextMapping);
    setMapping(nextMapping);
    setParseIssues(converted.issues);
    updateProject({ holes: converted.holes });
  }

  function assignDelays() {
    const selected = selectedHole ? { x: selectedHole.x, y: selectedHole.y } : undefined;
    const holes = assignTiming(project.holes, {
      pattern: project.timingPattern,
      settings: project.settings,
      selectedIds,
      initiationPoint: selected,
      initiationLine: selectedLine,
      vWidth: "medium",
    });
    updateProject({ holes });
    setStepIndex(0);
    setPlaying(false);
  }

  function toggleSelected(id: string) {
    setSelectedHoleId(id);
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]));
  }

  function editSelectedDelay(delayText: string) {
    const delay = delayText.trim() === "" ? undefined : Number(delayText);
    updateProject({
      holes: project.holes.map((hole) =>
        hole.id === selectedHoleId
          ? { ...hole, delayMs: Number.isFinite(delay) ? delay : undefined, timingGroup: hole.timingGroup ?? "Manual edit" }
          : hole
      ),
    });
  }

  function clearTiming() {
    updateProject({
      holes: project.holes.map((hole) => ({ ...hole, delayMs: undefined, firingOrder: undefined, timingGroup: undefined })),
    });
    setStepIndex(0);
    setPlaying(false);
  }

  function exportCsv() {
    const exportIssues = validateBlastDesign(project.holes, project.settings, true);
    const warning = exportWarnings(exportIssues);
    const csv = buildDelayAssignmentCsv(displayHoles, project.timingPattern);
    downloadTextFile(csv, `${safeName(project.projectName)}_delay_assignment_planning_draft.csv`, "text/csv;charset=utf-8;");
    setExportMessage(warning || "Delay assignment CSV exported as a Planning/Simulation Draft.");
  }

  function exportJson() {
    downloadTextFile(buildProjectJson({ ...project, holes: displayHoles }), `${safeName(project.projectName)}_project.json`, "application/json;charset=utf-8;");
    setExportMessage("Project JSON exported.");
  }

  function exportReport() {
    const html = buildPrintableReport({ ...project, holes: displayHoles }, issues);
    if (!openPrintableReport(html)) {
      downloadTextFile(html, `${safeName(project.projectName)}_delay_report_planning_draft.html`, "text/html;charset=utf-8;");
      setExportMessage("Printable report downloaded because the popup was blocked.");
    } else {
      setExportMessage("Printable report opened in a new tab.");
    }
  }

  const completeness = designCompleteness(project.holes);
  const toggleCollapsed = (key: string) => setCollapsed((prev) => ({ ...prev, [key]: !prev[key] }));

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Delay Design & Simulation</div>
            <div className="subtitle">Integrated planning module for CSV import, transparent delay assignment, sequence simulation, and draft export.</div>
          </div>
          <div className="pill">Planning/Simulation Draft</div>
        </div>
        <div className="error" style={{ marginTop: 10 }}>{SAFETY_DISCLAIMER}</div>
      </div>

      <div className="grid3">
        {kpi("Holes imported", project.holes.length)}
        {kpi("Assigned delays", `${assignedCount}/${project.holes.length || 0}`)}
        {kpi("Delay range", delays.length ? `${formatNum(Math.min(...delays), 0)}-${formatNum(Math.max(...delays), 0)} ms` : "-")}
        {kpi("Duration", delays.length ? `${formatNum(Math.max(...delays) - Math.min(...delays), 0)} ms` : "-")}
        {kpi("Warnings / errors", `${warningCount} / ${errorCount}`)}
        {kpi("Pattern", TIMING_PATTERN_LABELS[project.timingPattern])}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: focusCanvas ? "1fr" : "minmax(0, 1fr) minmax(280px, 330px)", gap: 12, alignItems: "start" }}>
        <div style={{ display: "grid", gap: 12 }}>
          <BlastCanvas
            holes={displayHoles}
            pattern={project.timingPattern}
            selectedIds={selectedIds}
            selectedHoleId={selectedHoleId}
            selectedLine={selectedLine}
            colorMode={colorMode}
            currentTime={currentTime}
            showLabels={showLabels}
            showOrder={showOrder}
            showFired={showFired}
            showUnfired={showUnfired}
            showWavefront={showWavefront}
            focusCanvas={focusCanvas}
            onSelect={toggleSelected}
            onFocusToggle={() => setFocusCanvas((value) => !value)}
          />
          <SimulationControls
            playing={playing}
            speed={speed}
            stepIndex={stepIndex}
            delayTimes={delayTimes}
            showLabels={showLabels}
            showOrder={showOrder}
            showFired={showFired}
            showUnfired={showUnfired}
            showWavefront={showWavefront}
            compact={focusCanvas}
            onPlaying={setPlaying}
            onSpeed={setSpeed}
            onStep={setStepIndex}
            onShowLabels={setShowLabels}
            onShowOrder={setShowOrder}
            onShowFired={setShowFired}
            onShowUnfired={setShowUnfired}
            onShowWavefront={setShowWavefront}
          />
          {!focusCanvas && (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 12 }}>
                <CollapsibleSection title="Project Setup" collapsed={collapsed.setup} onToggle={() => toggleCollapsed("setup")}>
                  <ProjectSetup
                    project={project}
                    columns={columns}
                    mapping={mapping}
                    onProjectName={(projectName) => updateProject({ projectName })}
                    onFile={handleFile}
                    onLoadSample={() => importCsvText(sampleCsv, "sample.csv")}
                    onMappingChange={(next) => setMapping(next)}
                    onApplyMapping={() => applyMapping()}
                  />
                </CollapsibleSection>
                <CollapsibleSection title="Timing Pattern & Delays" collapsed={collapsed.timing} onToggle={() => toggleCollapsed("timing")}>
                  <TimingControls
                    pattern={project.timingPattern}
                    settings={project.settings}
                    selectedCount={selectedIds.length}
                    hasLine={!!selectedLine}
                    onPattern={(timingPattern) => updateProject({ timingPattern })}
                    onSettings={updateSettings}
                    onAssign={assignDelays}
                    onClear={clearTiming}
                  />
                </CollapsibleSection>
                <CollapsibleSection title="Analysis Indicators" collapsed={collapsed.analysis} onToggle={() => toggleCollapsed("analysis")}>
                  <AnalysisDashboard holes={displayHoles} completeness={completeness} sim={sim} performanceSummary={performanceSummary} />
                </CollapsibleSection>
                <CollapsibleSection title="Export Drafts" collapsed={collapsed.export} onToggle={() => toggleCollapsed("export")}>
                  <ExportPanel
                    canExport={project.holes.length > 0}
                    hasDelays={assignedCount > 0}
                    message={exportMessage}
                    onCsv={exportCsv}
                    onJson={exportJson}
                    onReport={exportReport}
                  />
                </CollapsibleSection>
              </div>
            </>
          )}
        </div>

        {!focusCanvas && (
          <div style={{ display: "grid", gap: 12, position: "sticky", top: 92, alignSelf: "start" }}>
            <div className="card">
              <div className="sectionTitle">View Options</div>
              <label className="label">Colour holes by</label>
              <select className="input" value={colorMode} onChange={(e) => setColorMode(e.target.value as ColorMode)}>
                <option value="delay">Delay</option>
                <option value="row">Row</option>
                <option value="charge">Charge</option>
                <option value="depth">Depth</option>
                <option value="group">Timing group</option>
                <option value="fragmentation">Fragmentation</option>
                <option value="ppv">Ground vibration / PPV</option>
                <option value="airblast">Airblast</option>
                <option value="flyrock">Flyrock risk</option>
              </select>
              <div className="subtitle" style={{ marginTop: 8 }}>
                Click holes to select them. For directional-from-line, select two holes; the last two selected holes define the initiation line shown on the plan.
              </div>
            </div>
            <CollapsibleSection title="Selected Hole" collapsed={collapsed.details} onToggle={() => toggleCollapsed("details")}>
              <HoleDetailsPanel hole={selectedHole} onDelayChange={editSelectedDelay} onClear={() => editSelectedDelay("")} />
            </CollapsibleSection>
            <CollapsibleSection title={`Validation (${warningCount}/${errorCount})`} collapsed={collapsed.validation} onToggle={() => toggleCollapsed("validation")}>
              <ValidationPanel issues={issues} />
            </CollapsibleSection>
          </div>
        )}
      </div>
    </div>
  );
}

function ProjectSetup({
  project,
  columns,
  mapping,
  onProjectName,
  onFile,
  onLoadSample,
  onMappingChange,
  onApplyMapping,
}: {
  project: BlastProject;
  columns: string[];
  mapping: ColumnMapping;
  onProjectName: (value: string) => void;
  onFile: (file: File | null) => void;
  onLoadSample: () => void;
  onMappingChange: (mapping: ColumnMapping) => void;
  onApplyMapping: () => void;
}) {
  const select = (field: keyof ColumnMapping, label: string) => (
    <div>
      <label className="label">{label}</label>
      <select className="input" value={mapping[field] ?? ""} onChange={(e) => onMappingChange({ ...mapping, [field]: e.target.value || undefined })}>
        <option value="">Not mapped</option>
        {columns.map((column) => <option key={column} value={column}>{column}</option>)}
      </select>
    </div>
  );
  return (
    <div className="card">
      <div className="sectionTitle">Project Setup</div>
      <label className="label">Project name</label>
      <input className="input" value={project.projectName} onChange={(e) => onProjectName(e.target.value)} />
      <label className="label" style={{ marginTop: 10 }}>Import CSV</label>
      <input className="input" type="file" accept=".csv" onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
      <button className="btn" style={{ marginTop: 8 }} onClick={onLoadSample}>Load sample CSV</button>
      {columns.length ? (
        <div style={{ marginTop: 12, display: "grid", gap: 8 }}>
          <div className="subtitle">Imported: {project.importedFileName || "CSV"}. Confirm mapping if needed.</div>
          <div className="grid2">
            {select("id", "Hole ID")}
            {select("x", "X")}
            {select("y", "Y")}
            {select("z", "Z")}
            {select("depth", "Depth")}
            {select("charge", "Charge")}
          </div>
          <button className="btn" onClick={onApplyMapping}>Apply column mapping</button>
        </div>
      ) : null}
    </div>
  );
}

function TimingControls({
  pattern,
  settings,
  selectedCount,
  hasLine,
  onPattern,
  onSettings,
  onAssign,
  onClear,
}: {
  pattern: TimingPattern;
  settings: TimingSettings;
  selectedCount: number;
  hasLine: boolean;
  onPattern: (pattern: TimingPattern) => void;
  onSettings: (settings: Partial<TimingSettings>) => void;
  onAssign: () => void;
  onClear: () => void;
}) {
  const num = (label: string, key: keyof TimingSettings) => (
    <div>
      <label className="label">{label}</label>
      <input className="input" type="number" value={settings[key] as number} onChange={(e) => onSettings({ [key]: Number(e.target.value) } as Partial<TimingSettings>)} />
    </div>
  );
  return (
    <div className="card">
      <div className="sectionTitle">Pattern Selection & Timing</div>
      <label className="label">Timing pattern</label>
      <select className="input" value={pattern} onChange={(e) => onPattern(e.target.value as TimingPattern)}>
        {Object.entries(TIMING_PATTERN_LABELS).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
      </select>
      <div className="grid2" style={{ marginTop: 10 }}>
        {num("Start delay ms", "startDelayMs")}
        {num("In-row delay ms", "inRowDelayMs")}
        {num("Row-to-row delay ms", "rowDelayMs")}
        {num("Row tolerance", "rowTolerance")}
        {num("Minimum delay", "minDelayMs")}
        {num("Maximum delay", "maxDelayMs")}
        {num("Rounding increment", "delayIncrementMs")}
        <div>
          <label className="label">Direction</label>
          <select className="input" value={settings.direction} onChange={(e) => onSettings({ direction: e.target.value as TimingSettings["direction"] })}>
            <option value="leftToRight">Left to right</option>
            <option value="rightToLeft">Right to left</option>
            <option value="bottomToTop">Bottom to top</option>
            <option value="topToBottom">Top to bottom</option>
          </select>
        </div>
      </div>
      <div style={{ display: "grid", gap: 6, marginTop: 10 }}>
        <label className="label"><input type="checkbox" checked={settings.reverseOrder} onChange={(e) => onSettings({ reverseOrder: e.target.checked })} /> Reverse firing order</label>
        <label className="label"><input type="checkbox" checked={settings.applyRounding} onChange={(e) => onSettings({ applyRounding: e.target.checked })} /> Apply delay rounding</label>
      </div>
      <div className="subtitle" style={{ marginTop: 8 }}>
        {selectedCount} selected hole(s). Manual timing uses the selected order. Directional-from-line {hasLine ? "will use the highlighted selected line." : "works best after selecting two holes to define the line."}
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
        <button className="btn btnPrimary" onClick={onAssign}>Assign delays</button>
        <button className="btn" onClick={onClear}>Reset timing</button>
      </div>
    </div>
  );
}

function SimulationControls(props: {
  playing: boolean;
  speed: number;
  stepIndex: number;
  delayTimes: number[];
  showLabels: boolean;
  showOrder: boolean;
  showFired: boolean;
  showUnfired: boolean;
  showWavefront: boolean;
  compact?: boolean;
  onPlaying: (value: boolean) => void;
  onSpeed: (value: number) => void;
  onStep: (value: number) => void;
  onShowLabels: (value: boolean) => void;
  onShowOrder: (value: boolean) => void;
  onShowFired: (value: boolean) => void;
  onShowUnfired: (value: boolean) => void;
  onShowWavefront: (value: boolean) => void;
}) {
  const current = props.delayTimes[props.stepIndex];
  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        <div>
          <div className="sectionTitle">Simulation Controls</div>
          <div className="subtitle">{props.delayTimes.length ? `Step ${props.stepIndex + 1}/${props.delayTimes.length} at ${formatNum(current, 0)} ms` : "Assign delays to enable simulation."}</div>
        </div>
        {props.compact ? <div className="pill">Focus mode</div> : null}
      </div>
      <input className="input" type="range" min={0} max={Math.max(0, props.delayTimes.length - 1)} value={Math.min(props.stepIndex, Math.max(0, props.delayTimes.length - 1))} onChange={(e) => props.onStep(Number(e.target.value))} />
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 8, alignItems: "center" }}>
        <button className="btn btnPrimary" onClick={() => props.onPlaying(true)} disabled={!props.delayTimes.length || props.playing}>Play</button>
        <button className="btn" onClick={() => props.onPlaying(false)}>Pause</button>
        <button className="btn" onClick={() => { props.onPlaying(false); props.onStep(0); }}>Reset</button>
        <button className="btn" onClick={() => props.onStep(Math.min(props.stepIndex + 1, Math.max(0, props.delayTimes.length - 1)))}>Step next</button>
        <label className="label" style={{ marginLeft: 4 }}>Speed</label>
        <select className="input" style={{ width: 120 }} value={props.speed} onChange={(e) => props.onSpeed(Number(e.target.value))}>
          {[0.25, 0.5, 1, 2, 5].map((speed) => <option key={speed} value={speed}>{speed}x</option>)}
        </select>
      </div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 10 }}>
        <label className="label"><input type="checkbox" checked={props.showLabels} onChange={(e) => props.onShowLabels(e.target.checked)} /> Show hole IDs</label>
        <label className="label"><input type="checkbox" checked={props.showOrder} onChange={(e) => props.onShowOrder(e.target.checked)} /> Show firing order numbers</label>
        <label className="label"><input type="checkbox" checked={props.showFired} onChange={(e) => props.onShowFired(e.target.checked)} /> Show fired holes</label>
        <label className="label"><input type="checkbox" checked={props.showUnfired} onChange={(e) => props.onShowUnfired(e.target.checked)} /> Show unfired holes</label>
        <label className="label"><input type="checkbox" checked={props.showWavefront} onChange={(e) => props.onShowWavefront(e.target.checked)} /> Show active wavefront</label>
      </div>
    </div>
  );
}

function BlastCanvas({
  holes,
  pattern,
  selectedIds,
  selectedHoleId,
  selectedLine,
  colorMode,
  currentTime,
  showLabels,
  showOrder,
  showFired,
  showUnfired,
  showWavefront,
  focusCanvas,
  onSelect,
  onFocusToggle,
}: {
  holes: BlastHole[];
  pattern: TimingPattern;
  selectedIds: string[];
  selectedHoleId: string | null;
  selectedLine?: TimingLine;
  colorMode: ColorMode;
  currentTime?: number;
  showLabels: boolean;
  showOrder: boolean;
  showFired: boolean;
  showUnfired: boolean;
  showWavefront: boolean;
  focusCanvas: boolean;
  onSelect: (id: string) => void;
  onFocusToggle: () => void;
}) {
  const width = 1400;
  const height = focusCanvas ? 860 : 760;
  const pad = 44;
  const valid = holes.filter((hole) => Number.isFinite(hole.x) && Number.isFinite(hole.y));
  if (!valid.length) {
    return <div className="card" style={{ minHeight: 620, display: "grid", placeItems: "center" }}><div className="subtitle">Import CSV hole data to view the blast layout.</div></div>;
  }
  const xs = valid.map((hole) => hole.x);
  const ys = valid.map((hole) => hole.y);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const span = Math.max(xmax - xmin || 1, ymax - ymin || 1);
  const plotW = width - pad * 2;
  const plotH = height - pad * 2;
  const usedW = plotW * ((xmax - xmin || 1) / span);
  const usedH = plotH * ((ymax - ymin || 1) / span);
  const ox = pad + (plotW - usedW) / 2;
  const oy = pad + (plotH - usedH) / 2;
  const mapX = (x: number) => ox + ((x - xmin) / (xmax - xmin || 1)) * usedW;
  const mapY = (y: number) => height - oy - ((y - ymin) / (ymax - ymin || 1)) * usedH;
  const numericValue = (hole: BlastHole) => {
    if (colorMode === "delay") return hole.delayMs;
    if (colorMode === "row") return hole.rowIndex;
    if (colorMode === "charge") return hole.charge;
    if (colorMode === "depth") return hole.depth;
    if (colorMode === "fragmentation") return hole.estimatedFragmentationMm;
    if (colorMode === "ppv") return hole.estimatedPpvMmS;
    if (colorMode === "airblast") return hole.estimatedAirblastDb;
    if (colorMode === "flyrock") return hole.flyrockRisk === "high" ? 2 : hole.flyrockRisk === "moderate" ? 1 : 0;
    return hole.timingGroup ? Math.abs([...hole.timingGroup].reduce((sum, ch) => sum + ch.charCodeAt(0), 0)) : 0;
  };
  const values = valid.map(numericValue).filter((value): value is number => Number.isFinite(value));
  const vmin = values.length ? Math.min(...values) : 0;
  const vmax = values.length ? Math.max(...values) : 1;
  const colorFor = (hole: BlastHole) => {
    const value = numericValue(hole);
    if (!Number.isFinite(value)) return "#94a3b8";
    const t = vmax === vmin ? 0.5 : Math.max(0, Math.min(1, ((value as number) - vmin) / (vmax - vmin)));
    return `hsl(${220 - 200 * t}, 82%, 55%)`;
  };
  const visible = valid.filter((hole) => {
    if (currentTime == null || !Number.isFinite(hole.delayMs)) return true;
    if ((hole.delayMs as number) < currentTime && !showFired) return false;
    if ((hole.delayMs as number) > currentTime && !showUnfired) return false;
    return true;
  });
  const screenPts = valid.map((hole) => ({ hole, sx: mapX(hole.x), sy: mapY(hole.y) })).sort((a, b) => a.sx - b.sx);
  let minSpacing = Number.POSITIVE_INFINITY;
  for (let i = 0; i < screenPts.length; i += 1) {
    for (let j = i + 1; j < Math.min(screenPts.length, i + 16); j += 1) {
      const dx = screenPts[j].sx - screenPts[i].sx;
      if (Number.isFinite(minSpacing) && dx > minSpacing) break;
      const d = Math.hypot(dx, screenPts[j].sy - screenPts[i].sy);
      if (d > 0) minSpacing = Math.min(minSpacing, d);
    }
  }
  const baseRadius = Math.max(3.4, Math.min(focusCanvas ? 8.5 : 7, (Number.isFinite(minSpacing) ? minSpacing : 18) * 0.28));
  const lineStart = selectedLine ? { x: mapX(selectedLine.start.x), y: mapY(selectedLine.start.y) } : null;
  const lineEnd = selectedLine ? { x: mapX(selectedLine.end.x), y: mapY(selectedLine.end.y) } : null;
  const timed = valid.filter((hole) => Number.isFinite(hole.delayMs)).sort((a, b) => (a.delayMs ?? 0) - (b.delayMs ?? 0));
  const centroid = (items: BlastHole[]) => ({
    x: items.reduce((sum, hole) => sum + hole.x, 0) / Math.max(1, items.length),
    y: items.reduce((sum, hole) => sum + hole.y, 0) / Math.max(1, items.length),
  });
  const early = timed.slice(0, Math.max(1, Math.ceil(timed.length * 0.12)));
  const lateGroups = [0.35, 0.6, 0.85].map((ratio) => {
    const start = Math.max(0, Math.floor(timed.length * ratio) - Math.max(1, Math.ceil(timed.length * 0.04)));
    return timed.slice(start, Math.min(timed.length, start + Math.max(1, Math.ceil(timed.length * 0.08))));
  }).filter((group) => group.length > 0);
  const startCentroid = pattern === "lineDirectional" && lineStart && lineEnd
    ? { x: (lineStart.x + lineEnd.x) / 2, y: (lineStart.y + lineEnd.y) / 2 }
    : (() => {
        const c = centroid(early);
        return { x: mapX(c.x), y: mapY(c.y) };
      })();
  const throwArrows = timed.length
    ? lateGroups.map((group) => {
        const c = centroid(group);
        return { x1: startCentroid.x, y1: startCentroid.y, x2: mapX(c.x), y2: mapY(c.y) };
      }).filter((arrow) => Math.hypot(arrow.x2 - arrow.x1, arrow.y2 - arrow.y1) > 18)
    : [];
  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <div>
          <div className="sectionTitle">Blast Layout</div>
          <div className="subtitle">Large equal-aspect 2D plan view. Arrows show conceptual relief/throw direction from earlier to later timing. Red rings flag elevated flyrock screening risk.</div>
        </div>
        <button className="btn btnPrimary" onClick={onFocusToggle}>{focusCanvas ? "Show tools" : "Focus visualisation"}</button>
      </div>
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} style={{ marginTop: 10, background: "rgba(248,250,252,0.78)", borderRadius: 16 }}>
        <defs>
          <marker id="throwArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#f97316" />
          </marker>
        </defs>
        <rect x={ox} y={oy} width={usedW} height={usedH} rx={10} fill="#ffffff" stroke="rgba(15,23,42,0.12)" />
        {[0, 1, 2, 3].map((idx) => (
          <g key={idx}>
            <line x1={ox + (usedW / 3) * idx} x2={ox + (usedW / 3) * idx} y1={oy} y2={oy + usedH} stroke="rgba(148,163,184,0.18)" />
            <line y1={oy + (usedH / 3) * idx} y2={oy + (usedH / 3) * idx} x1={ox} x2={ox + usedW} stroke="rgba(148,163,184,0.18)" />
          </g>
        ))}
        {lineStart && lineEnd ? (
          <g>
            <line x1={lineStart.x} y1={lineStart.y} x2={lineEnd.x} y2={lineEnd.y} stroke="#111827" strokeWidth={2.4} strokeDasharray="8 5" />
            <circle cx={lineStart.x} cy={lineStart.y} r={baseRadius + 5} fill="none" stroke="#111827" strokeWidth={2} />
            <circle cx={lineEnd.x} cy={lineEnd.y} r={baseRadius + 5} fill="none" stroke="#111827" strokeWidth={2} />
            <text x={(lineStart.x + lineEnd.x) / 2} y={(lineStart.y + lineEnd.y) / 2 - 10} textAnchor="middle" fontSize="12" fill="#111827" fontWeight="900">Initiation line</text>
          </g>
        ) : null}
        {throwArrows.map((arrow, idx) => (
          <g key={`throw-${idx}`}>
            <line x1={arrow.x1} y1={arrow.y1} x2={arrow.x2} y2={arrow.y2} stroke="#f97316" strokeWidth={2.2} strokeDasharray="10 6" markerEnd="url(#throwArrow)" opacity={0.82} />
            {idx === 0 ? <text x={(arrow.x1 + arrow.x2) / 2} y={(arrow.y1 + arrow.y2) / 2 - 8} textAnchor="middle" fontSize="12" fill="#9a3412" fontWeight="900">Relief / throw direction</text> : null}
          </g>
        ))}
        {visible.map((hole) => {
          const x = mapX(hole.x);
          const y = mapY(hole.y);
          const isSelected = selectedIds.includes(hole.id) || selectedHoleId === hole.id;
          const isActive = currentTime != null && Number.isFinite(hole.delayMs) && Math.abs((hole.delayMs as number) - currentTime) < 1e-6;
          const isFired = currentTime != null && Number.isFinite(hole.delayMs) && (hole.delayMs as number) < currentTime;
          const fill = currentTime != null && !isFired && !isActive ? "#cbd5e1" : colorFor(hole);
          return (
            <g key={hole.id} onClick={() => onSelect(hole.id)} style={{ cursor: "pointer" }}>
              {hole.flyrockRisk === "high" || hole.flyrockRisk === "moderate" ? (
                <circle
                  cx={x}
                  cy={y}
                  r={baseRadius + (hole.flyrockRisk === "high" ? 10 : 7)}
                  fill="none"
                  stroke={hole.flyrockRisk === "high" ? "#ef4444" : "#f59e0b"}
                  strokeWidth={hole.flyrockRisk === "high" ? 2 : 1.4}
                  strokeDasharray="4 3"
                />
              ) : null}
              {showWavefront && isActive ? (
                <>
                  <circle cx={x} cy={y} r={baseRadius + 13} fill="rgba(37,99,235,0.08)" stroke="rgba(37,99,235,0.35)" strokeWidth={2} />
                  <circle cx={x} cy={y} r={baseRadius + 25} fill="none" stroke="rgba(37,99,235,0.16)" strokeWidth={2} />
                </>
              ) : null}
              <circle cx={x} cy={y} r={isActive ? baseRadius + 3 : baseRadius} fill={fill} opacity={isFired || isActive || currentTime == null ? 0.96 : 0.58} stroke={isSelected ? "#0f172a" : "rgba(15,23,42,0.28)"} strokeWidth={isSelected ? 2.4 : 0.8} />
              <title>{`${hole.id}\nX ${hole.x}\nY ${hole.y}\nDelay ${hole.delayMs ?? "-"} ms\nOrder ${hole.firingOrder ?? "-"}`}</title>
              {showLabels ? <text x={x + baseRadius + 4} y={y - baseRadius - 2} fontSize={focusCanvas ? "10" : "9"} fill="#0f172a" fontWeight="700">{hole.id}</text> : null}
              {showOrder && Number.isFinite(hole.firingOrder) ? <text x={x} y={y + 3} fontSize="8" textAnchor="middle" fill="#fff" fontWeight="900">{hole.firingOrder}</text> : null}
            </g>
          );
        })}
        <text x={ox} y={height - 18} fill="#64748b" fontSize="11">X {formatNum(xmin)} to {formatNum(xmax)}</text>
        <text x={width - ox} y={height - 18} fill="#64748b" fontSize="11" textAnchor="end">Y {formatNum(ymin)} to {formatNum(ymax)}</text>
      </svg>
    </div>
  );
}

function HoleDetailsPanel({ hole, onDelayChange, onClear }: { hole: BlastHole | null; onDelayChange: (value: string) => void; onClear: () => void }) {
  return (
    <div className="card">
      <div className="sectionTitle">Selected Hole</div>
      {hole ? (
        <div style={{ display: "grid", gap: 8 }}>
          <div className="subtitle">{hole.id} {hole.originalId ? `(original: ${hole.originalId})` : ""}</div>
          <div className="grid2">
            {kpi("X", formatNum(hole.x))}
            {kpi("Y", formatNum(hole.y))}
            {kpi("Depth", formatNum(hole.depth))}
            {kpi("Charge", formatNum(hole.charge))}
            {kpi("Delay", Number.isFinite(hole.delayMs) ? `${formatNum(hole.delayMs, 0)} ms` : "-")}
            {kpi("Order", hole.firingOrder ?? "-")}
            {kpi("Frag X50", Number.isFinite(hole.estimatedFragmentationMm) ? `${formatNum(hole.estimatedFragmentationMm, 0)} mm` : "-")}
            {kpi("PPV", Number.isFinite(hole.estimatedPpvMmS) ? `${formatNum(hole.estimatedPpvMmS)} mm/s` : "-")}
            {kpi("Airblast", Number.isFinite(hole.estimatedAirblastDb) ? `${formatNum(hole.estimatedAirblastDb, 1)} dB` : "-")}
            {kpi("Flyrock", `${hole.flyrockRisk ?? "low"}${Number.isFinite(hole.estimatedFlyrockDistanceM) ? ` / ${formatNum(hole.estimatedFlyrockDistanceM, 0)} m` : ""}`)}
          </div>
          {hole.performanceWarnings?.length ? (
            <div className="error">{hole.performanceWarnings.join(" ")}</div>
          ) : null}
          <label className="label">Manual edit delay ms</label>
          <input className="input" type="number" value={hole.delayMs ?? ""} onChange={(e) => onDelayChange(e.target.value)} />
          <button className="btn" onClick={onClear}>Clear selected delay</button>
        </div>
      ) : (
        <div className="subtitle">Click a hole in the plan view to inspect and manually edit it.</div>
      )}
    </div>
  );
}

function ValidationPanel({ issues }: { issues: ValidationIssue[] }) {
  return (
    <div className="card">
      <div className="sectionTitle">Validation</div>
      {issues.length ? (
        <div style={{ display: "grid", gap: 8, maxHeight: 360, overflow: "auto" }}>
          {issues.slice(0, 80).map((issue, idx) => (
            <div key={`${issue.message}-${idx}`} className={issue.severity === "error" ? "error" : "kpi"} style={{ padding: 9 }}>
              <strong>{issue.severity.toUpperCase()}</strong>: {issue.message}
              {issue.suggestion ? <div className="subtitle">{issue.suggestion}</div> : null}
            </div>
          ))}
        </div>
      ) : (
        <div className="subtitle">No validation issues reported.</div>
      )}
    </div>
  );
}

function AnalysisDashboard({
  holes,
  completeness,
  sim,
  performanceSummary,
}: {
  holes: BlastHole[];
  completeness: number;
  sim: { fired: BlastHole[]; active: BlastHole[]; unfired: BlastHole[] };
  performanceSummary: PerformanceSummary;
}) {
  const chargeValues = holes.map((hole) => hole.charge).filter((charge): charge is number => Number.isFinite(charge));
  const depthValues = holes.map((hole) => hole.depth).filter((depth): depth is number => Number.isFinite(depth));
  const delays = holes.map((hole) => hole.delayMs).filter((delay): delay is number => Number.isFinite(delay));
  const delayCounts = new Map<number, number>();
  delays.forEach((delay) => delayCounts.set(delay, (delayCounts.get(delay) ?? 0) + 1));
  const maxSameDelay = Math.max(0, ...delayCounts.values());
  const totalCharge = chargeValues.reduce((sum, value) => sum + value, 0);
  return (
    <div className="card">
      <div className="sectionTitle">Analysis & Design Indicators</div>
      <div className="grid3" style={{ marginTop: 10 }}>
        {kpi("Completeness", `${completeness}%`)}
        {kpi("Max holes same delay", maxSameDelay)}
        {kpi("Total charge", formatNum(totalCharge))}
        {kpi("Average charge", chargeValues.length ? formatNum(totalCharge / chargeValues.length) : "-")}
        {kpi("Depth range", depthValues.length ? `${formatNum(Math.min(...depthValues))}-${formatNum(Math.max(...depthValues))}` : "-")}
        {kpi("Simulation", `F ${sim.fired.length} / A ${sim.active.length} / U ${sim.unfired.length}`)}
        {kpi("Avg frag X50", Number.isFinite(performanceSummary.averageFragmentationMm) ? `${formatNum(performanceSummary.averageFragmentationMm, 0)} mm` : "-")}
        {kpi("Max PPV", Number.isFinite(performanceSummary.maxPpvMmS) ? `${formatNum(performanceSummary.maxPpvMmS)} mm/s` : "-")}
        {kpi("Max airblast", Number.isFinite(performanceSummary.maxAirblastDb) ? `${formatNum(performanceSummary.maxAirblastDb, 1)} dB` : "-")}
        {kpi("Flyrock risk", `${performanceSummary.highFlyrockRiskCount} high / ${performanceSummary.moderateFlyrockRiskCount} mod`)}
      </div>
      {performanceSummary.warnings.length ? (
        <div className="error" style={{ marginTop: 10 }}>{performanceSummary.warnings.join(" ")}</div>
      ) : null}
      <div className="subtitle" style={{ marginTop: 10 }}>
        Empirical-style planning estimates are uncalibrated screening values. They do not guarantee fragmentation, vibration, PPV, airblast, flyrock, or safety performance.
      </div>
    </div>
  );
}

function ExportPanel({ canExport, hasDelays, message, onCsv, onJson, onReport }: { canExport: boolean; hasDelays: boolean; message: string; onCsv: () => void; onJson: () => void; onReport: () => void }) {
  return (
    <div className="card">
      <div className="sectionTitle">Export</div>
      {!hasDelays ? <div className="subtitle">Assign delays before exporting the final delay assignment CSV.</div> : null}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
        <button className="btn btnPrimary" onClick={onCsv} disabled={!canExport || !hasDelays}>Export CSV</button>
        <button className="btn" onClick={onJson} disabled={!canExport}>Export JSON</button>
        <button className="btn" onClick={onReport} disabled={!canExport}>Printable report</button>
      </div>
      {message ? <div className="subtitle" style={{ marginTop: 8 }}>{message}</div> : null}
    </div>
  );
}
