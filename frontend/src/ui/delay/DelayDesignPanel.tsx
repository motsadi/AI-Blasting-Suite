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
import { Geomotion3D } from "./Geomotion3D";
import { createDemoBlastHoles } from "./geomotionData";

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
    projectName: "Geomotion North Pit — Bench 680",
    importedFileName: "geomotion_demo_bench_680.csv",
    holes: createDemoBlastHoles(),
    timingPattern: "rowByRow",
    settings: { ...DEFAULT_TIMING_SETTINGS, rowTolerance: 2.4 },
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

  function loadDemoProject() {
    const parsed = parseBlastCsv(sampleCsv);
    const rowTolerance = defaultRowTolerance(parsed.holes);
    const settings = { ...project.settings, rowTolerance };
    const holes = assignTiming(parsed.holes, {
      pattern: "rowByRow",
      settings,
    });
    setColumns(parsed.columns);
    setRawRows(parsed.rows);
    setMapping(parsed.mapping);
    setParseIssues(parsed.issues);
    setSelectedIds([]);
    setSelectedHoleId(null);
    setStepIndex(0);
    setPlaying(false);
    updateProject({
      projectName: "Geomotion North Pit — Bench 680",
      importedFileName: "geomotion_demo_bench_680.csv",
      holes,
      timingPattern: "rowByRow",
      settings,
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
            <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Delay Design & Simulation · Geomotion 3D</div>
            <div className="subtitle">Whole-mine 3D context with localized active-bench tie-up, CSV import, transparent delay assignment, playback, and draft export.</div>
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

      <div className={focusCanvas ? "delay-design-layout delay-design-layout-wide" : "delay-design-layout"}>
        <div style={{ display: "grid", gap: 12 }}>
          <Geomotion3D
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
            playing={playing}
            onPlaying={setPlaying}
            onResetSimulation={() => {
              setPlaying(false);
              setStepIndex(0);
            }}
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
                    onLoadSample={loadDemoProject}
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
                Click holes in Geomotion 3D to select them. For directional-from-line, the last two selected holes define the highlighted initiation line.
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
      <button className="btn" style={{ marginTop: 8 }} onClick={onLoadSample}>Restore Geomotion demo tie-up</button>
      <div className="subtitle" style={{ marginTop: 8 }}>
        Imported holes define the active bench tie-up; the synthetic whole-mine block model remains as spatial context.
      </div>
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
        {props.compact ? <div className="pill">Wide workspace</div> : null}
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
        <div className="subtitle">Click a hole in the Geomotion 3D view to inspect and manually edit it.</div>
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
