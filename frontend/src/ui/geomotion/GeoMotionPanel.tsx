import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { parseBlastCsv } from "../../lib/csvParser";
import { downloadTextFile } from "../../lib/exportCsv";
import { defaultRowTolerance } from "../../lib/rowDetection";
import { assignTiming } from "../../lib/timingAlgorithms";
import type { BlastHole, ValidationIssue } from "../../types/blast";
import { DEFAULT_TIMING_SETTINGS } from "../../types/blast";
import type {
  GeoMotionAssumptions,
  GeoMotionBlock,
  GeoMotionColor,
  GeoMotionMode,
  GeoMotionResult,
  GeoMotionView,
} from "../../types/geomotion";
import { DIAMOND_DEMO_ASSUMPTIONS, toGeoMotionHoles } from "../../types/geomotion";

const NOTICE = "Synthetic Demonstration / Uncalibrated — Planning Only";
const STORAGE_KEY = "geomotion_3d_demo_v1";

type Props = { apiBaseUrl: string; token: string };

function format(value: number | undefined, digits = 1) {
  return Number.isFinite(value) ? Number(value).toLocaleString(undefined, { maximumFractionDigits: digits }) : "—";
}

function metric(title: string, value: string, detail?: string) {
  return (
    <div className="kpi">
      <div className="kpiTitle">{title}</div>
      <div className="kpiValue">{value}</div>
      {detail ? <div className="subtitle">{detail}</div> : null}
    </div>
  );
}

function diamondReferenceCsv() {
  const rows = ["Hole ID,Depth,Charge,X,Y,Z"];
  const originX = -5357.116;
  const originY = 4521.675;
  const angle = (-17 * Math.PI) / 180;
  let index = 0;
  for (let row = 0; row < 13; row += 1) {
    for (let column = 0; column < 14; column += 1) {
      const localX = column * 7 + (row % 2) * 0.22;
      const localY = row * 6;
      const x = originX - localX * Math.cos(angle) + localY * Math.sin(angle);
      const y = originY - localX * Math.sin(angle) + localY * Math.cos(angle);
      const variation = Math.sin((index + 1) * 1.73) * 0.34;
      const depth = 15.69 + variation;
      const charge = Math.max(570, (depth - 5.02) * 61.4);
      const id = `${String.fromCharCode(65 + row)}${column + 1}`;
      rows.push(`${id},${depth.toFixed(3)},${charge.toFixed(6)},${x.toFixed(3)},${y.toFixed(3)},${(664 + depth).toFixed(3)}`);
      index += 1;
    }
  }
  return rows.join("\n");
}

function ensureTiming(holes: BlastHole[]) {
  if (holes.every((hole) => Number.isFinite(hole.delayMs))) return holes;
  const center = {
    x: holes.reduce((sum, hole) => sum + hole.x, 0) / Math.max(holes.length, 1),
    y: holes.reduce((sum, hole) => sum + hole.y, 0) / Math.max(holes.length, 1),
  };
  return assignTiming(holes, {
    pattern: "vCut",
    settings: { ...DEFAULT_TIMING_SETTINGS, rowTolerance: defaultRowTolerance(holes) },
    initiationPoint: center,
    vWidth: "medium",
  });
}

function colorFor(block: GeoMotionBlock, mode: GeoMotionColor, destination: boolean) {
  if (mode === "classification") {
    const classification = destination ? block.destination_class : block.source_class;
    return classification === "ORE" ? new THREE.Color("#16a34a") : new THREE.Color("#64748b");
  }
  if (mode === "facies") {
    return new THREE.Color({ VK: "#7c3aed", SVK_M1: "#0ea5e9", CONTACT: "#f59e0b", WASTE: "#64748b" }[block.facies]);
  }
  if (mode === "grade") {
    return new THREE.Color().setHSL(0.68 - Math.min(block.grade_cpht / 60, 1) * 0.68, 0.82, 0.5);
  }
  if (mode === "uncertainty") {
    return new THREE.Color().setHSL(0.33 - Math.min(block.uncertainty_m / 3, 1) * 0.33, 0.82, 0.5);
  }
  return new THREE.Color().setHSL(0.62 - Math.min(block.displacement_m / 15, 1) * 0.62, 0.82, 0.5);
}

function GeoMotionScene({
  result,
  progress,
  view,
  colorMode,
  showVectors,
}: {
  result: GeoMotionResult;
  progress: number;
  view: GeoMotionView;
  colorMode: GeoMotionColor;
  showVectors: boolean;
}) {
  const hostRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const width = Math.max(host.clientWidth, 320);
    const height = Math.max(host.clientHeight, 520);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#f8fafc");
    scene.fog = new THREE.Fog("#f8fafc", 180, 500);
    const camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 1500);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    host.replaceChildren(renderer.domElement);

    const all = result.blocks;
    const center = new THREE.Vector3(
      all.reduce((sum, block) => sum + block.source[0], 0) / all.length,
      all.reduce((sum, block) => sum + block.source[1], 0) / all.length,
      all.reduce((sum, block) => sum + block.source[2], 0) / all.length
    );
    const positions = new Float32Array(all.length * 3);
    const colors = new Float32Array(all.length * 3);
    all.forEach((block, index) => {
      const t = view === "source" ? 0 : view === "destination" ? 1 : progress;
      positions[index * 3] = block.source[0] + (block.destination[0] - block.source[0]) * t - center.x;
      positions[index * 3 + 1] = block.source[2] + (block.destination[2] - block.source[2]) * t - center.z;
      positions[index * 3 + 2] = -(block.source[1] + (block.destination[1] - block.source[1]) * t - center.y);
      const color = colorFor(block, colorMode, t > 0.5);
      colors.set(color.toArray(), index * 3);
    });
    const blockGeometry = new THREE.BufferGeometry();
    blockGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    blockGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const points = new THREE.Points(
      blockGeometry,
      new THREE.PointsMaterial({ size: Math.max(2.2, result.assumptions.cell_size_m * 0.55), vertexColors: true, opacity: 0.88, transparent: true })
    );
    scene.add(points);

    const floor = result.validation.floor_rl_m ?? Math.min(...all.map((block) => block.source[2]));
    const holeVertices: number[] = [];
    result.holes.forEach((hole) => {
      holeVertices.push(hole.x - center.x, hole.z - center.z, -(hole.y - center.y));
      holeVertices.push(hole.x - center.x, floor - center.z, -(hole.y - center.y));
    });
    const holeGeometry = new THREE.BufferGeometry();
    holeGeometry.setAttribute("position", new THREE.Float32BufferAttribute(holeVertices, 3));
    scene.add(new THREE.LineSegments(holeGeometry, new THREE.LineBasicMaterial({ color: "#111827", opacity: 0.5, transparent: true })));

    if (showVectors) {
      const vectorVertices: number[] = [];
      const step = Math.max(1, Math.ceil(all.length / 450));
      for (let index = 0; index < all.length; index += step) {
        const block = all[index];
        vectorVertices.push(block.source[0] - center.x, block.source[2] - center.z, -(block.source[1] - center.y));
        vectorVertices.push(block.destination[0] - center.x, block.destination[2] - center.z, -(block.destination[1] - center.y));
      }
      const vectorGeometry = new THREE.BufferGeometry();
      vectorGeometry.setAttribute("position", new THREE.Float32BufferAttribute(vectorVertices, 3));
      scene.add(new THREE.LineSegments(vectorGeometry, new THREE.LineBasicMaterial({ color: "#f97316", opacity: 0.42, transparent: true })));
    }

    const gridSize = Math.max(
      Math.max(...all.map((block) => block.source[0])) - Math.min(...all.map((block) => block.source[0])),
      Math.max(...all.map((block) => block.source[1])) - Math.min(...all.map((block) => block.source[1]))
    ) * 1.5;
    const grid = new THREE.GridHelper(gridSize, 18, "#94a3b8", "#dbe4ef");
    grid.position.y = floor - center.z;
    scene.add(grid);
    scene.add(new THREE.AmbientLight("#ffffff", 2.5));
    const light = new THREE.DirectionalLight("#ffffff", 2);
    light.position.set(60, 100, 30);
    scene.add(light);

    camera.position.set(gridSize * 0.7, gridSize * 0.55, gridSize * 0.72);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 2, 0);
    controls.update();
    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();
    const observer = new ResizeObserver(() => {
      const nextWidth = Math.max(host.clientWidth, 320);
      const nextHeight = Math.max(host.clientHeight, 520);
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight);
    });
    observer.observe(host);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      blockGeometry.dispose();
      holeGeometry.dispose();
      renderer.dispose();
      host.replaceChildren();
    };
  }, [result, progress, view, colorMode, showVectors]);

  return <div ref={hostRef} className="geomotionScene" aria-label="Interactive GeoMotion 3D blast movement view" />;
}

export function GeoMotionPanel({ apiBaseUrl, token }: Props) {
  const [projectName, setProjectName] = useState("680-665QS32-33 Diamond Demonstration");
  const [holes, setHoles] = useState<BlastHole[]>([]);
  const [fileName, setFileName] = useState("");
  const [issues, setIssues] = useState<ValidationIssue[]>([]);
  const [assumptions, setAssumptions] = useState<GeoMotionAssumptions>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? { ...DIAMOND_DEMO_ASSUMPTIONS, ...JSON.parse(saved) } : DIAMOND_DEMO_ASSUMPTIONS;
    } catch {
      return DIAMOND_DEMO_ASSUMPTIONS;
    }
  });
  const [mode, setMode] = useState<GeoMotionMode>("hybrid");
  const [result, setResult] = useState<GeoMotionResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(1);
  const [view, setView] = useState<GeoMotionView>("destination");
  const [colorMode, setColorMode] = useState<GeoMotionColor>("classification");
  const [showVectors, setShowVectors] = useState(true);

  useEffect(() => localStorage.setItem(STORAGE_KEY, JSON.stringify(assumptions)), [assumptions]);

  const inputSummary = useMemo(() => {
    const charge = holes.reduce((sum, hole) => sum + (hole.charge ?? 0), 0);
    const depths = holes.map((hole) => hole.depth).filter((value): value is number => Number.isFinite(value));
    return { charge, averageDepth: depths.length ? depths.reduce((sum, value) => sum + value, 0) / depths.length : 0 };
  }, [holes]);

  function loadCsv(text: string, name: string) {
    const parsed = parseBlastCsv(text);
    setHoles(ensureTiming(parsed.holes));
    setFileName(name);
    setIssues(parsed.issues);
    setResult(null);
    setError("");
  }

  async function handleFile(file: File | null) {
    if (!file) return;
    loadCsv(await file.text(), file.name);
  }

  async function runSimulation() {
    if (holes.length < 3) {
      setError("Import at least three valid blast holes before running GeoMotion.");
      return;
    }
    setRunning(true);
    setError("");
    try {
      const response = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/geomotion/simulate`, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
        body: JSON.stringify({
          project_name: projectName,
          seed: 66532,
          mode,
          holes: toGeoMotionHoles(holes),
          assumptions,
        }),
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(payload?.detail?.[0]?.msg || payload?.detail || `Simulation failed (${response.status})`);
      setResult(payload as GeoMotionResult);
      setProgress(1);
      setView("destination");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunning(false);
    }
  }

  function exportJson() {
    if (!result) return;
    downloadTextFile(JSON.stringify({ project_name: projectName, notice: NOTICE, ...result }, null, 2), "geomotion_3d_synthetic_result.json", "application/json");
  }

  function exportVectors() {
    if (!result) return;
    const header = "Block ID,Source X,Source Y,Source Z,Destination X,Destination Y,Destination Z,dX,dY,dZ,Displacement m,Uncertainty m,Facies,Source Class,Destination Class,Grade cpht,Tonnes,Contained Carats,Notice";
    const rows = result.blocks.map((block) => [
      block.id, ...block.source, ...block.destination, ...block.vector, block.displacement_m, block.uncertainty_m,
      block.facies, block.source_class, block.destination_class, block.grade_cpht, block.tonnes, block.contained_carats, `"${NOTICE}"`,
    ].join(","));
    downloadTextFile([header, ...rows].join("\n"), "geomotion_3d_movement_vectors_synthetic.csv", "text/csv");
  }

  const numberField = (label: string, key: keyof GeoMotionAssumptions, suffix: string) => (
    <label className="geomotionField">
      <span>{label}</span>
      <span className="geomotionInputWithUnit">
        <input
          className="input"
          type="number"
          step="any"
          value={assumptions[key]}
          onChange={(event) => setAssumptions((current) => ({ ...current, [key]: Number(event.target.value) }))}
        />
        <small>{suffix}</small>
      </span>
    </label>
  );

  return (
    <div className="geomotionWorkspace">
      <section className="geomotionHero">
        <div>
          <div className="geomotionEyebrow">PHYSICS-INFORMED ORE MOVEMENT</div>
          <h2>GeoMotion 3D Engine</h2>
          <p>Mass-conserving blast movement, synthetic diamond geology, and post-blast ore-control intelligence.</p>
        </div>
        <div className="geomotionNotice">{NOTICE}</div>
      </section>

      <div className="geomotionWorkflow">
        {["1  Tie-up", "2  Assumptions", "3  Physics + AI", "4  Ore control"].map((step, index) => (
          <div key={step} className={`geomotionStep ${index === 0 && !holes.length ? "active" : index === 2 && running ? "active" : result && index === 3 ? "active" : ""}`}>{step}</div>
        ))}
      </div>

      <div className="geomotionTopGrid">
        <section className="card">
          <div className="sectionTitle">Blast input</div>
          <label className="label">Project</label>
          <input className="input" value={projectName} onChange={(event) => setProjectName(event.target.value)} />
          <label className="label" style={{ marginTop: 10 }}>Charged-hole tie-up CSV</label>
          <input className="input" type="file" accept=".csv" onChange={(event) => handleFile(event.target.files?.[0] ?? null)} />
          <button className="btn" style={{ marginTop: 8 }} onClick={() => loadCsv(diamondReferenceCsv(), "680-665QS32-33_synthetic_reference.csv")}>
            Load 182-hole diamond demonstration
          </button>
          <div className="geomotionMiniGrid">
            {metric("Holes", format(holes.length, 0))}
            {metric("Charge", `${format(inputSummary.charge, 0)} kg`)}
            {metric("Mean depth", `${format(inputSummary.averageDepth, 2)} m`)}
            {metric("Source", fileName || "No CSV")}
          </div>
          {issues.length ? <div className="warningBox">{issues.length} CSV issue(s): {issues.slice(0, 2).map((issue) => issue.message).join(" ")}</div> : null}
        </section>

        <section className="card">
          <div className="sectionTitle">Diamond blast assumptions</div>
          <div className="geomotionAssumptions">
            {numberField("Burden", "burden_m", "m")}
            {numberField("Spacing", "spacing_m", "m")}
            {numberField("Hole diameter", "hole_diameter_mm", "mm")}
            {numberField("Average stemming", "stemming_m", "m")}
            {numberField("Subdrill", "subdrill_m", "m")}
            {numberField("Rock density", "rock_density_t_m3", "t/m³")}
            {numberField("Powder factor", "powder_factor_kg_m3", "kg/m³")}
            {numberField("Swell factor", "swell_factor", "ratio")}
            {numberField("Cutoff grade", "cutoff_grade_cpht", "cpht")}
            {numberField("Free-face azimuth", "free_face_azimuth_deg", "°")}
            {numberField("Model cell", "cell_size_m", "m")}
            {numberField("Relative energy", "explosive_relative_energy", "ratio")}
          </div>
        </section>

        <section className="card geomotionRunCard">
          <div className="sectionTitle">Simulation</div>
          <label className="label">Engine mode</label>
          <select className="input" value={mode} onChange={(event) => setMode(event.target.value as GeoMotionMode)}>
            <option value="physics">Physics baseline</option>
            <option value="hybrid">Physics + synthetic ML residual</option>
          </select>
          <div className="geomotionModelNote">
            {mode === "hybrid"
              ? "A synthetic random-forest residual modifies the constrained physics field. It demonstrates the future calibration architecture; it is not trained on mine measurements."
              : "Deterministic energy, relief, timing, confinement, heave and throw model without ML correction."}
          </div>
          <button className="btn btnPrimary geomotionRunButton" disabled={running || holes.length < 3} onClick={runSimulation}>
            {running ? "Computing movement field…" : "Run GeoMotion 3D"}
          </button>
          {error ? <div className="error">{error}</div> : null}
        </section>
      </div>

      {result ? (
        <>
          <div className="grid3">
            {metric("Ore recovery", `${format(result.metrics.ore_recovery_percent, 2)}%`, `${format(result.metrics.ore_tonnes_recovered, 0)} t retained`)}
            {metric("Ore loss", `${format(result.metrics.ore_loss_percent, 2)}%`, "Ore routed outside synthetic dig limit")}
            {metric("Dilution", `${format(result.metrics.dilution_percent, 2)}%`, `${format(result.metrics.waste_dilution_tonnes, 0)} t waste in ore stream`)}
            {metric("Carat recovery", `${format(result.metrics.carat_recovery_percent, 2)}%`, `${format(result.metrics.recovered_carats, 0)} synthetic carats`)}
            {metric("Feed grade", `${format(result.metrics.predicted_feed_grade_cpht, 2)} cpht`)}
            {metric("Mass balance", `${format(result.metrics.mass_balance_error_percent, 3)}% error`, `${format(result.metrics.total_tonnes, 0)} t modelled`)}
          </div>

          <section className="card geomotionViewerCard">
            <div className="geomotionViewerHeader">
              <div>
                <div className="sectionTitle">Interactive movement model</div>
                <div className="subtitle">Drag to orbit, scroll to zoom. Orange lines are sampled source-to-destination vectors.</div>
              </div>
              <div className="geomotionViewerControls">
                <select className="input" value={view} onChange={(event) => setView(event.target.value as GeoMotionView)}>
                  <option value="source">In-situ model</option>
                  <option value="movement">Movement timeline</option>
                  <option value="destination">Post-blast model</option>
                </select>
                <select className="input" value={colorMode} onChange={(event) => setColorMode(event.target.value as GeoMotionColor)}>
                  <option value="classification">Ore / waste</option>
                  <option value="facies">Kimberlite facies</option>
                  <option value="grade">Grade cpht</option>
                  <option value="displacement">Displacement</option>
                  <option value="uncertainty">Uncertainty</option>
                </select>
                <label className="label"><input type="checkbox" checked={showVectors} onChange={(event) => setShowVectors(event.target.checked)} /> Vectors</label>
              </div>
            </div>
            {view === "movement" ? (
              <div className="geomotionTimeline">
                <span>In situ</span>
                <input type="range" min={0} max={100} value={Math.round(progress * 100)} onChange={(event) => setProgress(Number(event.target.value) / 100)} />
                <span>Post-blast</span>
              </div>
            ) : null}
            <GeoMotionScene result={result} progress={progress} view={view} colorMode={colorMode} showVectors={showVectors} />
          </section>

          <div className="geomotionResultsGrid">
            <section className="card">
              <div className="sectionTitle">Movement and confidence</div>
              <div className="geomotionMiniGrid">
                {metric("Mean movement", `${format(result.metrics.mean_displacement_m, 2)} m`)}
                {metric("P95 movement", `${format(result.metrics.p95_displacement_m, 2)} m`)}
                {metric("Mean heave", `${format(result.metrics.mean_heave_m, 2)} m`)}
                {metric("Maximum throw", `${format(result.metrics.max_throw_m, 2)} m`)}
                {metric("Mean uncertainty", `${format(result.uncertainty.mean_m, 2)} m`)}
                {metric("P95 uncertainty", `${format(result.uncertainty.p95_m, 2)} m`)}
              </div>
              <div className="warningBox">{result.uncertainty.method}</div>
            </section>
            <section className="card">
              <div className="sectionTitle">Ore/waste mixing matrix</div>
              <table className="geomotionTable">
                <thead><tr><th>Source</th><th>Destination</th><th>Tonnes</th><th>Total</th></tr></thead>
                <tbody>{result.mixing_matrix.map((row) => (
                  <tr key={`${row.source}-${row.destination}`}><td>{row.source}</td><td>{row.destination}</td><td>{format(row.tonnes, 0)}</td><td>{format(row.percent_of_total, 2)}%</td></tr>
                ))}</tbody>
              </table>
            </section>
            <section className="card">
              <div className="sectionTitle">Validation and export</div>
              {result.validation.warnings.map((warning) => <div key={warning} className="warningBox">{warning}</div>)}
              <div className="geomotionExportButtons">
                <button className="btn btnPrimary" onClick={exportVectors}>Movement CSV</button>
                <button className="btn" onClick={exportJson}>Result JSON</button>
              </div>
            </section>
          </div>
        </>
      ) : (
        <section className="geomotionEmpty">
          <div className="geomotionEmptyIcon">3D</div>
          <h3>Import a tie-up and create the post-blast model</h3>
          <p>The engine will synthesize diamond geology, preserve tonnes and contained carats, calculate movement vectors, and report ore-control outcomes.</p>
        </section>
      )}
    </div>
  );
}
