import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { parseBlastCsv } from "../../lib/csvParser";
import { downloadTextFile } from "../../lib/exportCsv";
import { simulateGeoMotionLocally } from "../../lib/geomotionFallback";
import type { BlastHole, ValidationIssue } from "../../types/blast";
import type {
  GeoMotionAssumptions,
  GeoMotionBlock,
  GeoMotionColor,
  GeoMotionMode,
  GeoMotionRequest,
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
  const rows = ["Hole ID,Depth,Charge,X,Y,Z,Delay"];
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
      rows.push(`${id},${depth.toFixed(3)},${charge.toFixed(6)},${x.toFixed(3)},${y.toFixed(3)},${(664 + depth).toFixed(3)},${8000 + index * 8}`);
      index += 1;
    }
  }
  return rows.join("\n");
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
    const t = Math.round(Math.min(block.grade_cpht / 60, 1) * 11) / 11;
    return new THREE.Color().setHSL(0.68 - t * 0.68, 0.82, 0.5);
  }
  if (mode === "uncertainty") {
    const t = Math.round(Math.min(block.uncertainty_m / 3, 1) * 11) / 11;
    return new THREE.Color().setHSL(0.33 - t * 0.33, 0.82, 0.5);
  }
  if (mode === "burdenVelocity") {
    const t = Math.round(Math.min(block.burden_velocity_m_s / 8, 1) * 11) / 11;
    return new THREE.Color().setHSL(0.62 - t * 0.62, 0.84, 0.5);
  }
  if (mode === "impulse") {
    const t = Math.round(Math.min(block.peak_impulse_m_s / 8, 1) * 11) / 11;
    return new THREE.Color().setHSL(0.74 - t * 0.74, 0.84, 0.5);
  }
  const t = Math.round(Math.min(block.displacement_m / 15, 1) * 11) / 11;
  return new THREE.Color().setHSL(0.62 - t * 0.62, 0.82, 0.5);
}

function ColorLegend({ mode, blocks, destination }: { mode: GeoMotionColor; blocks: GeoMotionBlock[]; destination: boolean }) {
  const classificationCount = (classification: "ORE" | "WASTE") =>
    blocks.filter((block) => (destination ? block.destination_class : block.source_class) === classification).length;
  const faciesCount = (facies: GeoMotionBlock["facies"]) => blocks.filter((block) => block.facies === facies).length;
  const categorical = mode === "classification"
    ? [
        { label: `Ore (${classificationCount("ORE").toLocaleString()})`, color: "#16a34a" },
        { label: `Waste (${classificationCount("WASTE").toLocaleString()})`, color: "#64748b" },
      ]
    : mode === "facies"
      ? [
          { label: `VK (${faciesCount("VK").toLocaleString()})`, color: "#7c3aed" },
          { label: `SVK M1 (${faciesCount("SVK_M1").toLocaleString()})`, color: "#0ea5e9" },
          { label: `Contact (${faciesCount("CONTACT").toLocaleString()})`, color: "#f59e0b" },
          { label: `Waste (${faciesCount("WASTE").toLocaleString()})`, color: "#64748b" },
        ]
      : null;
  if (categorical) {
    return (
      <div className="geomotionLegend">
        {categorical.map((item) => (
          <span key={item.label}><i style={{ background: item.color }} />{item.label}</span>
        ))}
      </div>
    );
  }
  const title = {
    grade: "Low grade → High grade",
    displacement: "Low movement → High movement",
    uncertainty: "Low uncertainty → High uncertainty",
    burdenVelocity: "Low velocity → High velocity",
    impulse: "Low impulse → High impulse",
  }[mode];
  return <div className="geomotionLegend"><span className="geomotionGradient" />{title}</div>;
}

function GeoMotionScene({
  result,
  progress,
  view,
  colorMode,
  showVectors,
  verticalExaggeration,
  clipPercent,
  cameraPreset,
  seamPercent,
}: {
  result: GeoMotionResult;
  progress: number;
  view: GeoMotionView;
  colorMode: GeoMotionColor;
  showVectors: boolean;
  verticalExaggeration: number;
  clipPercent: number;
  cameraPreset: "perspective" | "plan" | "section";
  seamPercent: number;
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
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    host.replaceChildren(renderer.domElement);

    const all = result.blocks;
    const center = new THREE.Vector3(
      all.reduce((sum, block) => sum + block.source[0], 0) / all.length,
      all.reduce((sum, block) => sum + block.source[1], 0) / all.length,
      all.reduce((sum, block) => sum + block.source[2], 0) / all.length
    );
    const gridSize = Math.max(
      Math.max(...all.map((block) => block.source[0])) - Math.min(...all.map((block) => block.source[0])),
      Math.max(...all.map((block) => block.source[1])) - Math.min(...all.map((block) => block.source[1]))
    ) * 1.5;
    const lodScale = Math.max(1, Math.cbrt(result.transport?.stride || 1));
    const voxelSize = result.assumptions.cell_size_m * lodScale * (1 - seamPercent / 100);
    const blockGeometry = new THREE.BoxGeometry(voxelSize, voxelSize * verticalExaggeration, voxelSize);
    const clippingPlanes = clipPercent < 99
      ? [new THREE.Plane(new THREE.Vector3(-1, 0, 0), gridSize * (clipPercent / 100 - 0.5))]
      : [];
    renderer.localClippingEnabled = clippingPlanes.length > 0;
    const transform = new THREE.Matrix4();
    const colorBatches = new Map<string, { color: THREE.Color; matrices: THREE.Matrix4[] }>();
    all.forEach((block) => {
      const t = view === "source" ? 0 : view === "destination" ? 1 : progress;
      transform.makeTranslation(
        block.source[0] + (block.destination[0] - block.source[0]) * t - center.x,
        (block.source[2] + (block.destination[2] - block.source[2]) * t - center.z) * verticalExaggeration,
        -(block.source[1] + (block.destination[1] - block.source[1]) * t - center.y)
      );
      const color = colorFor(block, colorMode, t > 0.5);
      const key = color.getHexString();
      const batch = colorBatches.get(key) ?? { color, matrices: [] };
      batch.matrices.push(transform.clone());
      colorBatches.set(key, batch);
    });
    const voxelMaterials: THREE.MeshBasicMaterial[] = [];
    colorBatches.forEach((batch) => {
      const material = new THREE.MeshBasicMaterial({ color: batch.color, clippingPlanes });
      const mesh = new THREE.InstancedMesh(blockGeometry, material, batch.matrices.length);
      batch.matrices.forEach((matrix, index) => mesh.setMatrixAt(index, matrix));
      mesh.instanceMatrix.needsUpdate = true;
      voxelMaterials.push(material);
      scene.add(mesh);
    });

    const floor = result.validation.floor_rl_m ?? Math.min(...all.map((block) => block.source[2]));
    const holeVertices: number[] = [];
    result.holes.forEach((hole) => {
      holeVertices.push(hole.x - center.x, (hole.z - center.z) * verticalExaggeration, -(hole.y - center.y));
      holeVertices.push(hole.x - center.x, (floor - center.z) * verticalExaggeration, -(hole.y - center.y));
    });
    const holeGeometry = new THREE.BufferGeometry();
    holeGeometry.setAttribute("position", new THREE.Float32BufferAttribute(holeVertices, 3));
    scene.add(new THREE.LineSegments(holeGeometry, new THREE.LineBasicMaterial({ color: "#111827", opacity: 0.5, transparent: true })));

    if (showVectors) {
      const vectorVertices: number[] = [];
      const step = Math.max(1, Math.ceil(all.length / 450));
      for (let index = 0; index < all.length; index += step) {
        const block = all[index];
        vectorVertices.push(block.source[0] - center.x, (block.source[2] - center.z) * verticalExaggeration, -(block.source[1] - center.y));
        vectorVertices.push(block.destination[0] - center.x, (block.destination[2] - center.z) * verticalExaggeration, -(block.destination[1] - center.y));
      }
      const vectorGeometry = new THREE.BufferGeometry();
      vectorGeometry.setAttribute("position", new THREE.Float32BufferAttribute(vectorVertices, 3));
      scene.add(new THREE.LineSegments(vectorGeometry, new THREE.LineBasicMaterial({ color: "#f97316", opacity: 0.42, transparent: true })));
    }

    const grid = new THREE.GridHelper(gridSize, 18, "#94a3b8", "#dbe4ef");
    grid.position.y = (floor - center.z) * verticalExaggeration;
    scene.add(grid);
    scene.add(new THREE.AmbientLight("#ffffff", 2.5));
    const light = new THREE.DirectionalLight("#ffffff", 2);
    light.position.set(60, 100, 30);
    scene.add(light);

    if (cameraPreset === "plan") camera.position.set(0, gridSize * 1.35, 0.01);
    else if (cameraPreset === "section") camera.position.set(gridSize * 1.25, gridSize * 0.12, 0);
    else camera.position.set(gridSize * 0.7, gridSize * 0.55, gridSize * 0.72);
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
      voxelMaterials.forEach((material) => material.dispose());
      holeGeometry.dispose();
      renderer.dispose();
      host.replaceChildren();
    };
  }, [result, progress, view, colorMode, showVectors, verticalExaggeration, clipPercent, cameraPreset, seamPercent]);

  return <div ref={hostRef} className="geomotionScene" aria-label="Interactive GeoMotion 3D blast movement view" />;
}

export function GeoMotionPanel({ apiBaseUrl, token }: Props) {
  const [projectName, setProjectName] = useState("680-665QS32-33 Diamond Demonstration");
  const [holes, setHoles] = useState<BlastHole[]>([]);
  const [fileName, setFileName] = useState("");
  const [issues, setIssues] = useState<ValidationIssue[]>([]);
  const [inputErrors, setInputErrors] = useState<string[]>([]);
  const [assumptions, setAssumptions] = useState<GeoMotionAssumptions>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved
        ? { ...DIAMOND_DEMO_ASSUMPTIONS, ...JSON.parse(saved), cell_size_m: 1 }
        : DIAMOND_DEMO_ASSUMPTIONS;
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
  const [verticalExaggeration, setVerticalExaggeration] = useState(1);
  const [clipPercent, setClipPercent] = useState(100);
  const [cameraPreset, setCameraPreset] = useState<"perspective" | "plan" | "section">("perspective");
  const [seamPercent, setSeamPercent] = useState(1);
  const [datasetRefs, setDatasetRefs] = useState<GeoMotionRequest["site_data"]["datasets"]>([]);
  const [blockModelFile, setBlockModelFile] = useState<File | null>(null);

  useEffect(() => localStorage.setItem(STORAGE_KEY, JSON.stringify(assumptions)), [assumptions]);

  const inputSummary = useMemo(() => {
    const charge = holes.reduce((sum, hole) => sum + (hole.charge ?? 0), 0);
    const depths = holes.map((hole) => hole.depth).filter((value): value is number => Number.isFinite(value));
    return { charge, averageDepth: depths.length ? depths.reduce((sum, value) => sum + value, 0) / depths.length : 0 };
  }, [holes]);

  function loadCsv(text: string, name: string) {
    const parsed = parseBlastCsv(text);
    const nextIssues = [...parsed.issues];
    const rejected = parsed.holes.filter(
      (hole) =>
        !Number.isFinite(hole.depth) ||
        (hole.depth as number) < 1 ||
        !Number.isFinite(hole.charge) ||
        (hole.charge as number) <= 0
    );
    const valid = parsed.holes.filter((hole) => !rejected.includes(hole));
    if (rejected.length) {
      nextIssues.push({
        severity: "warning",
        message: `${rejected.length} physically invalid row(s) were excluded (depth < 1 m, missing/non-positive charge).`,
        suggestion: "Correct these records in the source charge sheet before a production study.",
      });
    }
    const errors: string[] = [];
    if (!parsed.mapping.delay) errors.push("A Delay column is required. GeoMotion will not invent firing times.");
    const missingDelay = valid.filter((hole) => !Number.isFinite(hole.delayMs));
    if (missingDelay.length) errors.push(`${missingDelay.length} valid hole row(s) have missing or non-numeric Delay.`);
    const delays = valid.map((hole) => hole.delayMs as number).filter(Number.isFinite);
    if (new Set(delays).size !== delays.length) errors.push("Every hole must have a unique cumulative firing time.");
    const minimumDelay = delays.length ? Math.min(...delays) : 0;
    const normalized = errors.length
      ? valid
      : valid.map((hole) => ({
          ...hole,
          originalDelayMs: hole.delayMs,
          delayMs: (hole.delayMs as number) - minimumDelay,
        }));
    setHoles(normalized);
    setFileName(name);
    setIssues(nextIssues);
    setInputErrors(errors);
    setResult(null);
    setError("");
  }

  async function handleFile(file: File | null) {
    if (!file) return;
    loadCsv(await file.text(), file.name);
  }

  async function registerDataset(kind: GeoMotionRequest["site_data"]["datasets"][number]["kind"], file: File | null) {
    if (!file) return;
    const text = await file.text();
    const records = Math.max(0, text.split(/\r?\n/).filter((line) => line.trim()).length - 1);
    if (kind === "grade_control_blocks") {
      setBlockModelFile(file);
      setResult(null);
    }
    setDatasetRefs((current) => [
      ...current.filter((dataset) => dataset.kind !== kind),
      { kind, filename: file.name, records, provenance: "measured", metadata: { status: "registered_for_backend_import" } },
    ]);
  }

  function buildRequest(): GeoMotionRequest {
    return {
      project_name: projectName,
      seed: 66532,
      mode,
      holes: toGeoMotionHoles(holes),
      assumptions,
      site_data: { datasets: datasetRefs, synthetic_defaults_enabled: !blockModelFile },
    };
  }

  async function runSimulation() {
    if (holes.length < 3) {
      setError("Import at least three valid blast holes before running GeoMotion.");
      return;
    }
    if (inputErrors.length || holes.some((hole) => !Number.isFinite(hole.delayMs))) {
      setError(inputErrors[0] || "Every hole requires a valid cumulative Delay.");
      return;
    }
    setRunning(true);
    setError("");
    try {
      const request = buildRequest();
      const response = blockModelFile
        ? await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/geomotion/simulate/upload`, {
            method: "POST",
            headers: { authorization: `Bearer ${token}` },
            body: (() => {
              const form = new FormData();
              form.append("request_json", JSON.stringify(request));
              form.append("block_model", blockModelFile);
              return form;
            })(),
          })
        : await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/geomotion/simulate`, {
            method: "POST",
            headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
            body: JSON.stringify(request),
          });
      const payload = await response.json().catch(() => null);
      if ((response.status === 404 || response.status === 405) && !blockModelFile) {
        const previewRequest: GeoMotionRequest = {
          ...request,
          assumptions: { ...request.assumptions, cell_size_m: 3, max_visual_blocks: 20000 },
        };
        setResult(simulateGeoMotionLocally(previewRequest));
        setProgress(1);
        setView("destination");
        return;
      }
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
    const header = "Block ID,Source X,Source Y,Source Z,Destination X,Destination Y,Destination Z,dX,dY,dZ,Velocity X,Velocity Y,Velocity Z,Displacement m,Uncertainty m,Peak Impulse m/s,Burden Velocity m/s,Contributing Event,Facies,Source Class,Destination Class,Grade cpht,Tonnes,Contained Carats,Physics Cell X m,Physics Cell Y m,Physics Cell Z m,Physics Cell Volume m3,Represented Cell Count,Provenance,Notice";
    const rows = result.blocks.map((block) => [
      block.id, ...block.source, ...block.destination, ...block.vector, ...block.velocity, block.displacement_m, block.uncertainty_m,
      block.peak_impulse_m_s, block.burden_velocity_m_s, block.contributing_event, block.facies, block.source_class,
      block.destination_class, block.grade_cpht, block.tonnes, block.contained_carats,
      ...(block.physics_cell_dimensions_m ?? [1, 1, 1]),
      block.physics_cell_volume_m3 ?? 1, block.represented_cell_count ?? 1, block.provenance, `"${NOTICE}"`,
    ].join(","));
    downloadTextFile([header, ...rows].join("\n"), "geomotion_3d_movement_vectors_synthetic.csv", "text/csv");
  }

  async function exportFullResolution() {
    setError("");
    try {
      const request = buildRequest();
      const response = blockModelFile
        ? await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/geomotion/export/upload`, {
            method: "POST",
            headers: { authorization: `Bearer ${token}` },
            body: (() => {
              const form = new FormData();
              form.append("request_json", JSON.stringify(request));
              form.append("block_model", blockModelFile);
              return form;
            })(),
          })
        : await fetch(`${apiBaseUrl.replace(/\/$/, "")}/v1/geomotion/export`, {
            method: "POST",
            headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
            body: JSON.stringify(request),
          });
      if (!response.ok) throw new Error(response.status === 404 ? "Deploy the GeoMotion Cloud Run backend to enable full 1 m³ cell exports." : `Full export failed (${response.status}).`);
      const url = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "geomotion_1m3_full_resolution.csv.gz";
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
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
  const physicsCellEdgeM = result?.metrics.voxel_edge_length_m ?? result?.metrics.voxel_size_m ?? 1;
  const physicsCellDimensions = result?.blocks[0]?.physics_cell_dimensions_m
    ?? [physicsCellEdgeM, physicsCellEdgeM, physicsCellEdgeM];
  const physicsCellVolumeM3 = result?.metrics.voxel_volume_m3
    ?? physicsCellDimensions[0] * physicsCellDimensions[1] * physicsCellDimensions[2];

  return (
    <div className="geomotionWorkspace">
      <section className="geomotionHero">
        <div>
          <div className="geomotionEyebrow">BLAST MOVEMENT • DILUTION • RECOVERY</div>
          <h2>GeoMotion 3D</h2>
          <p>Move the mine's 1 m³ block model (1 m × 1 m × 1 m cells) through the delay sequence, then identify ore loss, waste dilution, and practical post-blast dig outcomes.</p>
        </div>
        <div className="geomotionNotice">{NOTICE}</div>
      </section>

      <div className="geomotionWorkflow">
        {["1  Upload tie-up", "2  Block model (optional)", "3  Run movement", "4  Review ore control"].map((step, index) => (
          <div
            key={step}
            className={`geomotionStep ${
              (index === 0 && !holes.length) ||
              (index === 1 && holes.length && !result) ||
              (index === 2 && running) ||
              (index === 3 && result)
                ? "active"
                : ""
            }`}
          >
            {step}
          </div>
        ))}
      </div>

      <div className="geomotionTopGrid">
        <section className="card">
          <div className="sectionTitle">1. Tie-up with cumulative delays</div>
          <div className="subtitle">Required columns: Hole ID, X, Y, Z, Depth, Charge, Delay (ms). Every hole needs a unique firing time.</div>
          <label className="label">Project</label>
          <input className="input" value={projectName} onChange={(event) => setProjectName(event.target.value)} />
          <label className="label" style={{ marginTop: 10 }}>Delay-bearing charged-hole CSV</label>
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
          {inputErrors.map((message) => <div key={message} className="error">{message}</div>)}
          {holes.length && !inputErrors.length ? (
            <div className="geomotionModelNote">
              Timing normalized to first firing: 0–{format(Math.max(...holes.map((hole) => hole.delayMs ?? 0)), 1)} ms. Original cumulative values are preserved.
            </div>
          ) : null}

          <div className="geomotionDivider" />
          <div className="sectionTitle">2. Mining block model <span className="pill">Optional</span></div>
          <div className="subtitle">If available, add a 1 m³-cell CSV with X, Y, Z and Density. Every cell must be 1 m × 1 m × 1 m. Otherwise GeoMotion builds a simulated model at the same volume and dimensions.</div>
          <label className="geomotionDropzone">
            <span>{blockModelFile ? "Measured block model ready" : "Use my mining block model"}</span>
            <small>{blockModelFile ? blockModelFile.name : "Optional CSV · cells not exactly 1 m³ are rejected."}</small>
            <input type="file" accept=".csv" onChange={(event) => registerDataset("grade_control_blocks", event.target.files?.[0] ?? null)} />
          </label>
          {!blockModelFile && holes.length ? (
            <div className="geomotionModelNote">Simulated 1 m³ block model selected (1 m × 1 m × 1 m per cell). Results remain uncalibrated until mine geology is supplied.</div>
          ) : null}
        </section>

        <section className="card">
          <div className="sectionTitle">Movement assumptions</div>
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
            <label className="geomotionField">
              <span>Model cell</span>
              <span className="geomotionInputWithUnit">
                <input className="input" type="text" value="1 m × 1 m × 1 m" readOnly aria-label="Fixed model cell dimensions" />
                <small>1 m³</small>
              </span>
            </label>
            {numberField("Relative energy", "explosive_relative_energy", "ratio")}
          </div>
          <details className="geomotionAdvanced">
            <summary>Explosive, rock, joints and loader defaults</summary>
            <div className="geomotionAssumptions">
              {numberField("S135B density", "explosive_density_kg_m3", "kg/m³")}
              {numberField("S135B RWS", "explosive_rws_percent", "%")}
              {numberField("Nominal VOD", "vod_m_s", "m/s")}
              {numberField("VOD uncertainty", "vod_uncertainty_m_s", "m/s")}
              {numberField("UCS", "ucs_mpa", "MPa")}
              {numberField("Tensile strength", "tensile_strength_mpa", "MPa")}
              {numberField("Young's modulus", "youngs_modulus_gpa", "GPa")}
              {numberField("Poisson ratio", "poisson_ratio", "ratio")}
              {numberField("Damping", "damping_ratio", "ratio")}
              {numberField("Fragmentation index", "fragmentation_index", "0–1")}
              {numberField("Joint dip", "joint_dip_deg", "°")}
              {numberField("Joint direction", "joint_dip_direction_deg", "°")}
              {numberField("Joint spacing", "joint_spacing_m", "m")}
              {numberField("Joint persistence", "joint_persistence", "0–1")}
              {numberField("Loader bucket", "loader_bucket_t", "t")}
              {numberField("Minimum mining unit", "minimum_mining_unit_m", "m")}
            </div>
            <div className="warningBox">These values are synthetic/site assumptions until replaced by mine files or manufacturer records.</div>
          </details>
          <details className="geomotionAdvanced">
            <summary>Optional measured mine datasets</summary>
            {[
              ["geological_structures", "Geological structures/joints"],
              ["preblast_surface", "Pre-blast surface"],
              ["postblast_surface", "Post-blast surface"],
              ["movement_monitors", "Movement monitors"],
              ["dig_limits", "Dig limits"],
              ["loader_geometry", "Loader/MMU geometry"],
            ].map(([kind, label]) => (
              <label key={kind} className="geomotionField" style={{ marginTop: 8 }}>
                <span>{label}</span>
                <input className="input" type="file" accept=".csv" onChange={(event) => registerDataset(kind, event.target.files?.[0] ?? null)} />
              </label>
            ))}
            {datasetRefs.map((dataset) => <div key={dataset.kind} className="subtitle">{dataset.kind}: {dataset.filename} ({dataset.records} records, measured)</div>)}
            <div className="warningBox">Registered files replace synthetic providers only after backend schema validation. Unsupported or incomplete files remain excluded and are never silently treated as measured.</div>
          </details>
        </section>

        <section className="card geomotionRunCard">
          <div className="sectionTitle">3. Run movement model</div>
          <div className="subtitle">GeoMotion follows the firing sequence, moves every block, conserves tonnes and contained grade, then remaps the post-blast model.</div>
          <label className="label">Engine mode</label>
          <select className="input" value={mode} onChange={(event) => setMode(event.target.value as GeoMotionMode)}>
            <option value="physics">Event physics baseline</option>
            <option value="hybrid">Event physics + uncertainty realization</option>
          </select>
          <div className="geomotionModelNote">
            {mode === "hybrid"
              ? "Reduced-order timed detonation, gas impulse, burden velocity, dynamic relief, joints and conservative settlement with synthetic uncertainty."
              : "Reduced-order event physics without a site-calibrated residual. This is not FEM/DEM or certified S135B JWL modelling."}
          </div>
          <button
            className="btn btnPrimary geomotionRunButton"
            disabled={running || holes.length < 3 || !!inputErrors.length}
            onClick={runSimulation}
          >
            {running ? "Computing 1 m³ cell physics…" : "Run GeoMotion 3D"}
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
                  <option value="burdenVelocity">Burden velocity</option>
                  <option value="impulse">Peak impulse</option>
                </select>
                <label className="label"><input type="checkbox" checked={showVectors} onChange={(event) => setShowVectors(event.target.checked)} /> Vectors</label>
                <select className="input" value={cameraPreset} onChange={(event) => setCameraPreset(event.target.value as typeof cameraPreset)}>
                  <option value="perspective">Perspective</option>
                  <option value="plan">Plan</option>
                  <option value="section">Section</option>
                </select>
                <label className="label">Z exaggeration
                  <select className="input" value={verticalExaggeration} onChange={(event) => setVerticalExaggeration(Number(event.target.value))}>
                    <option value={1}>1x</option><option value={2}>2x</option><option value={3}>3x</option>
                  </select>
                </label>
                <label className="label">Voxel seam
                  <select className="input" value={seamPercent} onChange={(event) => setSeamPercent(Number(event.target.value))}>
                    <option value={0}>Joined</option><option value={1}>1%</option><option value={2}>2%</option><option value={3}>3%</option>
                  </select>
                </label>
              </div>
            </div>
            <div className="geomotionTimeline">
              <span>Section clip</span>
              <input type="range" min={5} max={100} value={clipPercent} onChange={(event) => setClipPercent(Number(event.target.value))} />
              <span>{clipPercent}%</span>
            </div>
            <ColorLegend mode={colorMode} blocks={result.blocks} destination={view === "destination" || (view === "movement" && progress > 0.5)} />
            {view === "movement" ? (
              <div>
                <div className="geomotionTimeline">
                  <span>In situ</span>
                  <input type="range" min={0} max={100} value={Math.round(progress * 100)} onChange={(event) => setProgress(Number(event.target.value) / 100)} />
                  <span>Post-blast</span>
                </div>
                {result.events.length ? (
                  <div className="subtitle" style={{ padding: "0 8px 8px" }}>
                    Event {Math.min(result.events.length, Math.floor(progress * result.events.length) + 1)}/{result.events.length} ·
                    hole {String(result.events[Math.min(result.events.length - 1, Math.floor(progress * result.events.length))]?.hole_id)} ·
                    {format(Number(result.events[Math.min(result.events.length - 1, Math.floor(progress * result.events.length))]?.actual_time_ms), 2)} ms
                  </div>
                ) : null}
              </div>
            ) : null}
            <GeoMotionScene
              result={result}
              progress={progress}
              view={view}
              colorMode={colorMode}
              showVectors={showVectors}
              verticalExaggeration={verticalExaggeration}
              clipPercent={clipPercent}
              cameraPreset={cameraPreset}
              seamPercent={seamPercent}
            />
          </section>

          <div className="geomotionResultsGrid">
            <section className="card">
              <div className="sectionTitle">Movement and confidence</div>
              <div className="geomotionMiniGrid">
                {metric("Mean movement", `${format(result.metrics.mean_displacement_m, 2)} m`)}
                {metric("P95 movement", `${format(result.metrics.p95_displacement_m, 2)} m`)}
                {metric("Mean heave", `${format(result.metrics.mean_heave_m, 2)} m`)}
                {metric("Maximum throw", `${format(result.metrics.max_throw_m, 2)} m`)}
                {metric("Max burden velocity", `${format(result.metrics.max_burden_velocity_m_s, 3)} m/s`)}
                {metric("Loader recovery", `${format(result.metrics.loader_recovery_percent, 2)}%`)}
                {metric("Loader dilution", `${format(result.metrics.loader_dilution_percent, 2)}%`)}
                {metric(
                  "Physics cells",
                  format(result.metrics.cells, 0),
                  `${format(physicsCellVolumeM3, 2)} m³ each (${format(physicsCellDimensions[0], 1)} m × ${format(physicsCellDimensions[1], 1)} m × ${format(physicsCellDimensions[2], 1)} m)`
                )}
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
                <button className="btn" onClick={exportFullResolution}>Full 1 m³-cell CSV.gz</button>
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
