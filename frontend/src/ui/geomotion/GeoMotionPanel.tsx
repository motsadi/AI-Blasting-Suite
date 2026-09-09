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
const WORKSPACE_STORAGE_KEY = "geomotion_3d_workspace_v2";
const DEFAULT_TIE_UP_FILE = "680-665QS32-33_synthetic_reference.csv";
const ORE_COLOR = "#ffd34d";
const WASTE_COLOR = "#d7dee6";

type Props = {
  apiBaseUrl: string;
  token: string;
  standalone?: boolean;
  userEmail?: string;
  onLogout?: () => void;
};

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

function parseGeoMotionTieUp(text: string) {
  const parsed = parseBlastCsv(text);
  const nextIssues = [...parsed.issues];
  const rejected = parsed.holes.filter(
    (hole) =>
      !Number.isFinite(hole.depth) ||
      (hole.depth as number) < 1 ||
      !Number.isFinite(hole.charge) ||
      (hole.charge as number) <= 0,
  );
  const valid = parsed.holes.filter((hole) => !rejected.includes(hole));
  if (rejected.length) {
    nextIssues.push({
      severity: "warning" as const,
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
  return {
    holes: errors.length
      ? valid
      : valid.map((hole) => ({
          ...hole,
          originalDelayMs: hole.delayMs,
          delayMs: (hole.delayMs as number) - minimumDelay,
        })),
    issues: nextIssues,
    errors,
  };
}

const DEFAULT_TIE_UP = parseGeoMotionTieUp(diamondReferenceCsv());

interface SavedWorkspace {
  projectName: string;
  fileName: string;
  holes: BlastHole[];
  issues: ValidationIssue[];
  inputErrors: string[];
}

function readSavedWorkspace(): SavedWorkspace | null {
  try {
    const raw = localStorage.getItem(WORKSPACE_STORAGE_KEY);
    if (!raw) return null;
    const value = JSON.parse(raw) as Partial<SavedWorkspace> & { version?: number };
    if (value.version !== 1 || !Array.isArray(value.holes) || value.holes.length > 5_000) return null;
    const holes = (value.holes as unknown[]).filter((candidate): candidate is BlastHole => {
      if (!candidate || typeof candidate !== "object") return false;
      const hole = candidate as Partial<BlastHole>;
      return (
        typeof hole.id === "string" &&
        typeof hole.x === "number" &&
        Number.isFinite(hole.x) &&
        typeof hole.y === "number" &&
        Number.isFinite(hole.y) &&
        typeof hole.depth === "number" &&
        Number.isFinite(hole.depth) &&
        typeof hole.charge === "number" &&
        Number.isFinite(hole.charge) &&
        typeof hole.delayMs === "number" &&
        Number.isFinite(hole.delayMs)
      );
    });
    if (holes.length < 3) return null;
    return {
      projectName: typeof value.projectName === "string" ? value.projectName : "GeoMotion project",
      fileName: typeof value.fileName === "string" ? value.fileName : "restored_tie_up.csv",
      holes,
      issues: Array.isArray(value.issues) ? value.issues : [],
      inputErrors: Array.isArray(value.inputErrors)
        ? value.inputErrors.filter((message): message is string => typeof message === "string")
        : [],
    };
  } catch {
    return null;
  }
}

function numericColorValue(block: GeoMotionBlock, mode: GeoMotionColor) {
  if (mode === "grade") return block.grade_cpht;
  if (mode === "uncertainty") return block.uncertainty_m;
  if (mode === "burdenVelocity") return block.burden_velocity_m_s;
  if (mode === "impulse") return block.peak_impulse_m_s;
  return block.displacement_m;
}

function percentile(values: number[], fraction: number) {
  if (!values.length) return 1;
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.min(ordered.length - 1, Math.floor((ordered.length - 1) * fraction))] || 1;
}

function numericColorMaximum(blocks: GeoMotionBlock[], mode: GeoMotionColor) {
  return Math.max(
    0.001,
    percentile(
      blocks
        .map((block) => numericColorValue(block, mode))
        .filter((value) => Number.isFinite(value) && value >= 0),
      0.95,
    ),
  );
}

const MOVEMENT_PALETTE = ["#38bdf8", "#2dd4bf", "#a3e635", "#fde047", "#fb923c", "#fb7185"];

function paletteColor(value: number, maximum: number) {
  const scaled = clamp(value / Math.max(maximum, 0.001), 0, 1) * (MOVEMENT_PALETTE.length - 1);
  const lowerIndex = Math.min(MOVEMENT_PALETTE.length - 2, Math.floor(scaled));
  return new THREE.Color(MOVEMENT_PALETTE[lowerIndex]).lerp(
    new THREE.Color(MOVEMENT_PALETTE[lowerIndex + 1]),
    scaled - lowerIndex,
  );
}

function colorFor(
  block: GeoMotionBlock,
  mode: GeoMotionColor,
  destination: boolean,
  numericMaximum: number,
) {
  if (mode === "classification") {
    const classification = destination ? block.destination_class : block.source_class;
    return classification === "ORE" ? new THREE.Color(ORE_COLOR) : new THREE.Color(WASTE_COLOR);
  }
  if (mode === "facies") {
    return new THREE.Color({ VK: "#a78bfa", SVK_M1: "#38bdf8", CONTACT: "#fb923c", WASTE: "#94a3b8" }[block.facies]);
  }
  return paletteColor(numericColorValue(block, mode), numericMaximum);
}

function ColorLegend({ mode, blocks, destination }: { mode: GeoMotionColor; blocks: GeoMotionBlock[]; destination: boolean }) {
  const classificationCount = (classification: "ORE" | "WASTE") =>
    blocks.filter((block) => (destination ? block.destination_class : block.source_class) === classification).length;
  const faciesCount = (facies: GeoMotionBlock["facies"]) => blocks.filter((block) => block.facies === facies).length;
  const categorical = mode === "classification"
    ? [
        { label: `Ore (${classificationCount("ORE").toLocaleString()})`, color: ORE_COLOR },
        { label: `Waste (${classificationCount("WASTE").toLocaleString()})`, color: WASTE_COLOR },
      ]
    : mode === "facies"
      ? [
          { label: `VK (${faciesCount("VK").toLocaleString()})`, color: "#a78bfa" },
          { label: `SVK M1 (${faciesCount("SVK_M1").toLocaleString()})`, color: "#38bdf8" },
          { label: `Contact (${faciesCount("CONTACT").toLocaleString()})`, color: "#fb923c" },
          { label: `Waste (${faciesCount("WASTE").toLocaleString()})`, color: "#94a3b8" },
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
  const titleByMode: Partial<Record<GeoMotionColor, string>> = {
    grade: "Grade",
    displacement: "Bulk movement",
    uncertainty: "Uncertainty",
    burdenVelocity: "Burden velocity",
    impulse: "Peak impulse",
  };
  const unitByMode: Partial<Record<GeoMotionColor, string>> = {
    grade: "cpht",
    displacement: "m",
    uncertainty: "m",
    burdenVelocity: "m/s",
    impulse: "m/s",
  };
  const maximum = numericColorMaximum(blocks, mode);
  return (
    <div className="geomotionLegend" aria-label={`${titleByMode[mode] ?? "Model value"} colour scale from zero to the 95th percentile`}>
      <strong>{titleByMode[mode] ?? "Model value"}</strong>
      <span>0</span>
      <span className="geomotionGradient" />
      <span>P95 {format(maximum, 2)} {unitByMode[mode] ?? ""}</span>
    </div>
  );
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

interface PlanPoint3D {
  x: number;
  y: number;
  z: number;
}

interface ModelLayout {
  centerX: number;
  centerY: number;
  centerZ: number;
  minX: number;
  maxX: number;
  widthM: number;
  lengthM: number;
  heightM: number;
  spanM: number;
  scaleM: number;
  cellCount: number;
}

type CameraPreset = "perspective" | "plan" | "section";

function niceScaleLength(spanM: number) {
  const target = Math.max(spanM / 5, 1);
  const magnitude = 10 ** Math.floor(Math.log10(target));
  const candidates = [1, 2, 5, 10].map((factor) => factor * magnitude);
  return candidates.filter((value) => value <= target).pop() ?? magnitude;
}

function buildModelLayout(holes: BlastHole[], result: GeoMotionResult | null): ModelLayout {
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  const blocks = result?.blocks ?? [];
  if (blocks.length) {
    const step = Math.max(1, Math.ceil(blocks.length / 8_000));
    const halfCell = Math.max(0.5, (result?.assumptions.cell_size_m ?? 1) / 2);
    blocks.forEach((block, index) => {
      if (index % step !== 0) return;
      xs.push(block.source[0], block.destination[0]);
      ys.push(block.source[1], block.destination[1]);
      zs.push(block.source[2], block.destination[2]);
    });
    if (xs.length) {
      xs.push(Math.min(...xs) - halfCell, Math.max(...xs) + halfCell);
      ys.push(Math.min(...ys) - halfCell, Math.max(...ys) + halfCell);
      zs.push(Math.min(...zs) - halfCell, Math.max(...zs) + halfCell);
    }
  } else {
    holes.forEach((hole) => {
      const collarZ = Number.isFinite(hole.z) ? (hole.z as number) : 0;
      xs.push(hole.x);
      ys.push(hole.y);
      zs.push(collarZ, collarZ - (hole.depth ?? 0));
    });
  }
  const minX = xs.length ? Math.min(...xs) : -8;
  const maxX = xs.length ? Math.max(...xs) : 8;
  const minY = ys.length ? Math.min(...ys) : -8;
  const maxY = ys.length ? Math.max(...ys) : 8;
  const minZ = zs.length ? Math.min(...zs) : -8;
  const maxZ = zs.length ? Math.max(...zs) : 8;
  const widthM = Math.max(maxX - minX, 2);
  const lengthM = Math.max(maxY - minY, 2);
  const heightM = Math.max(maxZ - minZ, 2);
  const spanM = Math.max(widthM, lengthM, heightM, 8);
  return {
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
    centerZ: (minZ + maxZ) / 2,
    minX,
    maxX,
    widthM,
    lengthM,
    heightM,
    spanM,
    scaleM: niceScaleLength(spanM),
    cellCount: blocks.length,
  };
}

function toScenePoint(point: PlanPoint3D, layout: ModelLayout, verticalExaggeration: number) {
  return new THREE.Vector3(
    point.x - layout.centerX,
    (point.z - layout.centerZ) * verticalExaggeration,
    -(point.y - layout.centerY),
  );
}

function cameraFrame(
  preset: CameraPreset,
  layout: ModelLayout,
  verticalExaggeration: number,
) {
  const width = Math.max(layout.widthM, 4);
  const length = Math.max(layout.lengthM, 4);
  const height = Math.max(layout.heightM * verticalExaggeration, 4);
  const fovDeg = 46;
  const halfDiag = Math.hypot(width, length, height) / 2;
  const distance = (halfDiag / Math.tan((fovDeg * Math.PI) / 360)) * 0.74;
  const target = new THREE.Vector3(0, 0, 0);
  if (preset === "plan") {
    return { target, position: new THREE.Vector3(0, distance, 0.04) };
  }
  if (preset === "section") {
    return { target, position: new THREE.Vector3(distance, height * 0.12, 0) };
  }
  return {
    target,
    position: new THREE.Vector3(0.68, 0.5, 0.78).normalize().multiplyScalar(distance),
  };
}

function setCameraPosition(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  preset: CameraPreset,
  layout: ModelLayout,
  verticalExaggeration: number,
) {
  const frame = cameraFrame(preset, layout, verticalExaggeration);
  controls.target.copy(frame.target);
  camera.position.copy(frame.position);
  camera.lookAt(controls.target);
  controls.update();
}

function GeoMotionScene({
  result,
  holes,
  progress,
  view,
  colorMode,
  showVectors,
  verticalExaggeration,
  clipPercent,
  cameraPreset,
  seamPercent,
}: {
  result: GeoMotionResult | null;
  holes: BlastHole[];
  progress: number;
  view: GeoMotionView;
  colorMode: GeoMotionColor;
  showVectors: boolean;
  verticalExaggeration: number;
  clipPercent: number;
  cameraPreset: CameraPreset;
  seamPercent: number;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const cameraTransitionRef = useRef<number | null>(null);
  const progressRef = useRef(progress);
  const layout = useMemo(
    () => buildModelLayout(holes, result),
    [holes, result],
  );
  progressRef.current = progress;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const width = Math.max(host.clientWidth, 320);
    const height = Math.max(host.clientHeight, 460);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#08131f");
    const camera = new THREE.PerspectiveCamera(46, width / height, 0.1, Math.max(2_400, layout.spanM * 18));
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.setSize(width, height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.toneMappingExposure = 1;
    renderer.localClippingEnabled = clipPercent < 99;
    renderer.domElement.setAttribute("aria-hidden", "true");
    host.replaceChildren(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.enablePan = true;
    controls.screenSpacePanning = true;
    controls.minDistance = Math.max(6, layout.spanM * 0.08);
    controls.maxDistance = layout.spanM * 12;
    controls.minPolarAngle = 0.02;
    controls.maxPolarAngle = Math.PI * 0.92;
    setCameraPosition(camera, controls, cameraPreset, layout, verticalExaggeration);
    cameraRef.current = camera;
    controlsRef.current = controls;

    const delayTimes = holes
      .map((hole) => hole.delayMs)
      .filter((delay): delay is number => Number.isFinite(delay))
      .sort((left, right) => left - right);

    const matrix = new THREE.Matrix4();

    let blockMesh: THREE.InstancedMesh | null = null;
    let blockEdgeMesh: THREE.InstancedMesh | null = null;
    let blockGeometry: THREE.BoxGeometry | null = null;
    let blockMaterial: THREE.MeshBasicMaterial | null = null;
    let blockEdgeMaterial: THREE.MeshBasicMaterial | null = null;
    const resultBlocks = result?.blocks ?? [];
    const lodScale = Math.max(1, Math.cbrt(result?.transport?.stride || 1));
    const fallbackVoxelSize = (result?.assumptions.cell_size_m ?? 1) * lodScale;
    const blockRotation = new THREE.Quaternion();
    const blockScale = new THREE.Vector3(1, 1, 1);
    const colorMaximum = numericColorMaximum(resultBlocks, colorMode);

    const blockPosition = (block: GeoMotionBlock, movement: number) => {
      const point = toScenePoint(
        { x: block.source[0], y: block.source[1], z: block.source[2] },
        layout,
        verticalExaggeration,
      );
      if (movement > 0) {
        point.x += block.vector[0] * movement;
        point.z -= block.vector[1] * movement;
        point.y += block.vector[2] * movement * verticalExaggeration;
      }
      return point;
    };
    const movementFor = (block: GeoMotionBlock, timelineProgress: number) => {
      if (view === "source") return 0;
      if (view === "destination") return 1;
      const minimumDelay = delayTimes[0] ?? 0;
      const maximumDelay = delayTimes[delayTimes.length - 1] ?? minimumDelay + 1;
      const currentDelay = minimumDelay + (maximumDelay - minimumDelay) * timelineProgress;
      const movementWindow = Math.max(18, (maximumDelay - minimumDelay) * 0.035);
      const raw = clamp((currentDelay - block.effective_time_ms) / movementWindow, 0, 1);
      return raw * raw * (3 - 2 * raw);
    };

    let clipPlane: THREE.Plane | null = null;
    if (resultBlocks.length && clipPercent < 99) {
      const minX = layout.minX - layout.centerX;
      const maxX = layout.maxX - layout.centerX;
      const cutoff = minX + (maxX - minX) * (clipPercent / 100);
      clipPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), cutoff);
    }

    if (resultBlocks.length) {
      blockGeometry = new THREE.BoxGeometry(1, 1, 1);
      blockMaterial = new THREE.MeshBasicMaterial({
        color: "#ffffff",
        toneMapped: false,
        clippingPlanes: clipPlane ? [clipPlane] : [],
      });
      blockMesh = new THREE.InstancedMesh(blockGeometry, blockMaterial, resultBlocks.length);
      blockMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      blockMesh.frustumCulled = false;
      blockMesh.renderOrder = 10;
      scene.add(blockMesh);
      blockEdgeMaterial = new THREE.MeshBasicMaterial({
        color: "#0b1220",
        wireframe: true,
        transparent: true,
        opacity: 0.42,
        toneMapped: false,
        clippingPlanes: clipPlane ? [clipPlane] : [],
      });
      blockEdgeMesh = new THREE.InstancedMesh(blockGeometry, blockEdgeMaterial, resultBlocks.length);
      blockEdgeMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      blockEdgeMesh.frustumCulled = false;
      blockEdgeMesh.renderOrder = 11;
      scene.add(blockEdgeMesh);
    }

    let vectorGeometry: THREE.BufferGeometry | null = null;
    let vectorMaterial: THREE.LineBasicMaterial | null = null;
    let arrowheadGeometry: THREE.ConeGeometry | null = null;
    let arrowheadMaterial: THREE.MeshBasicMaterial | null = null;
    if (showVectors && resultBlocks.length) {
      const vectorPositions: number[] = [];
      const vectorColors: number[] = [];
      const vectorSamples: Array<{ source: THREE.Vector3; destination: THREE.Vector3 }> = [];
      const sourceColor = new THREE.Color("#67e8f9");
      const destinationColor = new THREE.Color("#fb7185");
      const step = Math.max(1, Math.ceil(resultBlocks.length / 260));
      for (let index = 0; index < resultBlocks.length; index += step) {
        const block = resultBlocks[index];
        const source = blockPosition(block, 0);
        const destination = blockPosition(block, 1);
        if (source.distanceToSquared(destination) < 0.04) continue;
        vectorSamples.push({ source, destination });
        vectorPositions.push(source.x, source.y, source.z, destination.x, destination.y, destination.z);
        vectorColors.push(
          sourceColor.r, sourceColor.g, sourceColor.b,
          destinationColor.r, destinationColor.g, destinationColor.b,
        );
      }
      vectorGeometry = new THREE.BufferGeometry();
      vectorGeometry.setAttribute("position", new THREE.Float32BufferAttribute(vectorPositions, 3));
      vectorGeometry.setAttribute("color", new THREE.Float32BufferAttribute(vectorColors, 3));
      vectorMaterial = new THREE.LineBasicMaterial({
        vertexColors: true,
        opacity: 0.72,
        transparent: true,
        depthTest: false,
      });
      scene.add(new THREE.LineSegments(vectorGeometry, vectorMaterial));
      const arrowRadius = clamp(layout.spanM * 0.0036, 0.34, 0.72);
      arrowheadGeometry = new THREE.ConeGeometry(arrowRadius, arrowRadius * 2.8, 7);
      arrowheadMaterial = new THREE.MeshBasicMaterial({
        color: "#fb7185",
        depthTest: false,
        transparent: true,
        opacity: 0.94,
      });
      const arrowheads = new THREE.InstancedMesh(
        arrowheadGeometry,
        arrowheadMaterial,
        vectorSamples.length,
      );
      const arrowMatrix = new THREE.Matrix4();
      const arrowQuaternion = new THREE.Quaternion();
      const arrowScale = new THREE.Vector3(1, 1, 1);
      const up = new THREE.Vector3(0, 1, 0);
      vectorSamples.forEach(({ source, destination }, index) => {
        const direction = destination.clone().sub(source).normalize();
        arrowQuaternion.setFromUnitVectors(up, direction);
        arrowMatrix.compose(destination, arrowQuaternion, arrowScale);
        arrowheads.setMatrixAt(index, arrowMatrix);
      });
      arrowheads.instanceMatrix.needsUpdate = true;
      arrowheads.renderOrder = 17;
      scene.add(arrowheads);
    }

    const ambient = new THREE.HemisphereLight("#f8fafc", "#475569", 1.15);
    scene.add(ambient);
    const keyLight = new THREE.DirectionalLight("#fff7ed", 1.85);
    keyLight.position.set(layout.spanM * 0.9, layout.spanM * 1.45, layout.spanM * 0.7);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight("#e2e8f0", 0.7);
    fillLight.position.set(-layout.spanM * 0.85, layout.spanM * 0.45, -layout.spanM * 0.55);
    scene.add(fillLight);
    const rimLight = new THREE.DirectionalLight("#bae6fd", 0.45);
    rimLight.position.set(layout.spanM * 0.2, layout.spanM * 0.35, -layout.spanM);
    scene.add(rimLight);

    let lastProgress = Number.NaN;
    const updateTimeline = (timelineProgress: number) => {
      const displayProgress = view === "source" ? 0 : view === "destination" ? 1 : timelineProgress;
      const activeBlockMesh = blockMesh;
      if (!activeBlockMesh) return;
      resultBlocks.forEach((block, index) => {
        const movement = movementFor(block, displayProgress);
        const cubeSize = Math.max(0.1, block.size_m || fallbackVoxelSize) * (1 - seamPercent / 100);
        blockScale.set(cubeSize, cubeSize * verticalExaggeration, cubeSize);
        matrix.compose(blockPosition(block, movement), blockRotation, blockScale);
        activeBlockMesh.setMatrixAt(index, matrix);
        blockEdgeMesh?.setMatrixAt(index, matrix);
        activeBlockMesh.setColorAt(index, colorFor(block, colorMode, movement > 0.5, colorMaximum));
      });
      activeBlockMesh.instanceMatrix.needsUpdate = true;
      if (activeBlockMesh.instanceColor) activeBlockMesh.instanceColor.needsUpdate = true;
      activeBlockMesh.computeBoundingSphere();
      if (blockEdgeMesh) {
        blockEdgeMesh.instanceMatrix.needsUpdate = true;
        blockEdgeMesh.computeBoundingSphere();
      }
    };
    updateTimeline(progressRef.current);

    let frame = 0;
    let isVisible = true;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      const nextProgress = progressRef.current;
      if (nextProgress !== lastProgress) {
        updateTimeline(nextProgress);
        lastProgress = nextProgress;
      }
      if (isVisible) {
        controls.update();
        renderer.render(scene, camera);
      }
    };
    frame = requestAnimationFrame(animate);

    const observer = new ResizeObserver(() => {
      const nextWidth = Math.max(host.clientWidth, 320);
      const nextHeight = Math.max(host.clientHeight, 460);
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight);
    });
    observer.observe(host);
    const visibilityObserver =
      typeof IntersectionObserver === "undefined"
        ? null
        : new IntersectionObserver(([entry]) => {
            isVisible = entry.isIntersecting;
          });
    visibilityObserver?.observe(host);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      visibilityObserver?.disconnect();
      controls.dispose();
      cameraRef.current = null;
      controlsRef.current = null;
      blockGeometry?.dispose();
      blockMaterial?.dispose();
      blockEdgeMaterial?.dispose();
      vectorGeometry?.dispose();
      vectorMaterial?.dispose();
      arrowheadGeometry?.dispose();
      arrowheadMaterial?.dispose();
      renderer.renderLists.dispose();
      renderer.dispose();
      host.replaceChildren();
    };
  }, [result, holes, layout, view, colorMode, showVectors, verticalExaggeration, clipPercent, seamPercent]);

  useEffect(() => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    if (cameraTransitionRef.current != null) cancelAnimationFrame(cameraTransitionRef.current);
    const destination = cameraFrame(cameraPreset, layout, verticalExaggeration);
    const startPosition = camera.position.clone();
    const startTarget = controls.target.clone();
    const startedAt = performance.now();
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
      camera.position.copy(destination.position);
      controls.target.copy(destination.target);
      camera.lookAt(controls.target);
      controls.update();
      return;
    }
    const duration = 620;
    const move = (time: number) => {
      const raw = clamp((time - startedAt) / duration, 0, 1);
      const eased = raw * raw * (3 - 2 * raw);
      camera.position.lerpVectors(startPosition, destination.position, eased);
      controls.target.lerpVectors(startTarget, destination.target, eased);
      camera.lookAt(controls.target);
      controls.update();
      if (raw < 1) cameraTransitionRef.current = requestAnimationFrame(move);
      else cameraTransitionRef.current = null;
    };
    cameraTransitionRef.current = requestAnimationFrame(move);
    return () => {
      if (cameraTransitionRef.current != null) cancelAnimationFrame(cameraTransitionRef.current);
      cameraTransitionRef.current = null;
    };
  }, [cameraPreset, layout, verticalExaggeration]);

  function zoom(factor: number) {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    const offset = camera.position.clone().sub(controls.target).multiplyScalar(factor);
    const distance = clamp(offset.length(), controls.minDistance, controls.maxDistance);
    camera.position.copy(controls.target).add(offset.setLength(distance));
    controls.update();
  }

  function resetCamera() {
    if (cameraRef.current && controlsRef.current) {
      setCameraPosition(
        cameraRef.current,
        controlsRef.current,
        cameraPreset,
        layout,
        verticalExaggeration,
      );
    }
  }

  const delayValues = holes.map((hole) => hole.delayMs).filter((value): value is number => Number.isFinite(value));
  const currentDelay = delayValues.length
    ? [...delayValues].sort((left, right) => left - right)[
        Math.floor(clamp(progress, 0, 1) * Math.max(0, delayValues.length - 1))
      ]
    : null;
  const firedCount =
    view === "destination" || progress >= 0.999
      ? delayValues.length
      : delayValues.filter((delay) => currentDelay != null && delay < currentDelay).length;
  const orderedTimeline = [...holes]
    .filter((hole) => Number.isFinite(hole.delayMs))
    .sort(
      (left, right) =>
        (left.delayMs as number) - (right.delayMs as number) || left.id.localeCompare(right.id),
    );
  const activeEventIndex = orderedTimeline.length
    ? Math.min(orderedTimeline.length - 1, Math.floor(clamp(progress, 0, 1) * orderedTimeline.length))
    : 0;
  const visibleEventStart = Math.max(0, Math.min(activeEventIndex - 2, orderedTimeline.length - 6));
  const visibleEvents = orderedTimeline.slice(visibleEventStart, visibleEventStart + 6);

  return (
    <div className="geomotionSceneFrame">
      <div
        ref={hostRef}
        className="geomotionScene"
        role="img"
        data-testid="geomotion-block-model-scene"
        aria-label={
          result
            ? `Block model only, ${result.blocks.length.toLocaleString()} cells in a ${format(layout.widthM, 0)} by ${format(layout.lengthM, 0)} by ${format(layout.heightM, 0)} metre volume, coloured by ${colorMode === "classification" ? "ore and waste" : colorMode}.`
            : "Empty block-model viewport. Run GeoMotion 3D to load the 1 cubic metre cells."
        }
      />
      <div className="geomotionSceneHud geomotionSceneHudLeft" aria-hidden="true">
        <strong>BLOCK MODEL ONLY</strong>
        <span>
          {result
            ? `${format(layout.widthM, 0)} × ${format(layout.lengthM, 0)} × ${format(layout.heightM, 0)} m · ${result.blocks.length.toLocaleString()} cells`
            : "No cells loaded"}
        </span>
        <span>{result ? `${format(result.assumptions.cell_size_m || 1, 0)} m cubes · ore gold, waste stone` : "Run GeoMotion 3D to show the cubes"}</span>
      </div>
      <div className="geomotionSceneHud geomotionSceneHudRight" aria-live="polite">
        <strong>{result ? (view === "destination" ? "POST-BLAST MODEL" : view === "movement" ? "MOVING CELLS" : "IN-SITU MODEL") : "AWAITING SIMULATION"}</strong>
        <span>{holes.length} holes · {result ? `${result.blocks.length.toLocaleString()} visual cells` : "movement not run"}</span>
        <span>{view === "movement" && currentDelay != null ? `${currentDelay.toFixed(0)} ms · ${firedCount}/${delayValues.length} events` : view === "destination" ? "Post-blast state" : "Pre-blast state"}</span>
        <small>Uncalibrated planning model · cubes only</small>
      </div>
      <div className="geomotionFocusedEvents" data-testid="geomotion-focused-events">
        <strong>DELAY ORDER</strong>
        <ol start={visibleEventStart + 1}>
          {visibleEvents.map((hole, index) => {
            const eventIndex = visibleEventStart + index;
            const status =
              view === "destination" || progress >= 0.999
                ? "fired"
                : view === "movement" && eventIndex < activeEventIndex
                  ? "fired"
                  : view === "movement" && eventIndex === activeEventIndex
                    ? "current"
                    : "queued";
            return (
              <li
                key={hole.id}
                className={status}
                aria-current={status === "current" ? "step" : undefined}
                aria-label={`Event ${eventIndex + 1}, hole ${hole.id}, ${format(hole.delayMs, 0)} milliseconds, ${status}`}
              >
                <i aria-hidden="true" />
                <span>{eventIndex + 1}</span>
                <b>{hole.id}</b>
                <em>{format(hole.delayMs, 0)} ms</em>
              </li>
            );
          })}
        </ol>
      </div>
      <div className="geomotionScaleBar" aria-hidden="true"><span>{format(layout.scaleM, 0)} m reference</span><i /></div>
      <div className="geomotionSceneButtons" role="group" aria-label="3D camera controls">
        <button type="button" onClick={() => zoom(0.82)} aria-label="Zoom in">+</button>
        <button type="button" onClick={() => zoom(1.22)} aria-label="Zoom out">−</button>
        <button
          type="button"
          onClick={resetCamera}
          aria-label="Reset and fit the whole block model"
        >
          Fit
        </button>
      </div>
      <div className="geomotionOrbitHint" aria-hidden="true">Drag to orbit · Right-drag to pan · Wheel or +/− to zoom</div>
    </div>
  );
}

export function GeoMotionPanel({
  apiBaseUrl,
  token,
  standalone = false,
  userEmail,
  onLogout,
}: Props) {
  const restoredWorkspace = useMemo(() => readSavedWorkspace(), []);
  const [projectName, setProjectName] = useState(
    () => restoredWorkspace?.projectName ?? "680-665QS32-33 Diamond Demonstration",
  );
  const [holes, setHoles] = useState<BlastHole[]>(
    () => restoredWorkspace?.holes ?? DEFAULT_TIE_UP.holes.map((hole) => ({ ...hole })),
  );
  const [fileName, setFileName] = useState(
    () => restoredWorkspace?.fileName ?? DEFAULT_TIE_UP_FILE,
  );
  const [issues, setIssues] = useState<ValidationIssue[]>(
    () => restoredWorkspace?.issues ?? [...DEFAULT_TIE_UP.issues],
  );
  const [inputErrors, setInputErrors] = useState<string[]>(
    () => restoredWorkspace?.inputErrors ?? [...DEFAULT_TIE_UP.errors],
  );
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
  const [progress, setProgress] = useState(0);
  const [view, setView] = useState<GeoMotionView>("source");
  const [playing, setPlaying] = useState(false);
  const [colorMode, setColorMode] = useState<GeoMotionColor>("classification");
  const [showVectors, setShowVectors] = useState(false);
  const [verticalExaggeration, setVerticalExaggeration] = useState(1);
  const [clipPercent, setClipPercent] = useState(100);
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>("perspective");
  const [seamPercent, setSeamPercent] = useState(16);
  const [datasetRefs, setDatasetRefs] = useState<GeoMotionRequest["site_data"]["datasets"]>([]);
  const [blockModelFile, setBlockModelFile] = useState<File | null>(null);
  const [nativeFullscreen, setNativeFullscreen] = useState(false);
  const [fallbackFullscreen, setFallbackFullscreen] = useState(false);
  const [standaloneDark, setStandaloneDark] = useState(
    () => window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false,
  );
  const viewerRef = useRef<HTMLDivElement>(null);
  const blastDataRef = useRef<HTMLElement>(null);

  useEffect(() => localStorage.setItem(STORAGE_KEY, JSON.stringify(assumptions)), [assumptions]);

  useEffect(() => {
    localStorage.setItem(
      WORKSPACE_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        projectName,
        fileName,
        holes,
        issues,
        inputErrors,
      }),
    );
  }, [projectName, fileName, holes, issues, inputErrors]);

  useEffect(() => {
    if (!standalone) return;
    const previousTheme = document.body.dataset.theme;
    document.body.dataset.theme = standaloneDark ? "dark" : "light";
    return () => {
      if (previousTheme) document.body.dataset.theme = previousTheme;
      else delete document.body.dataset.theme;
    };
  }, [standalone, standaloneDark]);

  useEffect(() => {
    if (!playing) return;
    const timedHoles = holes.filter((hole) => Number.isFinite(hole.delayMs)).length;
    if (!timedHoles) {
      setPlaying(false);
      return;
    }
    const steps = clamp(timedHoles, 45, 120);
    const interval = window.setInterval(() => {
      setProgress((current) => Math.min(1, current + 1 / steps));
    }, 90);
    return () => window.clearInterval(interval);
  }, [playing, holes]);

  useEffect(() => {
    if (playing && progress >= 0.999) setPlaying(false);
  }, [playing, progress]);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setNativeFullscreen(document.fullscreenElement === viewerRef.current);
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  useEffect(() => {
    if (!fallbackFullscreen) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setFallbackFullscreen(false);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [fallbackFullscreen]);

  const inputSummary = useMemo(() => {
    const charge = holes.reduce((sum, hole) => sum + (hole.charge ?? 0), 0);
    const depths = holes.map((hole) => hole.depth).filter((value): value is number => Number.isFinite(value));
    return { charge, averageDepth: depths.length ? depths.reduce((sum, value) => sum + value, 0) / depths.length : 0 };
  }, [holes]);
  const modelLayout = useMemo(
    () => buildModelLayout(holes, result),
    [holes, result],
  );

  function loadCsv(text: string, name: string) {
    const parsed = parseGeoMotionTieUp(text);
    setHoles(parsed.holes);
    setFileName(name);
    setIssues(parsed.issues);
    setInputErrors(parsed.errors);
    setResult(null);
    setError("");
    setProgress(0);
    setView("source");
    setPlaying(false);
  }

  async function handleFile(file: File | null) {
    if (!file) return;
    loadCsv(await file.text(), file.name);
  }

  async function toggleFullscreen() {
    const element = viewerRef.current;
    if (!element) return;
    if (fallbackFullscreen) {
      setFallbackFullscreen(false);
      return;
    }
    if (document.fullscreenElement === element) {
      await document.exitFullscreen().catch(() => undefined);
      return;
    }
    if (element.requestFullscreen && document.fullscreenEnabled !== false) {
      try {
        await element.requestFullscreen();
        return;
      } catch {
        // Browser policy can reject native fullscreen; the accessible viewport fallback remains available.
      }
    }
    setFallbackFullscreen(true);
  }

  function startPlayback() {
    setView("movement");
    if (progress >= 0.999) setProgress(0);
    setPlaying(true);
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
          assumptions: { ...request.assumptions, cell_size_m: 2, max_visual_blocks: 20000 },
        };
        setResult(simulateGeoMotionLocally(previewRequest));
        setColorMode("classification");
        setProgress(0);
        setView("source");
        setPlaying(false);
        return;
      }
      if (!response.ok) throw new Error(payload?.detail?.[0]?.msg || payload?.detail || `Simulation failed (${response.status})`);
      setResult(payload as GeoMotionResult);
      setColorMode("classification");
      setProgress(0);
      setView("source");
      setPlaying(false);
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

  const numberField = (
    label: string,
    key: Exclude<keyof GeoMotionAssumptions, "electronic_scatter_enabled">,
    suffix: string,
  ) => (
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
  const timedHoles = [...holes]
    .filter((hole) => Number.isFinite(hole.delayMs))
    .sort((left, right) => (left.delayMs as number) - (right.delayMs as number));
  const playbackIndex = timedHoles.length
    ? Math.min(timedHoles.length - 1, Math.floor(progress * timedHoles.length))
    : 0;
  const playbackHole = timedHoles[playbackIndex];
  const fullscreen = nativeFullscreen || fallbackFullscreen;

  return (
    <div className={`geomotionWorkspace${standalone ? " geomotionWorkspaceStandalone" : ""}`}>
      {!standalone ? (
        <>
          <section className="geomotionHero">
            <div>
              <div className="geomotionEyebrow">BLAST MOVEMENT • DILUTION • RECOVERY</div>
              <h2>GeoMotion 3D</h2>
              <p>Move the mine's 1 m³ block model (1 m × 1 m × 1 m cells) through the delay sequence, then identify ore loss, waste dilution, and practical post-blast dig outcomes. This planning simulation never arms, programs, initiates, or connects to blasting hardware.</p>
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
        </>
      ) : null}

      <section
        ref={viewerRef}
        className={`card geomotionViewerCard${standalone ? " geomotionViewerStandalone" : ""}${fallbackFullscreen ? " geomotionViewerFallbackFullscreen" : ""}`}
        data-testid="geomotion-viewer"
      >
        <div className="geomotionViewerHeader">
          <div>
            {standalone ? <div className="geomotionEyebrow">GEOMOTION 3D · BLOCK MODEL WORKSPACE</div> : null}
            <div className="sectionTitle">1 m³ block model</div>
            <div className="subtitle">
              Only the mining block model is shown: individual cubes, ore versus waste colour, and the full model extent. No bench, highwall or mine context is rendered.
            </div>
          </div>
          <div className="geomotionViewerActions">
            {standalone ? (
              <button
                className="btn"
                type="button"
                onClick={() => blastDataRef.current?.scrollIntoView({ behavior: "smooth" })}
              >
                Blast data &amp; setup
              </button>
            ) : null}
            <button className="btn" type="button" onClick={playing ? () => setPlaying(false) : startPlayback} disabled={!timedHoles.length}>
              {playing ? "Pause sequence" : "Play sequence"}
            </button>
            {standalone ? (
              <button
                className="btn"
                type="button"
                aria-label="Toggle colour theme"
                onClick={() => setStandaloneDark((current) => !current)}
              >
                {standaloneDark ? "Light theme" : "Dark theme"}
              </button>
            ) : null}
            <button
              className="btn btnPrimary"
              type="button"
              data-testid="geomotion-fullscreen-button"
              onClick={toggleFullscreen}
              aria-pressed={fullscreen}
            >
              {fullscreen ? "Exit fullscreen" : "Enter browser fullscreen"}
            </button>
            {standalone && onLogout ? (
              <button className="btn" type="button" onClick={onLogout}>
                Sign out
              </button>
            ) : null}
          </div>
        </div>

        <div className="geomotionContextRow" aria-label="Block model summary">
          <span className="mode" data-testid="geomotion-model-scope">Block model only</span>
          <span>{format(modelLayout.widthM, 0)} × {format(modelLayout.lengthM, 0)} × {format(modelLayout.heightM, 0)} m</span>
          <span>{result ? `${result.blocks.length.toLocaleString()} cubes` : "Run to load cubes"}</span>
          <span>Cell 1 m × 1 m × 1 m</span>
          <span className="active">{holes.length} charged holes</span>
          {standalone && userEmail ? <span>{userEmail}</span> : null}
        </div>

        <div className="geomotionViewerControls" aria-label="Movement model view controls">
          <label>
            <span>Model state</span>
            <select
              className="input"
              value={view}
              onChange={(event) => {
                const nextView = event.target.value as GeoMotionView;
                setView(nextView);
                setPlaying(false);
                if (nextView === "source") setProgress(0);
                if (nextView === "destination") setProgress(1);
              }}
            >
              <option value="source">In-situ model</option>
              <option value="movement">Firing timeline</option>
              <option value="destination" disabled={!result}>Post-blast model</option>
            </select>
          </label>
          <label>
            <span>Active cells</span>
            <select className="input" value={colorMode} onChange={(event) => setColorMode(event.target.value as GeoMotionColor)} disabled={!result}>
              <option value="classification">Ore / waste</option>
              <option value="facies">Kimberlite facies</option>
              <option value="grade">Grade cpht</option>
              <option value="displacement">Displacement</option>
              <option value="uncertainty">Uncertainty</option>
              <option value="burdenVelocity">Burden velocity</option>
              <option value="impulse">Peak impulse</option>
            </select>
          </label>
          <label>
            <span>Camera</span>
            <select className="input" value={cameraPreset} onChange={(event) => setCameraPreset(event.target.value as typeof cameraPreset)}>
              <option value="perspective">Perspective</option>
              <option value="plan">Plan</option>
              <option value="section">Section</option>
            </select>
          </label>
          <label>
            <span>Z exaggeration</span>
            <select className="input" value={verticalExaggeration} onChange={(event) => setVerticalExaggeration(Number(event.target.value))}>
              <option value={1}>1x</option><option value={1.5}>1.5x</option><option value={2}>2x</option><option value={3}>3x</option>
            </select>
          </label>
          <label>
            <span>Voxel seam</span>
            <select className="input" value={seamPercent} onChange={(event) => setSeamPercent(Number(event.target.value))} disabled={!result}>
              <option value={0}>Joined</option>
              <option value={8}>8%</option>
              <option value={12}>12%</option>
              <option value={16}>16%</option>
              <option value={22}>22%</option>
            </select>
          </label>
          <label className="geomotionCheckbox">
            <input type="checkbox" checked={showVectors} onChange={(event) => setShowVectors(event.target.checked)} disabled={!result} />
            Movement vectors
          </label>
        </div>

        <div className="geomotionPitLegend" aria-label="Block model colour legend">
          <span><i className="oreCube" />Ore cube</span>
          <span><i className="wasteCube" />Waste cube</span>
          <span><i className="cubeEdge" />Cube edges</span>
          {showVectors ? <span><i className="movementVector" />Movement vector</span> : null}
          <span className="geomotionLegendNote">Individual cubes only. Displacement is uncalibrated bulk-rock movement—not flyrock, damage or an exclusion zone.</span>
        </div>
        {result ? (
          <ColorLegend mode={colorMode} blocks={result.blocks} destination={view === "destination" || (view === "movement" && progress > 0.5)} />
        ) : null}

        {result ? (
          <div className="geomotionTimeline">
            <span>Model cutaway</span>
            <input aria-label="Block model cutaway" type="range" min={5} max={100} value={clipPercent} onChange={(event) => setClipPercent(Number(event.target.value))} />
            <span>{clipPercent}%</span>
          </div>
        ) : null}

        {view === "movement" ? (
          <div className="geomotionPlayback">
            <button className="btn" type="button" onClick={playing ? () => setPlaying(false) : startPlayback} disabled={!timedHoles.length}>
              {playing ? "Pause" : "Play"}
            </button>
            <button className="btn" type="button" onClick={() => { setPlaying(false); setProgress(0); }}>Reset</button>
            <label className="geomotionTimeline">
              <span>0 ms</span>
              <input
                aria-label="Firing sequence position"
                type="range"
                min={0}
                max={100}
                value={Math.round(progress * 100)}
                onChange={(event) => {
                  setPlaying(false);
                  setProgress(Number(event.target.value) / 100);
                }}
              />
              <span>{format(timedHoles[timedHoles.length - 1]?.delayMs, 0)} ms</span>
            </label>
            <div className="subtitle" aria-live="polite">
              {playbackHole
                ? `Event ${playbackIndex + 1}/${timedHoles.length} · hole ${playbackHole.id} · ${format(playbackHole.delayMs, 1)} ms`
                : "Import a delay-bearing tie-up to play the sequence."}
            </div>
          </div>
        ) : null}

        <GeoMotionScene
          result={result}
          holes={holes}
          progress={progress}
          view={view}
          colorMode={colorMode}
          showVectors={showVectors}
          verticalExaggeration={verticalExaggeration}
          clipPercent={clipPercent}
          cameraPreset={cameraPreset}
          seamPercent={seamPercent}
        />
        <div className="geomotionViewerFooter">
          <span><strong>Cubes in local blast coordinates</strong> · Fit frames the whole model</span>
          <span>Planning/simulation only · no detonator, firing-system, or hardware control</span>
        </div>
        <span className="geomotionSrOnly" aria-live="polite">
          {`Block model view${fullscreen ? " in browser fullscreen" : standalone ? " in the dedicated window" : ""}. ${result ? "Ore and waste cubes are visible." : "Run the movement model to show the 1 cubic metre cubes."}`}
        </span>
      </section>

      <div className="geomotionTopGrid">
        <section className="card" id="geomotion-blast-data" ref={blastDataRef}>
          <div className="sectionTitle">1. Tie-up with cumulative delays</div>
          <div className="subtitle">Required columns: Hole ID, X, Y, Z, Depth, Charge, Delay (ms). Every hole needs a unique firing time.</div>
          <label className="label">Project</label>
          <input className="input" value={projectName} onChange={(event) => setProjectName(event.target.value)} />
          <label className="label" style={{ marginTop: 10 }}>Delay-bearing charged-hole CSV</label>
          <input className="input" type="file" accept=".csv" onChange={(event) => handleFile(event.target.files?.[0] ?? null)} />
          <button className="btn" style={{ marginTop: 8 }} onClick={() => loadCsv(diamondReferenceCsv(), DEFAULT_TIE_UP_FILE)}>
            Restore deterministic 182-hole demonstration
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
          <h3>The block model is ready to run</h3>
          <p>Run the movement model to load the 1 m³ ore and waste cubes, then review mass balance, recovery and dilution. The viewer shows the block model only—no bench or mine context.</p>
        </section>
      )}
    </div>
  );
}
