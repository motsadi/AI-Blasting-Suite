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
const BENCH_SURFACE_COLOR = "#475569";
const HIGHWALL_COLOR = "#604f43";
const PATTERN_ZONE_COLOR = "#22d3ee";
const MOVEMENT_ZONE_COLOR = "#fb923c";

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
    return classification === "ORE" ? new THREE.Color("#22c55e") : new THREE.Color("#64748b");
  }
  if (mode === "facies") {
    return new THREE.Color({ VK: "#8b5cf6", SVK_M1: "#0ea5e9", CONTACT: "#f59e0b", WASTE: "#64748b" }[block.facies]);
  }
  return paletteColor(numericColorValue(block, mode), numericMaximum);
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

interface PlanPoint2D {
  x: number;
  y: number;
}

interface PlanPoint3D extends PlanPoint2D {
  z: number;
}

interface BenchLayout {
  centerX: number;
  centerY: number;
  surfaceZ: number;
  forward: PlanPoint2D;
  cross: PlanPoint2D;
  floorBackM: number;
  floorFrontM: number;
  floorLeftM: number;
  floorRightM: number;
  patternPolygon: PlanPoint2D[];
  affectedPolygon: PlanPoint2D[];
  floorPolygon: PlanPoint2D[];
  influenceRadiusM: number;
  uncertaintyBufferM: number;
  widthM: number;
  lengthM: number;
  areaM2: number;
  depthM: number;
  highwallHeightM: number;
  spanM: number;
  scaleM: number;
}

function makeTextSprite(
  text: string,
  color: string,
  highlighted = false,
  worldWidth = 30,
) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.fillStyle = highlighted ? "rgba(69,36,5,.94)" : "rgba(5,12,23,.88)";
  context.strokeStyle = highlighted ? "rgba(251,146,60,.9)" : "rgba(148,163,184,.48)";
  context.lineWidth = 4;
  context.beginPath();
  context.roundRect(3, 3, 506, 122, 18);
  context.fill();
  context.stroke();
  context.fillStyle = color;
  context.font = `800 ${highlighted ? 41 : 37}px ui-sans-serif, system-ui, sans-serif`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, 256, 67);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(worldWidth, worldWidth / 4, 1);
  sprite.renderOrder = 20;
  return sprite;
}

type CameraPreset = "perspective" | "plan" | "section";

function median(values: number[]) {
  if (!values.length) return 0;
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2
    ? ordered[middle]
    : (ordered[middle - 1] + ordered[middle]) / 2;
}

function convexHull(points: PlanPoint2D[]) {
  const unique = Array.from(
    new Map(points.map((point) => [`${point.x.toFixed(5)}:${point.y.toFixed(5)}`, point])).values(),
  );
  if (unique.length <= 2) return unique;
  const sorted = [...unique].sort((left, right) => left.x - right.x || left.y - right.y);
  const cross = (origin: PlanPoint2D, left: PlanPoint2D, right: PlanPoint2D) =>
    (left.x - origin.x) * (right.y - origin.y) -
    (left.y - origin.y) * (right.x - origin.x);
  const lower: PlanPoint2D[] = [];
  sorted.forEach((point) => {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
      lower.pop();
    }
    lower.push(point);
  });
  const upper: PlanPoint2D[] = [];
  [...sorted].reverse().forEach((point) => {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
      upper.pop();
    }
    upper.push(point);
  });
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

function bufferedHull(points: PlanPoint2D[], radiusM: number) {
  const baseHull = convexHull(points);
  if (!baseHull.length) return [{ x: -5, y: -5 }, { x: 5, y: -5 }, { x: 5, y: 5 }, { x: -5, y: 5 }];
  const radius = Math.max(radiusM, 0.5);
  const ringPoints = baseHull.flatMap((point) =>
    Array.from({ length: 12 }, (_, index) => {
      const angle = (index / 12) * Math.PI * 2;
      return { x: point.x + Math.cos(angle) * radius, y: point.y + Math.sin(angle) * radius };
    }),
  );
  return convexHull(ringPoints);
}

function polygonArea(points: PlanPoint2D[]) {
  if (points.length < 3) return 0;
  return Math.abs(
    points.reduce((sum, point, index) => {
      const next = points[(index + 1) % points.length];
      return sum + point.x * next.y - next.x * point.y;
    }, 0) / 2,
  );
}

function niceScaleLength(spanM: number) {
  const target = Math.max(spanM / 5, 1);
  const magnitude = 10 ** Math.floor(Math.log10(target));
  const candidates = [1, 2, 5, 10].map((factor) => factor * magnitude);
  return candidates.filter((value) => value <= target).pop() ?? magnitude;
}

function pointFromBenchAxes(layout: BenchLayout, alongM: number, acrossM: number): PlanPoint2D {
  return {
    x: layout.centerX + layout.forward.x * alongM + layout.cross.x * acrossM,
    y: layout.centerY + layout.forward.y * alongM + layout.cross.y * acrossM,
  };
}

function projectOnBenchAxes(
  point: PlanPoint2D,
  centerX: number,
  centerY: number,
  forward: PlanPoint2D,
  crossAxis: PlanPoint2D,
) {
  const dx = point.x - centerX;
  const dy = point.y - centerY;
  return {
    along: dx * forward.x + dy * forward.y,
    across: dx * crossAxis.x + dy * crossAxis.y,
  };
}

function buildBenchLayout(
  holes: BlastHole[],
  result: GeoMotionResult | null,
  assumptions: GeoMotionAssumptions,
): BenchLayout {
  const holePoints = holes.map((hole) => ({ x: hole.x, y: hole.y }));
  const fallbackPoints = holePoints.length ? holePoints : [{ x: 0, y: 0 }];
  const blockStep = Math.max(1, Math.ceil((result?.blocks.length ?? 0) / 6_000));
  const movedPoints = result
    ? result.blocks.flatMap((block, index) =>
        index % blockStep === 0
          ? [
              { x: block.source[0], y: block.source[1] },
              { x: block.destination[0], y: block.destination[1] },
            ]
          : [],
      )
    : [];
  const influenceRadiusM = Math.max(assumptions.burden_m, assumptions.spacing_m, 1) * 0.9;
  const uncertaintyBufferM = result
    ? Math.max(result.uncertainty.p95_m || 0, (result.assumptions.cell_size_m || 1) * 0.5)
    : 0;
  const patternPolygon = bufferedHull(fallbackPoints, influenceRadiusM);
  const affectedPolygon = result
    ? bufferedHull([...fallbackPoints, ...movedPoints], uncertaintyBufferM)
    : patternPolygon;
  const affectedXs = affectedPolygon.map((point) => point.x);
  const affectedYs = affectedPolygon.map((point) => point.y);
  const centerX = (Math.min(...affectedXs) + Math.max(...affectedXs)) / 2;
  const centerY = (Math.min(...affectedYs) + Math.max(...affectedYs)) / 2;
  const azimuth = ((assumptions.free_face_azimuth_deg % 360) + 360) % 360;
  const radians = (azimuth * Math.PI) / 180;
  const forward = { x: Math.sin(radians), y: Math.cos(radians) };
  const crossAxis = { x: Math.cos(radians), y: -Math.sin(radians) };
  const projected = affectedPolygon.map((point) =>
    projectOnBenchAxes(point, centerX, centerY, forward, crossAxis),
  );
  const minAlong = Math.min(...projected.map((point) => point.along));
  const maxAlong = Math.max(...projected.map((point) => point.along));
  const minAcross = Math.min(...projected.map((point) => point.across));
  const maxAcross = Math.max(...projected.map((point) => point.across));
  const benchPaddingM = Math.max(8, influenceRadiusM * 0.8);
  const floorBackM = minAlong - benchPaddingM;
  const floorFrontM = maxAlong + benchPaddingM;
  const floorLeftM = minAcross - benchPaddingM;
  const floorRightM = maxAcross + benchPaddingM;
  const collarElevations = holes
    .map((hole) => hole.z)
    .filter((value): value is number => Number.isFinite(value));
  const sourceElevations = result?.blocks.map((block) => block.source[2]) ?? [];
  const surfaceZ = collarElevations.length
    ? median(collarElevations)
    : sourceElevations.length
      ? Math.max(...sourceElevations) + (result?.assumptions.cell_size_m ?? 1) / 2
      : 680;
  const averageDepth = holes.length
    ? holes.reduce((sum, hole) => sum + (hole.depth ?? 12), 0) / holes.length
    : 12;
  const partialLayout = {
    centerX,
    centerY,
    surfaceZ,
    forward,
    cross: crossAxis,
    floorBackM,
    floorFrontM,
    floorLeftM,
    floorRightM,
  };
  const floorPolygon = [
    pointFromBenchAxes(partialLayout as BenchLayout, floorBackM, floorLeftM),
    pointFromBenchAxes(partialLayout as BenchLayout, floorFrontM, floorLeftM),
    pointFromBenchAxes(partialLayout as BenchLayout, floorFrontM, floorRightM),
    pointFromBenchAxes(partialLayout as BenchLayout, floorBackM, floorRightM),
  ];
  const widthM = maxAcross - minAcross;
  const lengthM = maxAlong - minAlong;
  const spanM = Math.max(floorFrontM - floorBackM, floorRightM - floorLeftM, 36);
  return {
    ...partialLayout,
    patternPolygon,
    affectedPolygon,
    floorPolygon,
    influenceRadiusM,
    uncertaintyBufferM,
    widthM,
    lengthM,
    areaM2: polygonArea(affectedPolygon),
    depthM: Math.max(averageDepth + assumptions.subdrill_m, 8),
    highwallHeightM: clamp(averageDepth * 0.8, 8, 20),
    spanM,
    scaleM: niceScaleLength(spanM),
  };
}

function toScenePoint(point: PlanPoint3D, layout: BenchLayout, verticalExaggeration: number) {
  return new THREE.Vector3(
    point.x - layout.centerX,
    (point.z - layout.surfaceZ) * verticalExaggeration,
    -(point.y - layout.centerY),
  );
}

function cameraFrame(
  preset: CameraPreset,
  layout: BenchLayout,
  verticalExaggeration: number,
) {
  const span = layout.spanM;
  const target = new THREE.Vector3(0, -layout.depthM * verticalExaggeration * 0.28, 0);
  const sceneForward = new THREE.Vector3(layout.forward.x, 0, -layout.forward.y);
  const sceneCross = new THREE.Vector3(layout.cross.x, 0, -layout.cross.y);
  if (preset === "plan") {
    return { target, position: target.clone().add(new THREE.Vector3(0, span * 1.55, 0.01)) };
  }
  if (preset === "section") {
    return {
      target,
      position: target.clone().add(sceneCross.multiplyScalar(span * 1.45)).add(new THREE.Vector3(0, span * 0.2, 0)),
    };
  }
  return {
    target,
    position: target
      .clone()
      .add(sceneCross.multiplyScalar(span * 0.62))
      .add(sceneForward.multiplyScalar(-span * 0.88))
      .add(new THREE.Vector3(0, span * 0.72, 0)),
  };
}

function setCameraPosition(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  preset: CameraPreset,
  layout: BenchLayout,
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
  assumptions,
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
  assumptions: GeoMotionAssumptions;
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
    () => buildBenchLayout(holes, result, assumptions),
    [holes, result, assumptions],
  );
  progressRef.current = progress;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const width = Math.max(host.clientWidth, 320);
    const height = Math.max(host.clientHeight, 460);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#08131f");
    scene.fog = new THREE.Fog("#08131f", layout.spanM * 2.1, layout.spanM * 5.2);
    const camera = new THREE.PerspectiveCamera(43, width / height, 0.1, Math.max(1_200, layout.spanM * 12));
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.setSize(width, height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.localClippingEnabled = clipPercent < 99;
    renderer.domElement.setAttribute("aria-hidden", "true");
    host.replaceChildren(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.enablePan = true;
    controls.screenSpacePanning = true;
    controls.minDistance = Math.max(10, layout.spanM * 0.1);
    controls.maxDistance = layout.spanM * 6;
    controls.minPolarAngle = 0.04;
    controls.maxPolarAngle = Math.PI * 0.48;
    setCameraPosition(camera, controls, cameraPreset, layout, verticalExaggeration);
    cameraRef.current = camera;
    controlsRef.current = controls;

    const makePlanGeometry = (points: PlanPoint2D[]) => {
      const shape = new THREE.Shape();
      points.forEach((point, index) => {
        const x = point.x - layout.centerX;
        const y = point.y - layout.centerY;
        if (index === 0) shape.moveTo(x, y);
        else shape.lineTo(x, y);
      });
      shape.closePath();
      const geometry = new THREE.ShapeGeometry(shape);
      geometry.rotateX(-Math.PI / 2);
      return geometry;
    };
    const makeOutlineGeometry = (points: PlanPoint2D[], elevationM: number) =>
      new THREE.BufferGeometry().setFromPoints([
        ...points.map((point) =>
          toScenePoint({ ...point, z: layout.surfaceZ + elevationM }, layout, verticalExaggeration),
        ),
        toScenePoint({ ...points[0], z: layout.surfaceZ + elevationM }, layout, verticalExaggeration),
      ]);
    const makeQuadGeometry = (points: THREE.Vector3[]) => {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        points[0], points[1], points[2], points[0], points[2], points[3],
      ]);
      geometry.computeVertexNormals();
      return geometry;
    };

    const floorGeometry = makePlanGeometry(layout.floorPolygon);
    const floorMaterial = new THREE.MeshStandardMaterial({
      color: BENCH_SURFACE_COLOR,
      roughness: 0.94,
      metalness: 0.01,
      transparent: true,
      opacity: 0.74,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const floorMesh = new THREE.Mesh(floorGeometry, floorMaterial);
    floorMesh.position.y = -0.08;
    floorMesh.renderOrder = 1;
    scene.add(floorMesh);

    const gridPositions: number[] = [];
    const gridStepM = Math.max(5, niceScaleLength(layout.spanM) / 2);
    for (
      let along = Math.ceil(layout.floorBackM / gridStepM) * gridStepM;
      along <= layout.floorFrontM;
      along += gridStepM
    ) {
      const start = pointFromBenchAxes(layout, along, layout.floorLeftM);
      const end = pointFromBenchAxes(layout, along, layout.floorRightM);
      const startScene = toScenePoint({ ...start, z: layout.surfaceZ + 0.03 }, layout, verticalExaggeration);
      const endScene = toScenePoint({ ...end, z: layout.surfaceZ + 0.03 }, layout, verticalExaggeration);
      gridPositions.push(startScene.x, startScene.y, startScene.z, endScene.x, endScene.y, endScene.z);
    }
    for (
      let across = Math.ceil(layout.floorLeftM / gridStepM) * gridStepM;
      across <= layout.floorRightM;
      across += gridStepM
    ) {
      const start = pointFromBenchAxes(layout, layout.floorBackM, across);
      const end = pointFromBenchAxes(layout, layout.floorFrontM, across);
      const startScene = toScenePoint({ ...start, z: layout.surfaceZ + 0.03 }, layout, verticalExaggeration);
      const endScene = toScenePoint({ ...end, z: layout.surfaceZ + 0.03 }, layout, verticalExaggeration);
      gridPositions.push(startScene.x, startScene.y, startScene.z, endScene.x, endScene.y, endScene.z);
    }
    const gridGeometry = new THREE.BufferGeometry();
    gridGeometry.setAttribute("position", new THREE.Float32BufferAttribute(gridPositions, 3));
    const gridMaterial = new THREE.LineBasicMaterial({ color: "#cbd5e1", opacity: 0.13, transparent: true });
    scene.add(new THREE.LineSegments(gridGeometry, gridMaterial));

    const backLeft = pointFromBenchAxes(layout, layout.floorBackM, layout.floorLeftM);
    const backRight = pointFromBenchAxes(layout, layout.floorBackM, layout.floorRightM);
    const highwallGeometry = makeQuadGeometry([
      toScenePoint({ ...backLeft, z: layout.surfaceZ }, layout, verticalExaggeration),
      toScenePoint({ ...backRight, z: layout.surfaceZ }, layout, verticalExaggeration),
      toScenePoint({ ...backRight, z: layout.surfaceZ + layout.highwallHeightM }, layout, verticalExaggeration),
      toScenePoint({ ...backLeft, z: layout.surfaceZ + layout.highwallHeightM }, layout, verticalExaggeration),
    ]);
    const highwallMaterial = new THREE.MeshStandardMaterial({
      color: HIGHWALL_COLOR,
      roughness: 1,
      side: THREE.DoubleSide,
    });
    scene.add(new THREE.Mesh(highwallGeometry, highwallMaterial));

    const strataPositions: number[] = [];
    [0.25, 0.5, 0.75].forEach((fraction) => {
      const left = toScenePoint(
        { ...backLeft, z: layout.surfaceZ + layout.highwallHeightM * fraction },
        layout,
        verticalExaggeration,
      );
      const right = toScenePoint(
        { ...backRight, z: layout.surfaceZ + layout.highwallHeightM * fraction },
        layout,
        verticalExaggeration,
      );
      strataPositions.push(left.x, left.y, left.z, right.x, right.y, right.z);
    });
    const strataGeometry = new THREE.BufferGeometry();
    strataGeometry.setAttribute("position", new THREE.Float32BufferAttribute(strataPositions, 3));
    const strataMaterial = new THREE.LineBasicMaterial({ color: "#d6b98c", opacity: 0.3, transparent: true });
    scene.add(new THREE.LineSegments(strataGeometry, strataMaterial));

    const faceLeft = pointFromBenchAxes(layout, layout.floorFrontM, layout.floorLeftM);
    const faceRight = pointFromBenchAxes(layout, layout.floorFrontM, layout.floorRightM);
    const freeFaceGeometry = makeQuadGeometry([
      toScenePoint({ ...faceLeft, z: layout.surfaceZ }, layout, verticalExaggeration),
      toScenePoint({ ...faceRight, z: layout.surfaceZ }, layout, verticalExaggeration),
      toScenePoint({ ...faceRight, z: layout.surfaceZ - Math.min(layout.depthM * 0.45, 8) }, layout, verticalExaggeration),
      toScenePoint({ ...faceLeft, z: layout.surfaceZ - Math.min(layout.depthM * 0.45, 8) }, layout, verticalExaggeration),
    ]);
    const freeFaceMaterial = new THREE.MeshStandardMaterial({
      color: "#193e4b",
      roughness: 0.88,
      side: THREE.DoubleSide,
    });
    scene.add(new THREE.Mesh(freeFaceGeometry, freeFaceMaterial));

    const floorOutlineGeometry = makeOutlineGeometry(layout.floorPolygon, 0.08);
    const floorOutlineMaterial = new THREE.LineBasicMaterial({ color: "#94a3b8", opacity: 0.62, transparent: true });
    scene.add(new THREE.Line(floorOutlineGeometry, floorOutlineMaterial));

    const patternGeometry = makePlanGeometry(layout.patternPolygon);
    const patternMaterial = new THREE.MeshBasicMaterial({
      color: PATTERN_ZONE_COLOR,
      opacity: 0.1,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const patternMesh = new THREE.Mesh(patternGeometry, patternMaterial);
    patternMesh.position.y = 0.1;
    patternMesh.renderOrder = 3;
    scene.add(patternMesh);
    const patternOutlineGeometry = makeOutlineGeometry(layout.patternPolygon, 0.14);
    const patternOutlineMaterial = new THREE.LineDashedMaterial({
      color: PATTERN_ZONE_COLOR,
      dashSize: 2.8,
      gapSize: 1.6,
    });
    const patternOutline = new THREE.Line(patternOutlineGeometry, patternOutlineMaterial);
    patternOutline.computeLineDistances();
    scene.add(patternOutline);

    let affectedGeometry: THREE.ShapeGeometry | null = null;
    let affectedMaterial: THREE.MeshBasicMaterial | null = null;
    let affectedOutlineGeometry: THREE.BufferGeometry | null = null;
    let affectedOutlineMaterial: THREE.LineDashedMaterial | null = null;
    let affectedTubeGeometry: THREE.TubeGeometry | null = null;
    let affectedTubeMaterial: THREE.MeshBasicMaterial | null = null;
    if (result) {
      affectedGeometry = makePlanGeometry(layout.affectedPolygon);
      affectedMaterial = new THREE.MeshBasicMaterial({
        color: MOVEMENT_ZONE_COLOR,
        opacity: 0.16,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
      });
      const affectedMesh = new THREE.Mesh(affectedGeometry, affectedMaterial);
      affectedMesh.position.y = 0.18;
      affectedMesh.renderOrder = 4;
      scene.add(affectedMesh);
      affectedOutlineGeometry = makeOutlineGeometry(layout.affectedPolygon, 0.22);
      affectedOutlineMaterial = new THREE.LineDashedMaterial({
        color: MOVEMENT_ZONE_COLOR,
        dashSize: 3.6,
        gapSize: 1.8,
      });
      const affectedOutline = new THREE.Line(affectedOutlineGeometry, affectedOutlineMaterial);
      affectedOutline.computeLineDistances();
      scene.add(affectedOutline);
      const envelopeCurve = new THREE.CatmullRomCurve3(
        layout.affectedPolygon.map((point) =>
          toScenePoint({ ...point, z: layout.surfaceZ + 0.72 }, layout, verticalExaggeration),
        ),
        true,
        "centripetal",
        0.18,
      );
      affectedTubeGeometry = new THREE.TubeGeometry(
        envelopeCurve,
        Math.max(48, layout.affectedPolygon.length * 6),
        clamp(layout.spanM * 0.0032, 0.28, 0.58),
        7,
        true,
      );
      affectedTubeMaterial = new THREE.MeshBasicMaterial({
        color: "#fb923c",
        transparent: true,
        opacity: 0.92,
        depthTest: false,
        blending: THREE.AdditiveBlending,
      });
      const affectedTube = new THREE.Mesh(affectedTubeGeometry, affectedTubeMaterial);
      affectedTube.renderOrder = 17;
      scene.add(affectedTube);
    }

    const freeFaceDirection = new THREE.Vector3(layout.forward.x, 0, -layout.forward.y).normalize();
    const arrowOriginPlan = pointFromBenchAxes(
      layout,
      layout.floorFrontM - Math.max(8, layout.influenceRadiusM),
      0,
    );
    const arrowOrigin = toScenePoint(
      { ...arrowOriginPlan, z: layout.surfaceZ + 0.65 },
      layout,
      verticalExaggeration,
    );
    const freeFaceArrow = new THREE.ArrowHelper(
      freeFaceDirection,
      arrowOrigin,
      Math.min(layout.spanM * 0.16, 20),
      "#22d3ee",
      3,
      1.7,
    );
    scene.add(freeFaceArrow);

    const northOriginPlan = pointFromBenchAxes(
      layout,
      layout.floorBackM + Math.max(6, layout.influenceRadiusM * 0.6),
      layout.floorRightM - Math.max(6, layout.influenceRadiusM * 0.6),
    );
    const northOrigin = toScenePoint(
      { ...northOriginPlan, z: layout.surfaceZ + 0.65 },
      layout,
      verticalExaggeration,
    );
    const northArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, -1),
      northOrigin,
      Math.min(layout.spanM * 0.11, 13),
      "#f8fafc",
      2.4,
      1.4,
    );
    scene.add(northArrow);

    const labelSprites: THREE.Sprite[] = [];
    const labelWidth = clamp(layout.spanM * 0.22, 18, 34);
    const benchLabel = makeTextSprite(
      `ACTIVE BENCH · RL ${format(layout.surfaceZ, 0)} m`,
      "#dbeafe",
      false,
      labelWidth,
    );
    if (benchLabel) {
      const labelPlan = pointFromBenchAxes(layout, layout.floorBackM, 0);
      benchLabel.position.copy(
        toScenePoint(
          { ...labelPlan, z: layout.surfaceZ + layout.highwallHeightM + 2 },
          layout,
          verticalExaggeration,
        ),
      );
      labelSprites.push(benchLabel);
      scene.add(benchLabel);
    }
    const faceLabel = makeTextSprite("FREE FACE · MOVEMENT", "#a5f3fc", true, labelWidth * 0.82);
    if (faceLabel) {
      const labelPlan = pointFromBenchAxes(layout, layout.floorFrontM, 0);
      faceLabel.position.copy(
        toScenePoint({ ...labelPlan, z: layout.surfaceZ + 2.2 }, layout, verticalExaggeration),
      );
      labelSprites.push(faceLabel);
      scene.add(faceLabel);
    }
    const northLabel = makeTextSprite("N", "#f8fafc", false, 7);
    if (northLabel) {
      northLabel.position.copy(northOrigin.clone().add(new THREE.Vector3(0, 1.5, -Math.min(layout.spanM * 0.13, 15))));
      labelSprites.push(northLabel);
      scene.add(northLabel);
    }

    const holePoints = holes.map((hole) => ({
      hole,
      point: toScenePoint(
        { x: hole.x, y: hole.y, z: Number.isFinite(hole.z) ? (hole.z as number) : layout.surfaceZ },
        layout,
        verticalExaggeration,
      ),
    }));
    const orderedHoles = [...holePoints].sort(
      (left, right) =>
        (left.hole.delayMs ?? Number.POSITIVE_INFINITY) -
          (right.hole.delayMs ?? Number.POSITIVE_INFINITY) ||
        left.hole.id.localeCompare(right.hole.id),
    );
    const delayTimes = orderedHoles
      .map(({ hole }) => hole.delayMs)
      .filter((delay): delay is number => Number.isFinite(delay));
    const delayAt = (timelineProgress: number) => {
      if (!delayTimes.length) return Number.NEGATIVE_INFINITY;
      if (timelineProgress >= 0.999) return Number.POSITIVE_INFINITY;
      return delayTimes[Math.floor(clamp(timelineProgress, 0, 1) * Math.max(0, delayTimes.length - 1))];
    };

    const holeGeometry = new THREE.SphereGeometry(1.1, 10, 7);
    const holeMaterial = new THREE.MeshBasicMaterial({
      color: "#ffffff",
      vertexColors: true,
      depthTest: false,
      depthWrite: false,
    });
    const holeMesh = new THREE.InstancedMesh(holeGeometry, holeMaterial, holePoints.length);
    const matrix = new THREE.Matrix4();
    holePoints.forEach(({ point }, index) => {
      matrix.makeTranslation(point.x, point.y + 2.4, point.z);
      holeMesh.setMatrixAt(index, matrix);
    });
    holeMesh.instanceMatrix.needsUpdate = true;
    holeMesh.renderOrder = 12;
    scene.add(holeMesh);

    const stemPositions: number[] = [];
    holePoints.forEach(({ hole, point }) => {
      stemPositions.push(
        point.x,
        point.y + 1.2,
        point.z,
        point.x,
        point.y - clamp(hole.depth ?? 12, 6, 20) * verticalExaggeration,
        point.z,
      );
    });
    const stemGeometry = new THREE.BufferGeometry();
    stemGeometry.setAttribute("position", new THREE.Float32BufferAttribute(stemPositions, 3));
    const stemMaterial = new THREE.LineBasicMaterial({ color: "#dbeafe", opacity: 0.24, transparent: true });
    scene.add(new THREE.LineSegments(stemGeometry, stemMaterial));

    const tiePositions: number[] = [];
    orderedHoles.slice(1).forEach((current, index) => {
      const previous = orderedHoles[index];
      tiePositions.push(
        previous.point.x,
        previous.point.y + 2.2,
        previous.point.z,
        current.point.x,
        current.point.y + 2.2,
        current.point.z,
      );
    });
    const tieGeometry = new THREE.BufferGeometry();
    tieGeometry.setAttribute("position", new THREE.Float32BufferAttribute(tiePositions, 3));
    const tieColors = new Float32Array(tiePositions.length);
    tieGeometry.setAttribute("color", new THREE.BufferAttribute(tieColors, 3));
    const tieMaterial = new THREE.LineBasicMaterial({
      color: "#ffffff",
      vertexColors: true,
      transparent: true,
      opacity: 0.88,
      depthTest: false,
      depthWrite: false,
    });
    const tieLines = new THREE.LineSegments(tieGeometry, tieMaterial);
    tieLines.renderOrder = 11;
    scene.add(tieLines);

    let blockMesh: THREE.InstancedMesh | null = null;
    let blockGeometry: THREE.BoxGeometry | null = null;
    let blockMaterial: THREE.MeshStandardMaterial | null = null;
    const resultBlocks = result?.blocks ?? [];
    const lodScale = Math.max(1, Math.cbrt(result?.transport?.stride || 1));
    const voxelSize = (result?.assumptions.cell_size_m ?? 1) * lodScale * (1 - seamPercent / 100);
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
      const planXs = layout.affectedPolygon.map((point) => point.x - layout.centerX);
      const minX = Math.min(...planXs);
      const maxX = Math.max(...planXs);
      const cutoff = minX + (maxX - minX) * (clipPercent / 100);
      clipPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), cutoff);
    }

    if (resultBlocks.length) {
      blockGeometry = new THREE.BoxGeometry(voxelSize, voxelSize * verticalExaggeration, voxelSize);
      blockMaterial = new THREE.MeshStandardMaterial({
        color: "#ffffff",
        vertexColors: true,
        roughness: 0.58,
        metalness: 0.03,
        emissive: "#07131d",
        emissiveIntensity: 0.34,
        clippingPlanes: clipPlane ? [clipPlane] : [],
      });
      blockMesh = new THREE.InstancedMesh(blockGeometry, blockMaterial, resultBlocks.length);
      blockMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      blockMesh.renderOrder = 10;
      scene.add(blockMesh);
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

    const pulseGeometry = new THREE.RingGeometry(0.72, 1, 48);
    pulseGeometry.rotateX(-Math.PI / 2);
    const pulseMaterials = Array.from({ length: 3 }, () =>
      new THREE.MeshBasicMaterial({
        color: "#fbbf24",
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
      }),
    );
    const pulseRings = pulseMaterials.map((material) => {
      const ring = new THREE.Mesh(pulseGeometry, material);
      ring.visible = false;
      ring.renderOrder = 18;
      scene.add(ring);
      return ring;
    });
    const eventGlow = new THREE.PointLight("#fbbf24", 0, Math.max(24, layout.influenceRadiusM * 4), 2);
    scene.add(eventGlow);
    let activePulsePoint: THREE.Vector3 | null = null;

    const ambient = new THREE.HemisphereLight("#dbeafe", "#101923", 1.8);
    scene.add(ambient);
    const keyLight = new THREE.DirectionalLight("#fff7ed", 2.55);
    keyLight.position.set(layout.spanM * 1.2, layout.spanM * 1.8, layout.spanM * 0.9);
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight("#8ed8ef", 0.78);
    rimLight.position.set(-layout.spanM, layout.spanM * 0.55, -layout.spanM);
    scene.add(rimLight);

    let lastProgress = Number.NaN;
    const updateTimeline = (timelineProgress: number) => {
      const displayProgress = view === "source" ? 0 : view === "destination" ? 1 : timelineProgress;
      const currentDelay = delayAt(displayProgress);
      const queuedColor = new THREE.Color("#94a3b8");
      const activeColor = new THREE.Color("#fbbf24");
      const firedColor = new THREE.Color("#34d399");
      activePulsePoint = null;
      holePoints.forEach(({ hole }, index) => {
        const delay = hole.delayMs ?? Number.POSITIVE_INFINITY;
        const color =
          displayProgress >= 0.999 || delay < currentDelay
            ? firedColor
            : delay === currentDelay
              ? activeColor
              : queuedColor;
        holeMesh.setColorAt(index, color);
        if (view === "movement" && delay === currentDelay && displayProgress < 0.999) {
          activePulsePoint = holePoints[index].point.clone().add(new THREE.Vector3(0, 0.28, 0));
        }
      });
      if (holeMesh.instanceColor) holeMesh.instanceColor.needsUpdate = true;

      const colorAttribute = tieGeometry.getAttribute("color") as THREE.BufferAttribute;
      orderedHoles.slice(1).forEach((current, index) => {
        const delay = current.hole.delayMs ?? Number.POSITIVE_INFINITY;
        const color =
          displayProgress >= 0.999 || delay < currentDelay
            ? firedColor
            : delay === currentDelay
              ? activeColor
              : new THREE.Color("#67e8f9");
        colorAttribute.setXYZ(index * 2, color.r, color.g, color.b);
        colorAttribute.setXYZ(index * 2 + 1, color.r, color.g, color.b);
      });
      colorAttribute.needsUpdate = true;

      const activeBlockMesh = blockMesh;
      if (activeBlockMesh) {
        resultBlocks.forEach((block, index) => {
          const movement = movementFor(block, displayProgress);
          matrix.compose(blockPosition(block, movement), blockRotation, blockScale);
          activeBlockMesh.setMatrixAt(index, matrix);
          activeBlockMesh.setColorAt(index, colorFor(block, colorMode, movement > 0.5, colorMaximum));
        });
        activeBlockMesh.instanceMatrix.needsUpdate = true;
        if (activeBlockMesh.instanceColor) activeBlockMesh.instanceColor.needsUpdate = true;
        activeBlockMesh.computeBoundingSphere();
      }
    };
    updateTimeline(progressRef.current);

    let frame = 0;
    let isVisible = true;
    const animate = (time: number) => {
      frame = requestAnimationFrame(animate);
      const nextProgress = progressRef.current;
      if (nextProgress !== lastProgress) {
        updateTimeline(nextProgress);
        lastProgress = nextProgress;
      }
      pulseRings.forEach((ring, index) => {
        if (!activePulsePoint) {
          ring.visible = false;
          return;
        }
        const phase = ((time / 1_050) + index / pulseRings.length) % 1;
        const radius = 1.2 + phase * Math.max(layout.influenceRadiusM * 1.25, 7);
        ring.visible = true;
        ring.position.copy(activePulsePoint);
        ring.scale.set(radius, radius, radius);
        pulseMaterials[index].opacity = (1 - phase) * 0.48;
      });
      if (activePulsePoint) {
        eventGlow.position.copy(activePulsePoint).add(new THREE.Vector3(0, 2, 0));
        eventGlow.intensity = 2.4 + Math.sin(time / 90) * 0.7;
      } else {
        eventGlow.intensity = 0;
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
      labelSprites.forEach((sprite) => {
        sprite.material.map?.dispose();
        sprite.material.dispose();
      });
      [freeFaceArrow, northArrow].forEach((arrow) => {
        arrow.line.geometry.dispose();
        (arrow.line.material as THREE.Material).dispose();
        arrow.cone.geometry.dispose();
        (arrow.cone.material as THREE.Material).dispose();
      });
      floorGeometry.dispose();
      floorMaterial.dispose();
      gridGeometry.dispose();
      gridMaterial.dispose();
      highwallGeometry.dispose();
      highwallMaterial.dispose();
      strataGeometry.dispose();
      strataMaterial.dispose();
      freeFaceGeometry.dispose();
      freeFaceMaterial.dispose();
      floorOutlineGeometry.dispose();
      floorOutlineMaterial.dispose();
      patternGeometry.dispose();
      patternMaterial.dispose();
      patternOutlineGeometry.dispose();
      patternOutlineMaterial.dispose();
      affectedGeometry?.dispose();
      affectedMaterial?.dispose();
      affectedOutlineGeometry?.dispose();
      affectedOutlineMaterial?.dispose();
      affectedTubeGeometry?.dispose();
      affectedTubeMaterial?.dispose();
      holeGeometry.dispose();
      holeMaterial.dispose();
      stemGeometry.dispose();
      stemMaterial.dispose();
      tieGeometry.dispose();
      tieMaterial.dispose();
      blockGeometry?.dispose();
      blockMaterial?.dispose();
      vectorGeometry?.dispose();
      vectorMaterial?.dispose();
      arrowheadGeometry?.dispose();
      arrowheadMaterial?.dispose();
      pulseGeometry.dispose();
      pulseMaterials.forEach((material) => material.dispose());
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
        data-testid="geomotion-active-bench-scene"
        aria-label={`Active blast bench only, approximately ${format(layout.widthM, 0)} by ${format(layout.lengthM, 0)} metres, showing ${holes.length} ordered holes, the modelled rock footprint, free-face direction, and ${result ? "the predicted bulk-movement envelope" : "the planned influence envelope"}.`}
      />
      <div className="geomotionSceneHud geomotionSceneHudLeft" aria-hidden="true">
        <strong>ACTIVE BLAST BENCH ONLY</strong>
        <span>Collar RL {format(layout.surfaceZ, 0)} m · {format(layout.widthM, 0)} × {format(layout.lengthM, 0)} m movement footprint</span>
        <span>{holes.length} holes · free face {format(assumptions.free_face_azimuth_deg, 0)}° · north arrow shown</span>
      </div>
      <div className="geomotionSceneHud geomotionSceneHudRight" aria-live="polite">
        <strong>{result ? "PREDICTED BULK-MOVEMENT ENVELOPE" : "PLANNED ROCK INFLUENCE"}</strong>
        <span>{holes.length} holes · {result ? `${result.blocks.length.toLocaleString()} movement cells` : "movement not run"}</span>
        <span>{view === "movement" && currentDelay != null ? `${currentDelay.toFixed(0)} ms · ${firedCount}/${delayValues.length} fired` : view === "destination" ? "Post-blast state" : "Pre-blast state"}</span>
        <small>{result ? `Envelope includes ${format(layout.uncertaintyBufferM, 1)} m P95 uncertainty buffer` : `Influence radius ${format(layout.influenceRadiusM, 1)} m`} · uncalibrated</small>
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
          aria-label="Reset and fit active blast bench"
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
  const [colorMode, setColorMode] = useState<GeoMotionColor>("displacement");
  const [showVectors, setShowVectors] = useState(true);
  const [verticalExaggeration, setVerticalExaggeration] = useState(1.5);
  const [clipPercent, setClipPercent] = useState(100);
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>("perspective");
  const [seamPercent, setSeamPercent] = useState(1);
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
  const benchLayout = useMemo(
    () => buildBenchLayout(holes, result, assumptions),
    [holes, result, assumptions],
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
          assumptions: { ...request.assumptions, cell_size_m: 3, max_visual_blocks: 20000 },
        };
        setResult(simulateGeoMotionLocally(previewRequest));
        setColorMode("displacement");
        const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
        setProgress(reducedMotion ? 1 : 0);
        setView(reducedMotion ? "destination" : "movement");
        setPlaying(!reducedMotion);
        return;
      }
      if (!response.ok) throw new Error(payload?.detail?.[0]?.msg || payload?.detail || `Simulation failed (${response.status})`);
      setResult(payload as GeoMotionResult);
      setColorMode("displacement");
      const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
      setProgress(reducedMotion ? 1 : 0);
      setView(reducedMotion ? "destination" : "movement");
      setPlaying(!reducedMotion);
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
            {standalone ? <div className="geomotionEyebrow">GEOMOTION 3D · ACTIVE BENCH WORKSPACE</div> : null}
            <div className="sectionTitle">Active blast bench and movement envelope</div>
            <div className="subtitle">
              Bench-local view of the real tie-up coordinates, affected rock volume, free-face direction and predicted bulk movement. No whole-mine geometry is rendered.
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

        <div className="geomotionContextRow" aria-label="Active bench summary">
          <span className="mode" data-testid="geomotion-bench-scope">Active bench only</span>
          <span>Collar RL {format(benchLayout.surfaceZ, 0)} m</span>
          <span>{format(benchLayout.widthM, 0)} × {format(benchLayout.lengthM, 0)} m envelope</span>
          <span>{format(benchLayout.areaM2, 0)} m² plan area</span>
          <span>Free face {format(assumptions.free_face_azimuth_deg, 0)}°</span>
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
              <option value={0}>Joined</option><option value={1}>1%</option><option value={2}>2%</option><option value={3}>3%</option>
            </select>
          </label>
          <label className="geomotionCheckbox">
            <input type="checkbox" checked={showVectors} onChange={(event) => setShowVectors(event.target.checked)} disabled={!result} />
            Movement vectors
          </label>
        </div>

        <div className="geomotionPitLegend" aria-label="Active bench, movement-zone and firing-state legend">
          <span><i className="benchSurface" />Bench surface</span>
          <span><i className="patternZone" />Modelled rock footprint</span>
          {result ? <span><i className="movementZone" />Predicted movement envelope</span> : null}
          <span><i className="freeFace" />Free-face direction</span>
          <span><i className="tie" />Delay tie line</span>
          <span><i className="queued" />Queued hole</span>
          <span><i className="current" />Current firing</span>
          <span><i className="fired" />Fired hole</span>
          <span className="geomotionLegendNote">Movement envelope is uncalibrated bulk-rock displacement—not flyrock, damage or an exclusion zone.</span>
        </div>
        {result ? (
          <ColorLegend mode={colorMode} blocks={result.blocks} destination={view === "destination" || (view === "movement" && progress > 0.5)} />
        ) : null}

        {result ? (
          <div className="geomotionTimeline">
            <span>Bench cutaway</span>
            <input aria-label="Active bench cutaway" type="range" min={5} max={100} value={clipPercent} onChange={(event) => setClipPercent(Number(event.target.value))} />
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
          assumptions={assumptions}
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
          <span><strong>Local blast coordinates in metres</strong> · north and free-face direction shown</span>
          <span>Planning/simulation only · no detonator, firing-system, or hardware control</span>
        </div>
        <span className="geomotionSrOnly" aria-live="polite">
          {`Active blast bench view${fullscreen ? " in browser fullscreen" : standalone ? " in the dedicated window" : ""}. ${result ? "Predicted movement envelope is visible." : "Run the movement model to show the predicted movement envelope."}`}
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
          <h3>The active blast bench is ready</h3>
          <p>Run the movement model to add the predicted bulk-rock movement envelope, 1 m³ ore-control cells, directional vectors, mass balance, recovery, and dilution outcomes—without loading unrelated mine geometry.</p>
        </section>
      )}
    </div>
  );
}
