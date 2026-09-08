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
import {
  ACTIVE_BENCH,
  BENCH_LABELS,
  PIT_CONTEXT,
  PIT_SURFACE_CELLS,
  pointOnPit,
} from "./mineContext";
import type { MineMaterial, MinePoint3D } from "./mineContext";

const NOTICE = "Synthetic Demonstration / Uncalibrated — Planning Only";
const STORAGE_KEY = "geomotion_3d_demo_v1";
const DEFAULT_TIE_UP_FILE = "680-665QS32-33_synthetic_reference.csv";
const PIT_VERTICAL_ORIGIN = 690;

const MINE_MATERIAL_COLORS: Record<MineMaterial, string> = {
  waste: "#64748b",
  oxide: "#b7793f",
  transition: "#6d8f70",
  ore: "#2f9c88",
};

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
  const titleByMode: Partial<Record<GeoMotionColor, string>> = {
    grade: "Low grade → High grade",
    displacement: "Low movement → High movement",
    uncertainty: "Low uncertainty → High uncertainty",
    burdenVelocity: "Low velocity → High velocity",
    impulse: "Low impulse → High impulse",
  };
  return <div className="geomotionLegend"><span className="geomotionGradient" />{titleByMode[mode] ?? "Model value"}</div>;
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function toScenePoint(point: MinePoint3D, verticalExaggeration: number) {
  return new THREE.Vector3(
    point.x,
    (point.z - PIT_VERTICAL_ORIGIN) * verticalExaggeration,
    -point.y,
  );
}

function makeTextSprite(text: string, color: string, active = false) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.fillStyle = active ? "rgba(69,36,5,.94)" : "rgba(5,12,23,.86)";
  context.strokeStyle = active ? "rgba(251,191,36,.88)" : "rgba(148,163,184,.42)";
  context.lineWidth = 4;
  context.beginPath();
  context.roundRect(3, 3, 506, 122, 18);
  context.fill();
  context.stroke();
  context.fillStyle = color;
  context.font = `800 ${active ? 41 : 37}px ui-sans-serif, system-ui, sans-serif`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, 256, 67);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(active ? 135 : 96, active ? 34 : 24, 1);
  sprite.renderOrder = 20;
  return sprite;
}

function setCameraPosition(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  preset: "perspective" | "plan" | "section",
) {
  controls.target.set(0, -28, 0);
  if (preset === "plan") camera.position.set(0, 980, 0.01);
  else if (preset === "section") camera.position.set(850, 35, 0);
  else camera.position.set(610, 390, 650);
  camera.lookAt(controls.target);
  controls.update();
}

interface PlanBounds {
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
  maxSourceZ: number;
}

function buildPlanBounds(holes: BlastHole[], result: GeoMotionResult | null): PlanBounds {
  const blockPoints = result?.blocks.map((block) => block.source) ?? [];
  const xs = blockPoints.length ? blockPoints.map((point) => point[0]) : holes.map((hole) => hole.x);
  const ys = blockPoints.length ? blockPoints.map((point) => point[1]) : holes.map((hole) => hole.y);
  const zs = blockPoints.length
    ? blockPoints.map((point) => point[2])
    : holes.map((hole) => hole.z).filter((value): value is number => Number.isFinite(value));
  return {
    xmin: xs.length ? Math.min(...xs) : 0,
    xmax: xs.length ? Math.max(...xs) : 1,
    ymin: ys.length ? Math.min(...ys) : 0,
    ymax: ys.length ? Math.max(...ys) : 1,
    maxSourceZ: zs.length ? Math.max(...zs) : ACTIVE_BENCH.elevation,
  };
}

function mapToActiveBench(
  x: number,
  y: number,
  bounds: PlanBounds,
  verticalExaggeration: number,
) {
  const u = clamp((x - bounds.xmin) / Math.max(bounds.xmax - bounds.xmin, 1), 0, 1);
  const v = clamp((y - bounds.ymin) / Math.max(bounds.ymax - bounds.ymin, 1), 0, 1);
  const angleInset = 0.025;
  const radiusInset = 0.007;
  const angle =
    ACTIVE_BENCH.startAngle +
    angleInset +
    u * (ACTIVE_BENCH.endAngle - ACTIVE_BENCH.startAngle - angleInset * 2);
  const radius =
    ACTIVE_BENCH.innerRadius +
    radiusInset +
    v * (ACTIVE_BENCH.outerRadius - ACTIVE_BENCH.innerRadius - radiusInset * 2);
  return {
    angle,
    point: toScenePoint(pointOnPit(radius, angle, ACTIVE_BENCH.elevation + 1.4), verticalExaggeration),
  };
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
  cameraPreset: "perspective" | "plan" | "section";
  seamPercent: number;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const progressRef = useRef(progress);
  progressRef.current = progress;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const width = Math.max(host.clientWidth, 320);
    const height = Math.max(host.clientHeight, 460);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#07101c");
    scene.fog = new THREE.Fog("#07101c", 760, 1_650);
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.5, 3_000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.setSize(width, height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.localClippingEnabled = clipPercent < 99;
    renderer.domElement.setAttribute("aria-hidden", "true");
    host.replaceChildren(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.enablePan = true;
    controls.screenSpacePanning = true;
    controls.minDistance = 150;
    controls.maxDistance = 1_650;
    controls.minPolarAngle = 0.04;
    controls.maxPolarAngle = Math.PI * 0.54;
    setCameraPosition(camera, controls, cameraPreset);
    cameraRef.current = camera;
    controlsRef.current = controls;

    const pitPositions: number[] = [];
    const pitColors: number[] = [];
    const edgePositions: number[] = [];
    PIT_SURFACE_CELLS.forEach((cell) => {
      const points = cell.points.map((point) => toScenePoint(point, verticalExaggeration));
      const base = new THREE.Color(cell.active ? "#f59e0b" : MINE_MATERIAL_COLORS[cell.material]);
      if (cell.kind === "wall") base.multiplyScalar(0.68);
      else if (cell.kind === "terrain") base.multiplyScalar(0.82);
      const triangles = points.length === 3 ? [0, 1, 2] : [0, 1, 2, 0, 2, 3];
      triangles.forEach((index) => {
        const point = points[index];
        pitPositions.push(point.x, point.y, point.z);
        pitColors.push(base.r, base.g, base.b);
      });
      points.forEach((point, index) => {
        const next = points[(index + 1) % points.length];
        edgePositions.push(point.x, point.y + 0.08, point.z, next.x, next.y + 0.08, next.z);
      });
    });
    const pitGeometry = new THREE.BufferGeometry();
    pitGeometry.setAttribute("position", new THREE.Float32BufferAttribute(pitPositions, 3));
    pitGeometry.setAttribute("color", new THREE.Float32BufferAttribute(pitColors, 3));
    pitGeometry.computeVertexNormals();
    const pitMaterial = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.92,
      metalness: 0.02,
      side: THREE.DoubleSide,
    });
    const pitMesh = new THREE.Mesh(pitGeometry, pitMaterial);
    scene.add(pitMesh);

    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute("position", new THREE.Float32BufferAttribute(edgePositions, 3));
    const edgeMaterial = new THREE.LineBasicMaterial({
      color: "#dbeafe",
      opacity: 0.17,
      transparent: true,
    });
    scene.add(new THREE.LineSegments(edgeGeometry, edgeMaterial));

    const groundGeometry = new THREE.RingGeometry(650, 1_050, 80);
    groundGeometry.rotateX(-Math.PI / 2);
    groundGeometry.scale(1, 1, PIT_CONTEXT.lengthM / PIT_CONTEXT.widthM);
    const groundMaterial = new THREE.MeshStandardMaterial({
      color: "#17232c",
      roughness: 1,
      side: THREE.DoubleSide,
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.position.y = (760 - PIT_VERTICAL_ORIGIN) * verticalExaggeration - 0.5;
    scene.add(ground);

    const activeOutlinePoints = [
      pointOnPit(ACTIVE_BENCH.outerRadius + 0.006, ACTIVE_BENCH.startAngle, ACTIVE_BENCH.elevation + 2.4),
      pointOnPit(ACTIVE_BENCH.outerRadius + 0.006, ACTIVE_BENCH.endAngle, ACTIVE_BENCH.elevation + 2.4),
      pointOnPit(ACTIVE_BENCH.innerRadius - 0.006, ACTIVE_BENCH.endAngle, ACTIVE_BENCH.elevation + 2.4),
      pointOnPit(ACTIVE_BENCH.innerRadius - 0.006, ACTIVE_BENCH.startAngle, ACTIVE_BENCH.elevation + 2.4),
    ].map((point) => toScenePoint(point, verticalExaggeration));
    const activeOutlineGeometry = new THREE.BufferGeometry().setFromPoints([
      ...activeOutlinePoints,
      activeOutlinePoints[0],
    ]);
    const activeOutlineMaterial = new THREE.LineDashedMaterial({
      color: "#fbbf24",
      dashSize: 8,
      gapSize: 5,
      linewidth: 2,
    });
    const activeOutline = new THREE.Line(activeOutlineGeometry, activeOutlineMaterial);
    activeOutline.computeLineDistances();
    scene.add(activeOutline);

    const labelSprites: THREE.Sprite[] = [];
    BENCH_LABELS.forEach((label) => {
      const sprite = makeTextSprite(label.label, label.active ? "#fde68a" : "#dbeafe", label.active);
      if (!sprite) return;
      sprite.position.copy(toScenePoint(label.point, verticalExaggeration));
      sprite.position.y += label.active ? 17 : 11;
      labelSprites.push(sprite);
      scene.add(sprite);
    });
    const activeLabel = makeTextSprite(`${ACTIVE_BENCH.label} · ${ACTIVE_BENCH.section}`, "#fde68a", true);
    if (activeLabel) {
      activeLabel.position.copy(
        toScenePoint(
          pointOnPit(
            ACTIVE_BENCH.outerRadius + 0.16,
            (ACTIVE_BENCH.startAngle + ACTIVE_BENCH.endAngle) / 2,
            ACTIVE_BENCH.elevation + 38,
          ),
          verticalExaggeration,
        ),
      );
      activeLabel.scale.set(176, 42, 1);
      labelSprites.push(activeLabel);
      scene.add(activeLabel);
    }

    const bounds = buildPlanBounds(holes, result);
    const holePoints = holes.map((hole) => {
      const mapped = mapToActiveBench(hole.x, hole.y, bounds, verticalExaggeration);
      mapped.point.y += Number.isFinite(hole.z)
        ? clamp(((hole.z as number) - bounds.maxSourceZ) * verticalExaggeration, -3, 3)
        : 0;
      return { hole, ...mapped };
    });
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

    const holeGeometry = new THREE.SphereGeometry(4.1, 12, 8);
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
    const rawSpanX = Math.max(bounds.xmax - bounds.xmin, 1);
    const rawSpanY = Math.max(bounds.ymax - bounds.ymin, 1);
    const mappedArc = PIT_CONTEXT.widthM * ACTIVE_BENCH.outerRadius * (ACTIVE_BENCH.endAngle - ACTIVE_BENCH.startAngle);
    const mappedRadial = PIT_CONTEXT.widthM * (ACTIVE_BENCH.outerRadius - ACTIVE_BENCH.innerRadius) * 0.5;
    const planScale = clamp((mappedArc / rawSpanX + mappedRadial / rawSpanY) / 2, 0.65, 1.6);
    const lodScale = Math.max(1, Math.cbrt(result?.transport?.stride || 1));
    const voxelSize = (result?.assumptions.cell_size_m ?? 1) * lodScale * planScale * (1 - seamPercent / 100);
    const midpointAngle = (ACTIVE_BENCH.startAngle + ACTIVE_BENCH.endAngle) / 2;
    const blockRotation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -midpointAngle);
    const blockScale = new THREE.Vector3(1, 1, 1);

    const blockPosition = (block: GeoMotionBlock, movement: number) => {
      const mapped = mapToActiveBench(block.source[0], block.source[1], bounds, verticalExaggeration);
      mapped.point.y -= Math.max(0, bounds.maxSourceZ - block.source[2]) * verticalExaggeration;
      if (movement > 0) {
        const tangent = new THREE.Vector3(-Math.sin(mapped.angle), 0, -Math.cos(mapped.angle)).normalize();
        const radial = new THREE.Vector3(Math.cos(mapped.angle), 0, -Math.sin(mapped.angle)).normalize();
        mapped.point.addScaledVector(tangent, block.vector[0] * movement);
        mapped.point.addScaledVector(radial, block.vector[1] * movement);
        mapped.point.y += block.vector[2] * movement * verticalExaggeration;
      }
      return mapped.point;
    };
    const movementFor = (block: GeoMotionBlock, timelineProgress: number) => {
      if (view === "source") return 0;
      if (view === "destination") return 1;
      const minimumDelay = delayTimes[0] ?? 0;
      const maximumDelay = delayTimes[delayTimes.length - 1] ?? minimumDelay + 1;
      const currentDelay = minimumDelay + (maximumDelay - minimumDelay) * timelineProgress;
      const movementWindow = Math.max(18, (maximumDelay - minimumDelay) * 0.035);
      return clamp((currentDelay - block.effective_time_ms) / movementWindow, 0, 1);
    };

    let clipPlane: THREE.Plane | null = null;
    if (resultBlocks.length && clipPercent < 99) {
      const planPoints = [
        mapToActiveBench(bounds.xmin, bounds.ymin, bounds, verticalExaggeration).point,
        mapToActiveBench(bounds.xmax, bounds.ymax, bounds, verticalExaggeration).point,
      ];
      const minX = Math.min(...planPoints.map((point) => point.x));
      const maxX = Math.max(...planPoints.map((point) => point.x));
      const cutoff = minX + (maxX - minX) * (clipPercent / 100);
      clipPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), cutoff);
    }

    if (resultBlocks.length) {
      blockGeometry = new THREE.BoxGeometry(voxelSize, voxelSize * verticalExaggeration, voxelSize);
      blockMaterial = new THREE.MeshStandardMaterial({
        color: "#ffffff",
        vertexColors: true,
        roughness: 0.76,
        metalness: 0.02,
        clippingPlanes: clipPlane ? [clipPlane] : [],
      });
      blockMesh = new THREE.InstancedMesh(blockGeometry, blockMaterial, resultBlocks.length);
      blockMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      blockMesh.renderOrder = 10;
      scene.add(blockMesh);
    }

    let vectorGeometry: THREE.BufferGeometry | null = null;
    let vectorMaterial: THREE.LineBasicMaterial | null = null;
    if (showVectors && resultBlocks.length) {
      const vectorPositions: number[] = [];
      const step = Math.max(1, Math.ceil(resultBlocks.length / 450));
      for (let index = 0; index < resultBlocks.length; index += step) {
        const block = resultBlocks[index];
        const source = blockPosition(block, 0);
        const destination = blockPosition(block, 1);
        vectorPositions.push(source.x, source.y, source.z, destination.x, destination.y, destination.z);
      }
      vectorGeometry = new THREE.BufferGeometry();
      vectorGeometry.setAttribute("position", new THREE.Float32BufferAttribute(vectorPositions, 3));
      vectorMaterial = new THREE.LineBasicMaterial({ color: "#fb923c", opacity: 0.62, transparent: true });
      scene.add(new THREE.LineSegments(vectorGeometry, vectorMaterial));
    }

    const ambient = new THREE.HemisphereLight("#dbeafe", "#17232c", 2.3);
    scene.add(ambient);
    const keyLight = new THREE.DirectionalLight("#ffffff", 3.2);
    keyLight.position.set(360, 620, 220);
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight("#67e8f9", 1.25);
    rimLight.position.set(-420, 180, -360);
    scene.add(rimLight);

    let lastProgress = Number.NaN;
    const updateTimeline = (timelineProgress: number) => {
      const displayProgress = view === "source" ? 0 : view === "destination" ? 1 : timelineProgress;
      const currentDelay = delayAt(displayProgress);
      const queuedColor = new THREE.Color("#94a3b8");
      const activeColor = new THREE.Color("#fbbf24");
      const firedColor = new THREE.Color("#34d399");
      holePoints.forEach(({ hole }, index) => {
        const delay = hole.delayMs ?? Number.POSITIVE_INFINITY;
        const color =
          displayProgress >= 0.999 || delay < currentDelay
            ? firedColor
            : delay === currentDelay
              ? activeColor
              : queuedColor;
        holeMesh.setColorAt(index, color);
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
          activeBlockMesh.setColorAt(index, colorFor(block, colorMode, movement > 0.5));
        });
        activeBlockMesh.instanceMatrix.needsUpdate = true;
        if (activeBlockMesh.instanceColor) activeBlockMesh.instanceColor.needsUpdate = true;
        activeBlockMesh.computeBoundingSphere();
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
    animate();

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
      pitGeometry.dispose();
      pitMaterial.dispose();
      edgeGeometry.dispose();
      edgeMaterial.dispose();
      groundGeometry.dispose();
      groundMaterial.dispose();
      activeOutlineGeometry.dispose();
      activeOutlineMaterial.dispose();
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
      renderer.renderLists.dispose();
      renderer.dispose();
      host.replaceChildren();
    };
  }, [result, holes, view, colorMode, showVectors, verticalExaggeration, clipPercent, cameraPreset, seamPercent]);

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
      setCameraPosition(cameraRef.current, controlsRef.current, cameraPreset);
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

  return (
    <div className="geomotionSceneFrame">
      <div
        ref={hostRef}
        className="geomotionScene"
        role="img"
        data-testid="geomotion-whole-pit-scene"
        aria-label={`Interactive whole-mine 3D model of ${PIT_CONTEXT.name}, with the imported tie-up localized to ${ACTIVE_BENCH.label}, ${ACTIVE_BENCH.section}.`}
      />
      <div className="geomotionSceneHud geomotionSceneHudLeft" aria-hidden="true">
        <strong>WHOLE-MINE CONTEXT</strong>
        <span>{PIT_CONTEXT.benches} stepped benches · {PIT_CONTEXT.verticalRangeM} m vertical</span>
        <span>{PIT_CONTEXT.widthM / 1_000} × {(PIT_CONTEXT.lengthM / 1_000).toFixed(2)} km pit extent</span>
      </div>
      <div className="geomotionSceneHud geomotionSceneHudRight" aria-live="polite">
        <strong>ACTIVE TIE-UP · {ACTIVE_BENCH.label}</strong>
        <span>{holes.length} holes · {result ? `${result.blocks.length.toLocaleString()} movement cells` : "movement not run"}</span>
        <span>{view === "movement" && currentDelay != null ? `${currentDelay.toFixed(0)} ms · ${firedCount}/${delayValues.length} fired` : view === "destination" ? "Post-blast state" : "Pre-blast state"}</span>
        <small>Planning playback only · no device connection</small>
      </div>
      <div className="geomotionSceneButtons" role="group" aria-label="3D camera controls">
        <button type="button" onClick={() => zoom(0.82)} aria-label="Zoom in">+</button>
        <button type="button" onClick={() => zoom(1.22)} aria-label="Zoom out">−</button>
        <button type="button" onClick={resetCamera} aria-label="Reset and fit whole pit">Fit</button>
      </div>
      <div className="geomotionOrbitHint" aria-hidden="true">Drag to orbit · Right-drag to pan · Wheel or +/− to zoom</div>
    </div>
  );
}

export function GeoMotionPanel({ apiBaseUrl, token }: Props) {
  const [projectName, setProjectName] = useState("680-665QS32-33 Diamond Demonstration");
  const [holes, setHoles] = useState<BlastHole[]>(() => DEFAULT_TIE_UP.holes.map((hole) => ({ ...hole })));
  const [fileName, setFileName] = useState(DEFAULT_TIE_UP_FILE);
  const [issues, setIssues] = useState<ValidationIssue[]>(() => [...DEFAULT_TIE_UP.issues]);
  const [inputErrors, setInputErrors] = useState<string[]>(() => [...DEFAULT_TIE_UP.errors]);
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
  const [showVectors, setShowVectors] = useState(true);
  const [verticalExaggeration, setVerticalExaggeration] = useState(1);
  const [clipPercent, setClipPercent] = useState(100);
  const [cameraPreset, setCameraPreset] = useState<"perspective" | "plan" | "section">("perspective");
  const [seamPercent, setSeamPercent] = useState(1);
  const [datasetRefs, setDatasetRefs] = useState<GeoMotionRequest["site_data"]["datasets"]>([]);
  const [blockModelFile, setBlockModelFile] = useState<File | null>(null);
  const [nativeFullscreen, setNativeFullscreen] = useState(false);
  const [fallbackFullscreen, setFallbackFullscreen] = useState(false);
  const viewerRef = useRef<HTMLDivElement>(null);

  useEffect(() => localStorage.setItem(STORAGE_KEY, JSON.stringify(assumptions)), [assumptions]);

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
        setProgress(1);
        setView("destination");
        setPlaying(false);
        return;
      }
      if (!response.ok) throw new Error(payload?.detail?.[0]?.msg || payload?.detail || `Simulation failed (${response.status})`);
      setResult(payload as GeoMotionResult);
      setProgress(1);
      setView("destination");
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
    <div className="geomotionWorkspace">
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

      <section
        ref={viewerRef}
        className={`card geomotionViewerCard${fallbackFullscreen ? " geomotionViewerFallbackFullscreen" : ""}`}
        data-testid="geomotion-viewer"
      >
        <div className="geomotionViewerHeader">
          <div>
            <div className="sectionTitle">Whole-mine movement model</div>
            <div className="subtitle">
              Complete stepped pit context with the delay-bearing tie-up isolated to {ACTIVE_BENCH.label}, {ACTIVE_BENCH.section}. Drag to orbit and scroll to zoom.
            </div>
          </div>
          <div className="geomotionViewerActions">
            <button className="btn" type="button" onClick={playing ? () => setPlaying(false) : startPlayback} disabled={!timedHoles.length}>
              {playing ? "Pause sequence" : "Play sequence"}
            </button>
            <button
              className="btn btnPrimary"
              type="button"
              data-testid="geomotion-fullscreen-button"
              onClick={toggleFullscreen}
              aria-pressed={fullscreen}
            >
              {fullscreen ? "Exit fullscreen" : "View fullscreen"}
            </button>
          </div>
        </div>

        <div className="geomotionContextRow" aria-label="Mine context summary">
          <span>Whole mine · {PIT_CONTEXT.name}</span>
          <span>{PIT_CONTEXT.renderedCells} deterministic context cells</span>
          <span>{PIT_CONTEXT.benches} benches · {PIT_CONTEXT.verticalRangeM} m vertical</span>
          <span className="active">Active · {ACTIVE_BENCH.label} / {ACTIVE_BENCH.section} / {holes.length} holes</span>
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
              <option value={1}>1x</option><option value={2}>2x</option><option value={3}>3x</option>
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

        <div className="geomotionPitLegend" aria-label="Whole-mine material legend">
          {(Object.entries(MINE_MATERIAL_COLORS) as Array<[MineMaterial, string]>).map(([material, color]) => (
            <span key={material}><i style={{ background: color }} />{material === "ore" ? "Ore envelope" : material[0].toUpperCase() + material.slice(1)}</span>
          ))}
          <span><i className="active" />Active bench</span>
          <span><i className="tie" />Delay tie-up</span>
          <span className="geomotionLegendNote">Synthetic pit context · imported hole data remains the active tie-up</span>
        </div>
        {result ? (
          <ColorLegend mode={colorMode} blocks={result.blocks} destination={view === "destination" || (view === "movement" && progress > 0.5)} />
        ) : null}

        {result ? (
          <div className="geomotionTimeline">
            <span>Section clip</span>
            <input aria-label="Active section clip" type="range" min={5} max={100} value={clipPercent} onChange={(event) => setClipPercent(Number(event.target.value))} />
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
          <span><strong>{PIT_CONTEXT.coordinateSystem}</strong> · deterministic seed {PIT_CONTEXT.seed}</span>
          <span>Planning/simulation only · no detonator, firing-system, or hardware control</span>
        </div>
        <span className="geomotionSrOnly" aria-live="polite">
          {fullscreen ? "GeoMotion whole-mine fullscreen view active." : "GeoMotion whole-mine embedded view active."}
        </span>
      </section>

      <div className="geomotionTopGrid">
        <section className="card">
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
          <h3>Whole-pit context is ready</h3>
          <p>Run the movement model to add 1 m³ ore-control cells, movement vectors, mass balance, recovery, and dilution outcomes to the active bench.</p>
        </section>
      )}
    </div>
  );
}
