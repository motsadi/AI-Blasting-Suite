import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent, WheelEvent } from "react";
import type { BlastHole, ColorMode, TimingLine, TimingPattern } from "../../types/blast";
import {
  ACTIVE_BENCH,
  BENCH_LABELS,
  MINE_EXTENT_POINTS,
  PIT_MODEL,
  WHOLE_MINE_CELLS,
  pointOnPit,
  positionHolesOnActiveBench,
} from "./geomotionData";
import type { MineCell, MineMaterial, MinePoint3D, PositionedBlastHole } from "./geomotionData";

const SVG_WIDTH = 1280;
const SVG_HEIGHT = 720;
const DEFAULT_CAMERA = { yaw: 25, elevation: 38, zoom: 1 };

const MATERIAL_COLOURS: Record<MineMaterial, { top: string; wall: string; label: string }> = {
  waste: { top: "#64748b", wall: "#3f4b5d", label: "Waste" },
  oxide: { top: "#b7793f", wall: "#81522e", label: "Oxide" },
  transition: { top: "#6d8f70", wall: "#48654d", label: "Transition" },
  ore: { top: "#2f9c88", wall: "#1e6f64", label: "Ore envelope" },
};

interface Camera {
  yaw: number;
  elevation: number;
  zoom: number;
}

interface RawProjection {
  x: number;
  y: number;
  depth: number;
}

interface Geomotion3DProps {
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
  playing: boolean;
  onPlaying: (value: boolean) => void;
  onResetSimulation: () => void;
  onSelect: (id: string) => void;
  onFocusToggle: () => void;
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function rawProject(point: MinePoint3D, camera: Camera): RawProjection {
  const yaw = (camera.yaw * Math.PI) / 180;
  const elevation = (camera.elevation * Math.PI) / 180;
  const cosYaw = Math.cos(yaw);
  const sinYaw = Math.sin(yaw);
  const cosElevation = Math.cos(elevation);
  const sinElevation = Math.sin(elevation);
  const localZ = point.z - 690;
  const rotatedX = point.x * cosYaw - point.y * sinYaw;
  const rotatedY = point.x * sinYaw + point.y * cosYaw;
  return {
    x: rotatedX,
    y: rotatedY * sinElevation - localZ * cosElevation,
    depth: rotatedY * cosElevation + localZ * sinElevation,
  };
}

function pointsAttribute(points: Array<{ x: number; y: number }>) {
  return points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");
}

function numericValue(hole: BlastHole, colorMode: ColorMode) {
  if (colorMode === "delay") return hole.delayMs;
  if (colorMode === "row") return hole.rowIndex;
  if (colorMode === "charge") return hole.charge;
  if (colorMode === "depth") return hole.depth;
  if (colorMode === "fragmentation") return hole.estimatedFragmentationMm;
  if (colorMode === "ppv") return hole.estimatedPpvMmS;
  if (colorMode === "airblast") return hole.estimatedAirblastDb;
  if (colorMode === "flyrock") return hole.flyrockRisk === "high" ? 2 : hole.flyrockRisk === "moderate" ? 1 : 0;
  return hole.timingGroup
    ? Math.abs([...hole.timingGroup].reduce((sum, character) => sum + character.charCodeAt(0), 0))
    : 0;
}

function stateFor(hole: BlastHole, currentTime?: number) {
  if (currentTime == null || !Number.isFinite(hole.delayMs)) return "ready" as const;
  if ((hole.delayMs as number) < currentTime) return "fired" as const;
  if (Math.abs((hole.delayMs as number) - currentTime) < 1e-6) return "active" as const;
  return "unfired" as const;
}

function cellFill(cell: MineCell) {
  if (cell.active) return cell.kind === "wall" ? "#b76b16" : "#f59e0b";
  const palette = MATERIAL_COLOURS[cell.material];
  return cell.kind === "wall" ? palette.wall : palette.top;
}

function tieGroups(positioned: PositionedBlastHole[]) {
  if (!positioned.length) return [] as PositionedBlastHole[][];
  if (positioned.some(({ hole }) => Number.isFinite(hole.rowIndex))) {
    const groups = new Map<number, PositionedBlastHole[]>();
    positioned.forEach((item) => {
      const row = Number.isFinite(item.hole.rowIndex) ? (item.hole.rowIndex as number) : -1;
      const group = groups.get(row) ?? [];
      group.push(item);
      groups.set(row, group);
    });
    return [...groups.entries()]
      .sort(([a], [b]) => a - b)
      .map(([, group]) =>
        group.sort(
          (a, b) =>
            (a.hole.columnIndex ?? Number.POSITIVE_INFINITY) - (b.hole.columnIndex ?? Number.POSITIVE_INFINITY) ||
            a.hole.x - b.hole.x,
        ),
      );
  }
  return [
    [...positioned].sort(
      (a, b) =>
        (a.hole.firingOrder ?? Number.POSITIVE_INFINITY) - (b.hole.firingOrder ?? Number.POSITIVE_INFINITY) ||
        (a.hole.delayMs ?? Number.POSITIVE_INFINITY) - (b.hole.delayMs ?? Number.POSITIVE_INFINITY) ||
        a.hole.y - b.hole.y ||
        a.hole.x - b.hole.x,
    ),
  ];
}

function colourModeLabel(mode: ColorMode) {
  const labels: Record<ColorMode, string> = {
    delay: "Delay",
    row: "Row",
    charge: "Charge",
    depth: "Depth",
    group: "Timing group",
    fragmentation: "Fragmentation",
    ppv: "PPV",
    airblast: "Airblast",
    flyrock: "Flyrock risk",
  };
  return labels[mode];
}

export function Geomotion3D({
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
  playing,
  onPlaying,
  onResetSimulation,
  onSelect,
  onFocusToggle,
}: Geomotion3DProps) {
  const [camera, setCamera] = useState<Camera>(DEFAULT_CAMERA);
  const [nativeFullscreen, setNativeFullscreen] = useState(false);
  const [fallbackFullscreen, setFallbackFullscreen] = useState(false);
  const shellRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    yaw: number;
    elevation: number;
    moved: boolean;
  } | null>(null);
  const suppressHoleClickRef = useRef(false);

  const projection = useMemo(() => {
    const extent = MINE_EXTENT_POINTS.map((point) => rawProject(point, camera));
    const xmin = Math.min(...extent.map((point) => point.x));
    const xmax = Math.max(...extent.map((point) => point.x));
    const ymin = Math.min(...extent.map((point) => point.y));
    const ymax = Math.max(...extent.map((point) => point.y));
    const fitScale = Math.min((SVG_WIDTH - 112) / Math.max(1, xmax - xmin), (SVG_HEIGHT - 100) / Math.max(1, ymax - ymin));
    const scale = fitScale * camera.zoom;
    const centerX = SVG_WIDTH / 2 - ((xmin + xmax) / 2) * scale;
    const centerY = SVG_HEIGHT / 2 - ((ymin + ymax) / 2) * scale + 14;
    return {
      scale,
      point(point: MinePoint3D) {
        const raw = rawProject(point, camera);
        return {
          x: centerX + raw.x * scale,
          y: centerY + raw.y * scale,
          depth: raw.depth,
        };
      },
    };
  }, [camera]);

  const projectedCells = useMemo(
    () =>
      WHOLE_MINE_CELLS.map((cell) => {
        const points = cell.points.map(projection.point);
        return {
          cell,
          points,
          depth: points.reduce((sum, point) => sum + point.depth, 0) / points.length,
        };
      }).sort((a, b) => a.depth - b.depth),
    [projection],
  );

  const positionedHoles = useMemo(() => positionHolesOnActiveBench(holes), [holes]);
  const projectedHoles = useMemo(
    () =>
      positionedHoles.map((item) => ({
        ...item,
        screen: projection.point(item.collar),
        toeScreen: projection.point(item.toe),
      })),
    [positionedHoles, projection],
  );
  const projectedById = useMemo(() => new Map(projectedHoles.map((item) => [item.hole.id, item])), [projectedHoles]);
  const groups = useMemo(() => tieGroups(positionedHoles), [positionedHoles]);
  const values = useMemo(
    () => holes.map((hole) => numericValue(hole, colorMode)).filter((value): value is number => Number.isFinite(value)),
    [holes, colorMode],
  );
  const valueMin = values.length ? Math.min(...values) : 0;
  const valueMax = values.length ? Math.max(...values) : 1;
  const colourFor = (hole: BlastHole) => {
    const value = numericValue(hole, colorMode);
    if (!Number.isFinite(value)) return "#94a3b8";
    const ratio = valueMax === valueMin ? 0.5 : clamp(((value as number) - valueMin) / (valueMax - valueMin), 0, 1);
    return `hsl(${215 - ratio * 190}, 82%, ${56 + ratio * 3}%)`;
  };
  const visibleHoleIds = useMemo(
    () =>
      new Set(
        holes
          .filter((hole) => {
            const state = stateFor(hole, currentTime);
            return !((state === "fired" && !showFired) || (state === "unfired" && !showUnfired));
          })
          .map((hole) => hole.id),
      ),
    [holes, currentTime, showFired, showUnfired],
  );
  const firedCount = holes.filter((hole) => stateFor(hole, currentTime) === "fired").length;
  const activeCount = holes.filter((hole) => stateFor(hole, currentTime) === "active").length;
  const timedCount = holes.filter((hole) => Number.isFinite(hole.delayMs)).length;

  const activeOutline = useMemo(
    () =>
      [
        pointOnPit(ACTIVE_BENCH.outerRadius + 0.004, ACTIVE_BENCH.startAngle, ACTIVE_BENCH.elevation + 1),
        pointOnPit(ACTIVE_BENCH.outerRadius + 0.004, ACTIVE_BENCH.endAngle, ACTIVE_BENCH.elevation + 1),
        pointOnPit(ACTIVE_BENCH.innerRadius - 0.004, ACTIVE_BENCH.endAngle, ACTIVE_BENCH.elevation + 1),
        pointOnPit(ACTIVE_BENCH.innerRadius - 0.004, ACTIVE_BENCH.startAngle, ACTIVE_BENCH.elevation + 1),
      ].map(projection.point),
    [projection],
  );

  const selectedLinePoints = useMemo(() => {
    if (!selectedLine) return null;
    const nearest = (x: number, y: number) =>
      projectedHoles.reduce<(typeof projectedHoles)[number] | null>((best, item) => {
        if (!best) return item;
        const distance = Math.hypot(item.hole.x - x, item.hole.y - y);
        const bestDistance = Math.hypot(best.hole.x - x, best.hole.y - y);
        return distance < bestDistance ? item : best;
      }, null);
    const start = nearest(selectedLine.start.x, selectedLine.start.y);
    const end = nearest(selectedLine.end.x, selectedLine.end.y);
    return start && end ? { start: start.screen, end: end.screen } : null;
  }, [projectedHoles, selectedLine]);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setNativeFullscreen(document.fullscreenElement === shellRef.current);
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  useEffect(() => {
    if (!fallbackFullscreen) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") setFallbackFullscreen(false);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [fallbackFullscreen]);

  async function toggleFullscreen() {
    const element = shellRef.current;
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
        // Browser policy can reject fullscreen; the viewport fallback remains available.
      }
    }
    setFallbackFullscreen(true);
  }

  function handlePointerDown(event: PointerEvent<SVGSVGElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      yaw: camera.yaw,
      elevation: camera.elevation,
      moved: false,
    };
  }

  function handlePointerMove(event: PointerEvent<SVGSVGElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const dx = event.clientX - drag.startX;
    const dy = event.clientY - drag.startY;
    if (Math.hypot(dx, dy) > 3) drag.moved = true;
    if (!drag.moved) return;
    setCamera((current) => ({
      ...current,
      yaw: drag.yaw + dx * 0.32,
      elevation: clamp(drag.elevation - dy * 0.24, 16, 76),
    }));
  }

  function handlePointerEnd(event: PointerEvent<SVGSVGElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    suppressHoleClickRef.current = drag.moved;
    window.setTimeout(() => {
      suppressHoleClickRef.current = false;
    }, 0);
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }

  function handleWheel(event: WheelEvent<SVGSVGElement>) {
    event.preventDefault();
    const factor = event.deltaY > 0 ? 0.9 : 1.1;
    setCamera((current) => ({ ...current, zoom: clamp(current.zoom * factor, 0.72, 2.6) }));
  }

  function selectHole(id: string) {
    if (suppressHoleClickRef.current) {
      suppressHoleClickRef.current = false;
      return;
    }
    onSelect(id);
  }

  function selectHoleFromKeyboard(event: KeyboardEvent<SVGGElement>, id: string) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(id);
    }
  }

  const fullscreen = nativeFullscreen || fallbackFullscreen;
  const activeLabelPoint = projection.point(
    pointOnPit((ACTIVE_BENCH.outerRadius + ACTIVE_BENCH.innerRadius) / 2, ACTIVE_BENCH.endAngle + 0.02, ACTIVE_BENCH.elevation + 3),
  );
  const scaleBarLength = 200 * projection.scale;

  return (
    <div
      ref={shellRef}
      className={`card geomotion-shell${focusCanvas ? " geomotion-wide" : ""}${fallbackFullscreen ? " geomotion-fallback-fullscreen" : ""}`}
    >
      <div className="geomotion-header">
        <div>
          <div className="geomotion-kicker">GEOMOTION 3D</div>
          <div className="sectionTitle geomotion-title">Whole-mine blast context</div>
          <div className="subtitle">
            Complete synthetic open-pit block model with the delay-bearing tie-up isolated to {ACTIVE_BENCH.label}, {ACTIVE_BENCH.section}.
          </div>
        </div>
        <div className="geomotion-header-actions">
          <button className="btn" type="button" onClick={() => onPlaying(!playing)} disabled={!timedCount}>
            {playing ? "Pause playback" : "Play sequence"}
          </button>
          <button className="btn" type="button" onClick={onFocusToggle}>
            {focusCanvas ? "Show design tools" : "Wide workspace"}
          </button>
          <button className="btn btnPrimary" type="button" onClick={toggleFullscreen}>
            {fullscreen ? "Exit fullscreen" : "View fullscreen"}
          </button>
        </div>
      </div>

      <div className="geomotion-context-row" aria-label="Mine model summary">
        <span className="geomotion-context-chip">Whole mine · {PIT_MODEL.widthM / 1000} × {(PIT_MODEL.lengthM / 1000).toFixed(2)} km</span>
        <span className="geomotion-context-chip">{PIT_MODEL.blockCount} model cells · 7 stepped benches · {PIT_MODEL.verticalRangeM} m vertical</span>
        <span className="geomotion-context-chip geomotion-context-chip-active">
          Active tie-up · {holes.length} holes · {ACTIVE_BENCH.label}
        </span>
        <span className="geomotion-context-chip">Synthetic deterministic context</span>
      </div>

      <div className="geomotion-viewport">
        <svg
          className="geomotion-svg"
          viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
          role="group"
          aria-labelledby="geomotion-scene-title"
          aria-describedby="geomotion-scene-description"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerEnd}
          onPointerCancel={handlePointerEnd}
          onWheel={handleWheel}
          onDoubleClick={() => setCamera(DEFAULT_CAMERA)}
        >
          <title id="geomotion-scene-title">Geomotion 3D whole-mine open pit and active blast section</title>
          <desc id="geomotion-scene-description">
            A rotatable complete stepped open-pit block model. The active delay tie-up is highlighted on the east section of Bench 680.
          </desc>
          <defs>
            <radialGradient id="geomotion-sky" cx="48%" cy="20%" r="85%">
              <stop offset="0%" stopColor="#17385c" />
              <stop offset="58%" stopColor="#0b1a2c" />
              <stop offset="100%" stopColor="#050a13" />
            </radialGradient>
            <linearGradient id="geomotion-ground" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#182c35" stopOpacity="0.72" />
              <stop offset="100%" stopColor="#071017" stopOpacity="0.2" />
            </linearGradient>
            <filter id="geomotion-glow" x="-80%" y="-80%" width="260%" height="260%">
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <rect width={SVG_WIDTH} height={SVG_HEIGHT} fill="url(#geomotion-sky)" />
          <ellipse cx={SVG_WIDTH / 2} cy={SVG_HEIGHT * 0.61} rx={SVG_WIDTH * 0.48} ry={SVG_HEIGHT * 0.36} fill="url(#geomotion-ground)" />

          {projectedCells.map(({ cell, points }) => (
            <polygon
              key={cell.id}
              points={pointsAttribute(points)}
              fill={cellFill(cell)}
              fillOpacity={cell.active ? 0.98 : cell.kind === "wall" ? 0.91 : 0.94}
              stroke={cell.active ? "#fcd34d" : "rgba(226,232,240,0.20)"}
              strokeWidth={cell.active ? 1.5 : 0.55}
              vectorEffect="non-scaling-stroke"
            >
              <title>{`${cell.kind === "terrain" ? "Surface block" : `Bench ${cell.benchIndex}`} · ${MATERIAL_COLOURS[cell.material].label} · synthetic grade ${cell.grade.toFixed(2)}%${cell.active ? " · active blast section" : ""}`}</title>
            </polygon>
          ))}

          <polygon
            points={pointsAttribute(activeOutline)}
            fill="rgba(245,158,11,0.10)"
            stroke="#fbbf24"
            strokeWidth={3}
            strokeDasharray="8 5"
            vectorEffect="non-scaling-stroke"
          />

          {groups.map((group, groupIndex) => {
            const screenGroup = group
              .map((item) => projectedById.get(item.hole.id))
              .filter((item): item is NonNullable<typeof item> => Boolean(item && visibleHoleIds.has(item.hole.id)));
            return screenGroup.slice(1).map((item, index) => {
              const previous = screenGroup[index];
              const itemState = stateFor(item.hole, currentTime);
              const previousState = stateFor(previous.hole, currentTime);
              const active = itemState === "active" || previousState === "active";
              const fired = itemState === "fired" && previousState === "fired";
              return (
                <line
                  key={`tie-${groupIndex}-${previous.hole.id}-${item.hole.id}`}
                  x1={previous.screen.x}
                  y1={previous.screen.y}
                  x2={item.screen.x}
                  y2={item.screen.y}
                  stroke={active ? "#fbbf24" : fired ? "#34d399" : "#67e8f9"}
                  strokeWidth={active ? 3.2 : 2}
                  strokeOpacity={active ? 1 : 0.78}
                  vectorEffect="non-scaling-stroke"
                />
              );
            });
          })}

          {groups.slice(1).map((group, index) => {
            const previousGroup = groups[index];
            const from = previousGroup[previousGroup.length - 1];
            const to = group[0];
            const fromScreen = from ? projectedById.get(from.hole.id) : undefined;
            const toScreen = to ? projectedById.get(to.hole.id) : undefined;
            if (!fromScreen || !toScreen || !visibleHoleIds.has(from.hole.id) || !visibleHoleIds.has(to.hole.id)) return null;
            return (
              <line
                key={`row-tie-${from.hole.id}-${to.hole.id}`}
                x1={fromScreen.screen.x}
                y1={fromScreen.screen.y}
                x2={toScreen.screen.x}
                y2={toScreen.screen.y}
                stroke="#67e8f9"
                strokeWidth={1.7}
                strokeDasharray="5 4"
                strokeOpacity={0.68}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}

          {selectedLinePoints ? (
            <g>
              <line
                x1={selectedLinePoints.start.x}
                y1={selectedLinePoints.start.y}
                x2={selectedLinePoints.end.x}
                y2={selectedLinePoints.end.y}
                stroke="#ffffff"
                strokeWidth={3}
                strokeDasharray="8 5"
                vectorEffect="non-scaling-stroke"
              />
              <text
                x={(selectedLinePoints.start.x + selectedLinePoints.end.x) / 2}
                y={(selectedLinePoints.start.y + selectedLinePoints.end.y) / 2 - 10}
                className="geomotion-map-label"
                textAnchor="middle"
              >
                Initiation line
              </text>
            </g>
          ) : null}

          {projectedHoles.map((item, holeIndex) => {
            if (!visibleHoleIds.has(item.hole.id)) return null;
            const state = stateFor(item.hole, currentTime);
            const selected = selectedIds.includes(item.hole.id) || selectedHoleId === item.hole.id;
            const fill = state === "unfired" ? "#94a3b8" : state === "fired" ? "#34d399" : colourFor(item.hole);
            const radius = selected ? 7.5 : state === "active" ? 8.5 : 5.8;
            const showHoleLabel =
              showLabels &&
              (holes.length <= 24 ||
                camera.zoom >= 1.45 ||
                selected ||
                state === "active" ||
                (Number.isFinite(item.hole.columnIndex) ? item.hole.columnIndex === 0 : holeIndex === 0));
            const overviewLabel = holes.length > 24 && camera.zoom < 1.45;
            const holeLabel = overviewLabel ? item.hole.id.replace(/^B\d+-/, "") : item.hole.id;
            const labelRowOffset = overviewLabel ? (item.hole.rowIndex ?? 0) * 4 : 0;
            return (
              <g
                key={item.hole.id}
                data-hole="true"
                role="button"
                tabIndex={0}
                aria-label={`${item.hole.id}, delay ${item.hole.delayMs ?? "not assigned"} milliseconds`}
                className="geomotion-hole"
                onClick={() => selectHole(item.hole.id)}
                onKeyDown={(event) => selectHoleFromKeyboard(event, item.hole.id)}
              >
                <line
                  x1={item.toeScreen.x}
                  y1={item.toeScreen.y}
                  x2={item.screen.x}
                  y2={item.screen.y}
                  stroke={selected ? "#ffffff" : "#0f172a"}
                  strokeWidth={selected ? 2.2 : 1.2}
                  strokeOpacity={0.8}
                  vectorEffect="non-scaling-stroke"
                />
                {showWavefront && state === "active" ? (
                  <>
                    <circle
                      className="geomotion-wavefront"
                      cx={item.screen.x}
                      cy={item.screen.y}
                      r={17}
                      fill="none"
                      stroke="#fbbf24"
                      strokeWidth={3}
                      vectorEffect="non-scaling-stroke"
                    />
                    <circle
                      cx={item.screen.x}
                      cy={item.screen.y}
                      r={27}
                      fill="none"
                      stroke="#fde68a"
                      strokeWidth={1.5}
                      strokeOpacity={0.58}
                      vectorEffect="non-scaling-stroke"
                    />
                  </>
                ) : null}
                {item.hole.flyrockRisk === "high" || item.hole.flyrockRisk === "moderate" ? (
                  <circle
                    cx={item.screen.x}
                    cy={item.screen.y}
                    r={radius + 7}
                    fill="none"
                    stroke={item.hole.flyrockRisk === "high" ? "#fb7185" : "#fbbf24"}
                    strokeWidth={1.6}
                    strokeDasharray="3 3"
                    vectorEffect="non-scaling-stroke"
                  />
                ) : null}
                <circle
                  cx={item.screen.x}
                  cy={item.screen.y}
                  r={radius}
                  fill={fill}
                  fillOpacity={state === "unfired" ? 0.72 : 1}
                  stroke={selected ? "#ffffff" : state === "active" ? "#fef3c7" : "#0b1220"}
                  strokeWidth={selected || state === "active" ? 2.5 : 1.1}
                  vectorEffect="non-scaling-stroke"
                  filter={state === "active" ? "url(#geomotion-glow)" : undefined}
                />
                {showOrder && Number.isFinite(item.hole.firingOrder) ? (
                  <text x={item.screen.x} y={item.screen.y + 2.7} className="geomotion-order-label" textAnchor="middle">
                    {item.hole.firingOrder}
                  </text>
                ) : null}
                {showHoleLabel ? (
                  <text
                    x={overviewLabel ? item.screen.x - radius - 4 : item.screen.x + radius + 4}
                    y={item.screen.y - radius - 2 - labelRowOffset}
                    className="geomotion-hole-label"
                    textAnchor={overviewLabel ? "end" : undefined}
                  >
                    {holeLabel}
                  </text>
                ) : null}
                <title>{`${item.hole.id}
Source X ${item.hole.x.toFixed(2)} · Y ${item.hole.y.toFixed(2)} · Z ${item.hole.z?.toFixed(2) ?? "-"}
${ACTIVE_BENCH.label} / ${ACTIVE_BENCH.section}
Delay ${item.hole.delayMs ?? "-"} ms · Order ${item.hole.firingOrder ?? "-"}
Depth ${item.hole.depth ?? "-"} m · Charge ${item.hole.charge ?? "-"} kg`}</title>
              </g>
            );
          })}

          {BENCH_LABELS.map((label) => {
            const point = projection.point(label.point);
            return (
              <g key={label.id} pointerEvents="none">
                <line
                  x1={point.x - 4}
                  y1={point.y}
                  x2={point.x - 24}
                  y2={point.y}
                  stroke={label.active ? "#fbbf24" : "rgba(226,232,240,0.62)"}
                  strokeWidth={label.active ? 2 : 1}
                  vectorEffect="non-scaling-stroke"
                />
                <text
                  x={point.x - 29}
                  y={point.y + 4}
                  className={label.active ? "geomotion-bench-label geomotion-bench-label-active" : "geomotion-bench-label"}
                  textAnchor="end"
                >
                  {label.label}
                </text>
              </g>
            );
          })}

          <g pointerEvents="none">
            <line
              x1={activeLabelPoint.x}
              y1={activeLabelPoint.y}
              x2={activeLabelPoint.x + 70}
              y2={activeLabelPoint.y - 44}
              stroke="#fbbf24"
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
            <rect
              x={activeLabelPoint.x + 66}
              y={activeLabelPoint.y - 70}
              width={178}
              height={44}
              rx={8}
              fill="rgba(8,17,31,0.92)"
              stroke="#fbbf24"
            />
            <text x={activeLabelPoint.x + 78} y={activeLabelPoint.y - 52} className="geomotion-active-label">
              ACTIVE BLAST SECTION
            </text>
            <text x={activeLabelPoint.x + 78} y={activeLabelPoint.y - 37} className="geomotion-active-sublabel">
              {ACTIVE_BENCH.label} · {ACTIVE_BENCH.section}
            </text>
          </g>

          <g className="geomotion-scale" transform={`translate(54 ${SVG_HEIGHT - 46})`}>
            <line x1={0} y1={0} x2={scaleBarLength} y2={0} />
            <line x1={0} y1={-5} x2={0} y2={5} />
            <line x1={scaleBarLength} y1={-5} x2={scaleBarLength} y2={5} />
            <text x={scaleBarLength / 2} y={-9} textAnchor="middle">200 m</text>
          </g>
        </svg>

        <div className="geomotion-hud geomotion-hud-left">
          <div className="geomotion-hud-title">WHOLE-MINE BLOCK MODEL</div>
          <div className="geomotion-legend-grid">
            {(Object.keys(MATERIAL_COLOURS) as MineMaterial[]).map((material) => (
              <span key={material} className="geomotion-legend-item">
                <span className="geomotion-swatch" style={{ background: MATERIAL_COLOURS[material].top }} />
                {MATERIAL_COLOURS[material].label}
              </span>
            ))}
            <span className="geomotion-legend-item">
              <span className="geomotion-swatch geomotion-swatch-active" />
              Active bench
            </span>
            <span className="geomotion-legend-item">
              <span className="geomotion-line-swatch" />
              Delay tie-up
            </span>
          </div>
        </div>

        <div className="geomotion-hud geomotion-hud-right">
          <div className="geomotion-hud-title">FIRING PLAYBACK</div>
          <div className="geomotion-firing-time">{currentTime == null ? "Ready" : `${currentTime.toFixed(0)} ms`}</div>
          <div className="geomotion-hud-copy">
            {timedCount ? `${firedCount} fired · ${activeCount} current · ${Math.max(0, timedCount - firedCount - activeCount)} queued` : "Assign delays to the active tie-up"}
          </div>
          <div className="geomotion-planning-note">Planning playback only · no device connection</div>
        </div>

        <div className="geomotion-view-controls" aria-label="3D view controls">
          <button
            type="button"
            aria-label="Zoom in"
            title="Zoom in"
            onClick={() => setCamera((current) => ({ ...current, zoom: clamp(current.zoom * 1.15, 0.72, 2.6) }))}
          >
            +
          </button>
          <button
            type="button"
            aria-label="Zoom out"
            title="Zoom out"
            onClick={() => setCamera((current) => ({ ...current, zoom: clamp(current.zoom / 1.15, 0.72, 2.6) }))}
          >
            −
          </button>
          <button type="button" aria-label="Reset 3D view" title="Fit whole pit" onClick={() => setCamera(DEFAULT_CAMERA)}>
            Fit
          </button>
        </div>

        <div className="geomotion-compass" aria-hidden="true">
          <span>N</span>
          <i />
        </div>

        <div className="geomotion-colour-key">
          <span>{colourModeLabel(colorMode)}</span>
          <i />
          <small>{values.length ? `${valueMin.toFixed(0)} – ${valueMax.toFixed(0)}` : "No values"}</small>
        </div>

        {!holes.length ? (
          <div className="geomotion-empty-state">
            <strong>Whole mine context loaded</strong>
            <span>Import a hole CSV to define the active tie-up on {ACTIVE_BENCH.label}.</span>
          </div>
        ) : null}

        <div className="geomotion-orbit-hint">Drag to orbit · Wheel or +/− to zoom · Zoom in for all hole IDs</div>
        <div className="geomotion-sr-only" aria-live="polite">
          {fullscreen ? "Geomotion 3D fullscreen view active." : "Geomotion 3D embedded view active."}
        </div>
      </div>

      <div className="geomotion-footer">
        <span>
          <strong>{PIT_MODEL.name}</strong> · {PIT_MODEL.coordinateSystem}
        </span>
        <span>
          Pattern: {pattern} · Context is deterministic synthetic planning data; imported hole coordinates remain unchanged in exports.
        </span>
        {currentTime != null ? (
          <button type="button" className="geomotion-text-button" onClick={onResetSimulation}>
            Reset firing playback
          </button>
        ) : null}
      </div>
    </div>
  );
}
