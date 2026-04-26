import type { BlastHole, TimingLine, TimingPattern, TimingPoint, TimingSettings } from "../types/blast";
import { detectRows } from "./rowDetection";

export interface TimingOptions {
  pattern: TimingPattern;
  settings: TimingSettings;
  selectedIds?: string[];
  initiationPoint?: TimingPoint;
  initiationLine?: TimingLine;
  vWidth?: "narrow" | "medium" | "wide";
}

function roundDelay(delay: number, settings: TimingSettings) {
  if (!settings.applyRounding) return delay;
  const increment = Math.max(1, settings.delayIncrementMs);
  return Math.round(delay / increment) * increment;
}

function withSequentialDelays(ordered: BlastHole[], settings: TimingSettings, group: string) {
  const sequence = settings.reverseOrder ? [...ordered].reverse() : ordered;
  return sequence.map((hole, index) => ({
    ...hole,
    firingOrder: index + 1,
    timingGroup: group,
    delayMs: roundDelay(settings.startDelayMs + index * settings.inRowDelayMs, settings),
  }));
}

function mergeTimed(original: BlastHole[], timed: BlastHole[]) {
  const byId = new Map(timed.map((hole) => [hole.id, hole]));
  return original.map((hole) => byId.get(hole.id) ?? hole);
}

function distance(a: TimingPoint, b: TimingPoint) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function projectAlongLine(point: TimingPoint, line: TimingLine) {
  const dx = line.end.x - line.start.x;
  const dy = line.end.y - line.start.y;
  const len2 = dx * dx + dy * dy || 1;
  return ((point.x - line.start.x) * dx + (point.y - line.start.y) * dy) / len2;
}

function perpendicularDistance(point: TimingPoint, line: TimingLine) {
  const dx = line.end.x - line.start.x;
  const dy = line.end.y - line.start.y;
  const len = Math.hypot(dx, dy) || 1;
  return Math.abs(dy * point.x - dx * point.y + line.end.x * line.start.y - line.end.y * line.start.x) / len;
}

function estimateSpacing(holes: BlastHole[]) {
  if (holes.length < 2) return 1;
  const nearest = holes.map((hole, idx) => {
    let best = Number.POSITIVE_INFINITY;
    holes.forEach((candidate, jdx) => {
      if (idx === jdx) return;
      const d = distance(hole, candidate);
      if (d > 0) best = Math.min(best, d);
    });
    return best;
  }).filter(Number.isFinite);
  if (!nearest.length) return 1;
  nearest.sort((a, b) => a - b);
  return nearest[Math.floor(nearest.length / 2)] || 1;
}

function centerOfHoles(holes: BlastHole[]): TimingPoint {
  return {
    x: holes.reduce((sum, hole) => sum + hole.x, 0) / Math.max(1, holes.length),
    y: holes.reduce((sum, hole) => sum + hole.y, 0) / Math.max(1, holes.length),
  };
}

function assignByRows(holes: BlastHole[], settings: TimingSettings) {
  const { holes: indexed } = detectRows(holes, settings.rowTolerance, settings.direction);
  const timed = indexed.map((hole) => {
    const rowIndex = hole.rowIndex ?? 0;
    const columnIndex = hole.columnIndex ?? 0;
    const order = rowIndex * 10000 + columnIndex + 1;
    return {
      ...hole,
      firingOrder: order,
      timingGroup: `Row ${rowIndex + 1}`,
      delayMs: roundDelay(settings.startDelayMs + rowIndex * settings.rowDelayMs + columnIndex * settings.inRowDelayMs, settings),
    };
  });
  const ordered = [...timed].sort((a, b) => (a.delayMs ?? 0) - (b.delayMs ?? 0) || (a.firingOrder ?? 0) - (b.firingOrder ?? 0));
  return ordered.map((hole, idx) => ({ ...hole, firingOrder: settings.reverseOrder ? ordered.length - idx : idx + 1 }));
}

function assignChevron(holes: BlastHole[], settings: TimingSettings, point?: TimingPoint) {
  const center = point ?? centerOfHoles(holes);
  const { holes: indexed } = detectRows(holes, settings.rowTolerance, settings.direction);
  return indexed.map((hole) => {
    const rowIndex = hole.rowIndex ?? 0;
    const rowMates = indexed.filter((candidate) => candidate.rowIndex === rowIndex);
    const lateralOrder = [...rowMates].sort((a, b) => Math.abs(a.x - center.x) - Math.abs(b.x - center.x)).findIndex((candidate) => candidate.id === hole.id);
    return {
      ...hole,
      firingOrder: rowIndex * 1000 + lateralOrder + 1,
      timingGroup: `Chevron row ${rowIndex + 1}`,
      delayMs: roundDelay(settings.startDelayMs + rowIndex * settings.rowDelayMs + lateralOrder * settings.inRowDelayMs, settings),
    };
  }).sort((a, b) => (a.delayMs ?? 0) - (b.delayMs ?? 0)).map((hole, idx, arr) => ({
    ...hole,
    firingOrder: settings.reverseOrder ? arr.length - idx : idx + 1,
  }));
}

function assignVcut(holes: BlastHole[], settings: TimingSettings, point?: TimingPoint, width: TimingOptions["vWidth"] = "medium") {
  const apex = point ?? centerOfHoles(holes);
  const widthFactor = width === "narrow" ? 0.75 : width === "wide" ? 1.35 : 1;
  const ordered = [...holes].sort((a, b) => {
    const da = distance(a, apex) + Math.abs(a.y - apex.y) * (1 / widthFactor);
    const db = distance(b, apex) + Math.abs(b.y - apex.y) * (1 / widthFactor);
    return da - db || a.x - b.x;
  });
  return withSequentialDelays(ordered, settings, `V-cut ${width}`);
}

function assignBoxCut(holes: BlastHole[], settings: TimingSettings, point?: TimingPoint) {
  const center = point ?? centerOfHoles(holes);
  const sortedDistances = holes.map((hole) => distance(hole, center)).sort((a, b) => a - b);
  const ringSize = Math.max(1, Math.ceil(holes.length / 8));
  const ringBreaks = sortedDistances.filter((_, idx) => idx % ringSize === 0);
  const withRings = holes.map((hole) => {
    const d = distance(hole, center);
    const ringIndex = Math.max(0, ringBreaks.findIndex((ring, idx) => d >= ring && d < (ringBreaks[idx + 1] ?? Number.POSITIVE_INFINITY)));
    return { ...hole, rowIndex: ringIndex };
  });
  const timed = withRings.map((hole) => {
    const ringIndex = hole.rowIndex ?? 0;
    const ringMates = withRings.filter((candidate) => candidate.rowIndex === ringIndex).sort((a, b) => Math.atan2(a.y - center.y, a.x - center.x) - Math.atan2(b.y - center.y, b.x - center.x));
    const withinRingOrder = ringMates.findIndex((candidate) => candidate.id === hole.id);
    return {
      ...hole,
      columnIndex: withinRingOrder,
      timingGroup: `Ring ${ringIndex + 1}`,
      delayMs: roundDelay(settings.startDelayMs + ringIndex * settings.rowDelayMs + withinRingOrder * settings.inRowDelayMs, settings),
    };
  });
  return timed.sort((a, b) => (a.delayMs ?? 0) - (b.delayMs ?? 0)).map((hole, idx, arr) => ({ ...hole, firingOrder: settings.reverseOrder ? arr.length - idx : idx + 1 }));
}

function assignFromPoint(holes: BlastHole[], settings: TimingSettings, point?: TimingPoint) {
  const origin = point ?? centerOfHoles(holes);
  const ordered = [...holes].sort((a, b) => distance(a, origin) - distance(b, origin) || a.x - b.x);
  return withSequentialDelays(ordered, settings, "Point directional");
}

function assignFromLine(holes: BlastHole[], settings: TimingSettings, line?: TimingLine) {
  const xs = holes.map((hole) => hole.x);
  const ys = holes.map((hole) => hole.y);
  const defaultLine =
    settings.direction === "bottomToTop" || settings.direction === "topToBottom"
      ? {
          start: { x: Math.min(...xs), y: Math.min(...ys) },
          end: { x: Math.min(...xs), y: Math.max(...ys) },
        }
      : {
          start: { x: Math.min(...xs), y: Math.min(...ys) },
          end: { x: Math.max(...xs), y: Math.min(...ys) },
        };
  const activeLine = line ?? defaultLine;
  const bandSize = Math.max(settings.rowTolerance, estimateSpacing(holes) * 0.6, 0.01);
  const withBands = holes.map((hole) => ({
    hole,
    bandIndex: Math.floor(perpendicularDistance(hole, activeLine) / bandSize),
    along: projectAlongLine(hole, activeLine),
  }));
  const bands = Array.from(new Set(withBands.map((item) => item.bandIndex))).sort((a, b) => a - b);
  const timed: BlastHole[] = [];
  bands.forEach((bandIndex) => {
    const band = withBands
      .filter((item) => item.bandIndex === bandIndex)
      .sort((a, b) => a.along - b.along || a.hole.x - b.hole.x || a.hole.y - b.hole.y);
    band.forEach((item, withinBandOrder) => {
      timed.push({
        ...item.hole,
        rowIndex: bandIndex,
        columnIndex: withinBandOrder,
        timingGroup: `Line band ${bandIndex + 1}`,
        delayMs: roundDelay(settings.startDelayMs + bandIndex * settings.rowDelayMs + withinBandOrder * settings.inRowDelayMs, settings),
      });
    });
  });
  const ordered = timed.sort((a, b) => (a.delayMs ?? 0) - (b.delayMs ?? 0) || (a.columnIndex ?? 0) - (b.columnIndex ?? 0));
  return ordered.map((hole, index) => ({
    ...hole,
    firingOrder: settings.reverseOrder ? ordered.length - index : index + 1,
  }));
}

export function assignTiming(holes: BlastHole[], options: TimingOptions): BlastHole[] {
  const base = holes.map((hole) => ({ ...hole }));
  if (!base.length) return base;
  let timed: BlastHole[];
  switch (options.pattern) {
    case "manual": {
      const selected = options.selectedIds?.length ? base.filter((hole) => options.selectedIds?.includes(hole.id)) : base;
      timed = withSequentialDelays(selected, options.settings, "Manual sequence");
      return mergeTimed(base, timed);
    }
    case "chevron":
      return assignChevron(base, options.settings, options.initiationPoint);
    case "vCut":
      return assignVcut(base, options.settings, options.initiationPoint, options.vWidth);
    case "boxCut":
      return assignBoxCut(base, options.settings, options.initiationPoint);
    case "pointDirectional":
      return assignFromPoint(base, options.settings, options.initiationPoint);
    case "lineDirectional":
      return assignFromLine(base, options.settings, options.initiationLine);
    case "rowByRow":
    default:
      return assignByRows(base, options.settings);
  }
}
