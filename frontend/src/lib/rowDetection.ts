import type { BlastHole, TimingDirection } from "../types/blast";

export interface RowDetectionResult {
  holes: BlastHole[];
  rowCount: number;
}

function axisValue(hole: BlastHole, direction: TimingDirection) {
  return direction === "bottomToTop" || direction === "topToBottom" ? hole.x : hole.y;
}

function columnValue(hole: BlastHole, direction: TimingDirection) {
  return direction === "bottomToTop" || direction === "topToBottom" ? hole.y : hole.x;
}

function sortRows(rows: BlastHole[][], direction: TimingDirection) {
  const ordered = [...rows].sort((a, b) => {
    const av = a.reduce((sum, hole) => sum + axisValue(hole, direction), 0) / Math.max(1, a.length);
    const bv = b.reduce((sum, hole) => sum + axisValue(hole, direction), 0) / Math.max(1, b.length);
    return av - bv;
  });
  if (direction === "topToBottom" || direction === "rightToLeft") ordered.reverse();
  return ordered;
}

export function detectRows(holes: BlastHole[], tolerance: number, direction: TimingDirection): RowDetectionResult {
  if (!holes.length) return { holes: [], rowCount: 0 };
  const tol = Math.max(0.01, tolerance);
  const sorted = [...holes].sort((a, b) => axisValue(a, direction) - axisValue(b, direction));
  const rows: BlastHole[][] = [];

  sorted.forEach((hole) => {
    const target = axisValue(hole, direction);
    const row = rows.find((candidate) => {
      const center = candidate.reduce((sum, item) => sum + axisValue(item, direction), 0) / candidate.length;
      return Math.abs(center - target) <= tol;
    });
    if (row) {
      row.push(hole);
    } else {
      rows.push([hole]);
    }
  });

  const orderedRows = sortRows(rows, direction);
  const reverseColumns = direction === "rightToLeft" || direction === "topToBottom";
  const result = orderedRows.flatMap((row, rowIndex) =>
    [...row]
      .sort((a, b) => {
        const diff = columnValue(a, direction) - columnValue(b, direction);
        return reverseColumns ? -diff : diff;
      })
      .map((hole, columnIndex) => ({ ...hole, rowIndex, columnIndex }))
  );

  return { holes: result, rowCount: orderedRows.length };
}

export function defaultRowTolerance(holes: BlastHole[]) {
  if (holes.length < 2) return 1;
  const ys = [...new Set(holes.map((hole) => hole.y).filter(Number.isFinite))].sort((a, b) => a - b);
  const diffs = ys.slice(1).map((value, idx) => Math.abs(value - ys[idx])).filter((value) => value > 0);
  if (!diffs.length) return 1;
  const median = diffs.sort((a, b) => a - b)[Math.floor(diffs.length / 2)];
  return Math.max(0.5, median * 0.45);
}
