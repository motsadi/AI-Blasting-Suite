import type { BlastHole } from "../../types/blast";

export interface MinePoint3D {
  x: number;
  y: number;
  z: number;
}

export type MineMaterial = "waste" | "oxide" | "transition" | "ore";

export interface MineCell {
  id: string;
  kind: "terrain" | "wall" | "bench" | "floor";
  benchIndex: number;
  material: MineMaterial;
  grade: number;
  active: boolean;
  points: MinePoint3D[];
}

export interface BenchLabel {
  id: string;
  label: string;
  elevation: number;
  point: MinePoint3D;
  active?: boolean;
}

export interface PositionedBlastHole {
  hole: BlastHole;
  collar: MinePoint3D;
  toe: MinePoint3D;
}

export const PIT_MODEL = {
  name: "Geomotion North Pit",
  widthM: 1_120,
  lengthM: 860,
  verticalRangeM: 140,
  blockCount: 600,
  coordinateSystem: "Local mine grid / synthetic",
} as const;

export const ACTIVE_BENCH = {
  label: "Bench 680",
  section: "East 04",
  elevation: 680,
  outerRadius: 0.505,
  innerRadius: 0.44,
  startAngle: -0.3,
  endAngle: 0.3,
} as const;

const SEGMENTS = 40;
const BENCH_DROP_M = 20;
const CREST_ELEVATION = 760;
const BENCH_STEP = 0.14;
const WALL_RUN = 0.075;
const PIT_RADIUS_X = PIT_MODEL.widthM / 2;
const PIT_RADIUS_Y = PIT_MODEL.lengthM / 2;

function angleInActiveSection(angle: number) {
  return angle >= ACTIVE_BENCH.startAngle && angle <= ACTIVE_BENCH.endAngle;
}

function radiusVariation(angle: number, radius: number) {
  return 1 + 0.045 * Math.sin(angle * 3 + 0.35) + 0.018 * Math.cos(angle * 5 - radius * 1.7);
}

export function pointOnPit(radius: number, angle: number, z: number): MinePoint3D {
  const variation = radiusVariation(angle, radius);
  return {
    x: PIT_RADIUS_X * radius * variation * Math.cos(angle),
    y: PIT_RADIUS_Y * radius * (1 + 0.022 * Math.sin(angle * 4 - 0.5)) * Math.sin(angle),
    z,
  };
}

function materialFor(benchIndex: number, segmentIndex: number): { material: MineMaterial; grade: number } {
  const signal =
    0.48 +
    0.24 * Math.sin(segmentIndex * 0.61 + benchIndex * 0.83) +
    0.18 * Math.cos(segmentIndex * 0.27 - benchIndex * 0.74);
  const grade = Math.max(0.05, Math.min(1.45, 0.18 + signal * 0.92 + benchIndex * 0.035));
  if (grade > 1.02) return { material: "ore", grade };
  if (grade > 0.72) return { material: "transition", grade };
  if (benchIndex < 3 && grade > 0.42) return { material: "oxide", grade };
  return { material: "waste", grade };
}

function ringCell(
  id: string,
  kind: MineCell["kind"],
  benchIndex: number,
  segmentIndex: number,
  outerRadius: number,
  innerRadius: number,
  outerElevation: number,
  innerElevation: number,
) {
  const a0 = (segmentIndex / SEGMENTS) * Math.PI * 2 - Math.PI;
  const a1 = ((segmentIndex + 1) / SEGMENTS) * Math.PI * 2 - Math.PI;
  const composition = materialFor(benchIndex, segmentIndex);
  const active =
    kind === "bench" &&
    innerElevation === ACTIVE_BENCH.elevation &&
    (angleInActiveSection(a0) || angleInActiveSection(a1) || (a0 < ACTIVE_BENCH.startAngle && a1 > ACTIVE_BENCH.endAngle));
  return {
    id,
    kind,
    benchIndex,
    ...composition,
    active,
    points: [
      pointOnPit(outerRadius, a0, outerElevation),
      pointOnPit(outerRadius, a1, outerElevation),
      pointOnPit(innerRadius, a1, innerElevation),
      pointOnPit(innerRadius, a0, innerElevation),
    ],
  } satisfies MineCell;
}

function buildWholeMineCells() {
  const cells: MineCell[] = [];

  for (let segment = 0; segment < SEGMENTS; segment += 1) {
    cells.push(ringCell(`terrain-${segment}`, "terrain", 0, segment, 1.18, 1, CREST_ELEVATION, CREST_ELEVATION));
  }

  for (let transition = 0; transition < 7; transition += 1) {
    const topRadius = 1 - transition * BENCH_STEP;
    const toeRadius = Math.max(0.02, topRadius - WALL_RUN);
    const nextRadius = Math.max(0, topRadius - BENCH_STEP);
    const topElevation = CREST_ELEVATION - transition * BENCH_DROP_M;
    const floorElevation = topElevation - BENCH_DROP_M;

    for (let segment = 0; segment < SEGMENTS; segment += 1) {
      cells.push(
        ringCell(
          `wall-${transition}-${segment}`,
          "wall",
          transition + 1,
          segment,
          topRadius,
          toeRadius,
          topElevation,
          floorElevation,
        ),
      );
      if (transition < 6 && nextRadius > 0) {
        cells.push(
          ringCell(
            `bench-${transition}-${segment}`,
            "bench",
            transition + 1,
            segment,
            toeRadius,
            nextRadius,
            floorElevation,
            floorElevation,
          ),
        );
      }
    }
  }

  const floorElevation = CREST_ELEVATION - 7 * BENCH_DROP_M;
  const floorRadius = 1 - 6 * BENCH_STEP - WALL_RUN;
  for (let segment = 0; segment < SEGMENTS; segment += 1) {
    const a0 = (segment / SEGMENTS) * Math.PI * 2 - Math.PI;
    const a1 = ((segment + 1) / SEGMENTS) * Math.PI * 2 - Math.PI;
    const composition = materialFor(7, segment);
    cells.push({
      id: `floor-${segment}`,
      kind: "floor",
      benchIndex: 7,
      ...composition,
      active: false,
      points: [
        { x: 0, y: 0, z: floorElevation },
        pointOnPit(floorRadius, a0, floorElevation),
        pointOnPit(floorRadius, a1, floorElevation),
      ],
    });
  }

  return cells;
}

export const WHOLE_MINE_CELLS = buildWholeMineCells();

export const BENCH_LABELS: BenchLabel[] = [
  { id: "crest", label: "Crest 760", elevation: 760, point: pointOnPit(1.02, 2.62, 760) },
  { id: "b740", label: "Bench 740", elevation: 740, point: pointOnPit(0.89, 2.62, 740) },
  { id: "b720", label: "Bench 720", elevation: 720, point: pointOnPit(0.75, 2.62, 720) },
  { id: "b700", label: "Bench 700", elevation: 700, point: pointOnPit(0.61, 2.62, 700) },
  {
    id: "b680",
    label: `${ACTIVE_BENCH.label} · active`,
    elevation: 680,
    point: pointOnPit(0.475, ACTIVE_BENCH.endAngle + 0.035, ACTIVE_BENCH.elevation),
    active: true,
  },
  { id: "b660", label: "Bench 660", elevation: 660, point: pointOnPit(0.33, 2.62, 660) },
  { id: "b640", label: "Bench 640", elevation: 640, point: pointOnPit(0.19, 2.62, 640) },
  { id: "floor", label: "Pit floor 620", elevation: 620, point: { x: 0, y: 0, z: 620 } },
];

export const MINE_EXTENT_POINTS = [
  ...Array.from({ length: SEGMENTS }, (_, index) => pointOnPit(1.2, (index / SEGMENTS) * Math.PI * 2 - Math.PI, CREST_ELEVATION)),
  { x: 0, y: 0, z: 610 },
  { x: 0, y: 0, z: 780 },
];

export function positionHolesOnActiveBench(holes: BlastHole[]): PositionedBlastHole[] {
  const valid = holes.filter((hole) => Number.isFinite(hole.x) && Number.isFinite(hole.y));
  if (!valid.length) return [];
  const xs = valid.map((hole) => hole.x);
  const ys = valid.map((hole) => hole.y);
  const zs = valid.map((hole) => hole.z).filter((value): value is number => Number.isFinite(value));
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const zmean = zs.length ? zs.reduce((sum, value) => sum + value, 0) / zs.length : ACTIVE_BENCH.elevation;
  const angleInset = 0.035;
  const radiusInset = 0.008;

  return valid.map((hole, index) => {
    const u = xmax === xmin ? (index + 0.5) / valid.length : (hole.x - xmin) / (xmax - xmin);
    const v = ymax === ymin ? 0.5 : (hole.y - ymin) / (ymax - ymin);
    const angle =
      ACTIVE_BENCH.startAngle +
      angleInset +
      u * (ACTIVE_BENCH.endAngle - ACTIVE_BENCH.startAngle - angleInset * 2);
    const radius = ACTIVE_BENCH.innerRadius + radiusInset + v * (ACTIVE_BENCH.outerRadius - ACTIVE_BENCH.innerRadius - radiusInset * 2);
    const collarElevation =
      ACTIVE_BENCH.elevation +
      (Number.isFinite(hole.z) ? Math.max(-2.5, Math.min(2.5, (hole.z as number) - zmean)) : 0) +
      1.2;
    const collar = pointOnPit(radius, angle, collarElevation);
    return {
      hole,
      collar,
      toe: {
        ...collar,
        z: collar.z - Math.max(6, Math.min(18, Number.isFinite(hole.depth) ? (hole.depth as number) : 12)),
      },
    };
  });
}

export function createDemoBlastHoles(): BlastHole[] {
  const rows = 6;
  const columns = 10;
  const holes: BlastHole[] = [];
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const index = row * columns + column;
      const depth = 13.6 + 0.48 * Math.sin(index * 0.73) + 0.2 * Math.cos(row * 1.4);
      const charge = 548 + 22 * Math.sin(index * 0.39 + 0.4) + row * 3.5;
      holes.push({
        id: `B680-R${String(row + 1).padStart(2, "0")}-H${String(column + 1).padStart(2, "0")}`,
        x: 501_240 + column * 9.2 + (row % 2) * 4.6,
        y: 7_286_410 + row * 5.4,
        z: 680 + 0.35 * Math.sin(column * 0.62) - row * 0.08,
        depth: Number(depth.toFixed(2)),
        charge: Number(charge.toFixed(1)),
        rowIndex: row,
        columnIndex: column,
        delayMs: row * 42 + column * 17,
        timingGroup: `Row ${row + 1}`,
      });
    }
  }
  const firingOrder = new Map(
    [...holes]
      .sort((a, b) => (a.delayMs ?? 0) - (b.delayMs ?? 0) || (a.rowIndex ?? 0) - (b.rowIndex ?? 0) || (a.columnIndex ?? 0) - (b.columnIndex ?? 0))
      .map((hole, index) => [hole.id, index + 1]),
  );
  return holes.map((hole) => ({ ...hole, firingOrder: firingOrder.get(hole.id) }));
}
