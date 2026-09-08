export interface MinePoint3D {
  x: number;
  y: number;
  z: number;
}

export type MineMaterial = "waste" | "oxide" | "transition" | "ore";

export interface PitSurfaceCell {
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
  point: MinePoint3D;
  active?: boolean;
}

export const PIT_CONTEXT = {
  name: "GeoMotion North Pit",
  widthM: 1_120,
  lengthM: 860,
  verticalRangeM: 140,
  renderedCells: 600,
  benches: 7,
  coordinateSystem: "Local mine grid",
  seed: 66532,
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
const PIT_RADIUS_X = PIT_CONTEXT.widthM / 2;
const PIT_RADIUS_Y = PIT_CONTEXT.lengthM / 2;

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

function materialFor(benchIndex: number, segmentIndex: number) {
  const signal =
    0.48 +
    0.24 * Math.sin(segmentIndex * 0.61 + benchIndex * 0.83) +
    0.18 * Math.cos(segmentIndex * 0.27 - benchIndex * 0.74);
  const grade = Math.max(0.05, Math.min(1.45, 0.18 + signal * 0.92 + benchIndex * 0.035));
  if (grade > 1.02) return { material: "ore" as const, grade };
  if (grade > 0.72) return { material: "transition" as const, grade };
  if (benchIndex < 3 && grade > 0.42) return { material: "oxide" as const, grade };
  return { material: "waste" as const, grade };
}

function ringCell(
  id: string,
  kind: PitSurfaceCell["kind"],
  benchIndex: number,
  segmentIndex: number,
  outerRadius: number,
  innerRadius: number,
  outerElevation: number,
  innerElevation: number,
): PitSurfaceCell {
  const startAngle = (segmentIndex / SEGMENTS) * Math.PI * 2 - Math.PI;
  const endAngle = ((segmentIndex + 1) / SEGMENTS) * Math.PI * 2 - Math.PI;
  const active =
    kind === "bench" &&
    innerElevation === ACTIVE_BENCH.elevation &&
    endAngle >= ACTIVE_BENCH.startAngle &&
    startAngle <= ACTIVE_BENCH.endAngle;
  return {
    id,
    kind,
    benchIndex,
    ...materialFor(benchIndex, segmentIndex),
    active,
    points: [
      pointOnPit(outerRadius, startAngle, outerElevation),
      pointOnPit(outerRadius, endAngle, outerElevation),
      pointOnPit(innerRadius, endAngle, innerElevation),
      pointOnPit(innerRadius, startAngle, innerElevation),
    ],
  };
}

function buildPitSurfaceCells() {
  const cells: PitSurfaceCell[] = [];

  for (let segment = 0; segment < SEGMENTS; segment += 1) {
    cells.push(ringCell(`terrain-${segment}`, "terrain", 0, segment, 1.18, 1, CREST_ELEVATION, CREST_ELEVATION));
  }

  for (let transition = 0; transition < PIT_CONTEXT.benches; transition += 1) {
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
      if (transition < PIT_CONTEXT.benches - 1 && nextRadius > 0) {
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

  const floorElevation = CREST_ELEVATION - PIT_CONTEXT.benches * BENCH_DROP_M;
  const floorRadius = 1 - (PIT_CONTEXT.benches - 1) * BENCH_STEP - WALL_RUN;
  for (let segment = 0; segment < SEGMENTS; segment += 1) {
    const startAngle = (segment / SEGMENTS) * Math.PI * 2 - Math.PI;
    const endAngle = ((segment + 1) / SEGMENTS) * Math.PI * 2 - Math.PI;
    cells.push({
      id: `floor-${segment}`,
      kind: "floor",
      benchIndex: PIT_CONTEXT.benches,
      ...materialFor(PIT_CONTEXT.benches, segment),
      active: false,
      points: [
        { x: 0, y: 0, z: floorElevation },
        pointOnPit(floorRadius, startAngle, floorElevation),
        pointOnPit(floorRadius, endAngle, floorElevation),
      ],
    });
  }
  return cells;
}

export const PIT_SURFACE_CELLS = buildPitSurfaceCells();

export const BENCH_LABELS: BenchLabel[] = [
  { id: "crest", label: "Crest 760", point: pointOnPit(1.02, 2.62, 760) },
  { id: "b740", label: "740", point: pointOnPit(0.89, 2.62, 740) },
  { id: "b720", label: "720", point: pointOnPit(0.75, 2.62, 720) },
  { id: "b700", label: "700", point: pointOnPit(0.61, 2.62, 700) },
  {
    id: "b680",
    label: "680 · active",
    point: pointOnPit(0.475, ACTIVE_BENCH.endAngle + 0.035, ACTIVE_BENCH.elevation),
    active: true,
  },
  { id: "b660", label: "660", point: pointOnPit(0.33, 2.62, 660) },
  { id: "b640", label: "640", point: pointOnPit(0.19, 2.62, 640) },
  { id: "floor", label: "Floor 620", point: { x: 0, y: 0, z: 620 } },
];

if (PIT_SURFACE_CELLS.length !== PIT_CONTEXT.renderedCells) {
  throw new Error(`Expected ${PIT_CONTEXT.renderedCells} deterministic pit cells, received ${PIT_SURFACE_CELLS.length}.`);
}
