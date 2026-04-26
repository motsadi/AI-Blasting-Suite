import type { BlastHole } from "../types/blast";

export function uniqueDelayTimes(holes: BlastHole[]) {
  return Array.from(new Set(holes.map((hole) => hole.delayMs).filter((delay): delay is number => Number.isFinite(delay)))).sort((a, b) => a - b);
}

export function simulationState(holes: BlastHole[], timeMs: number | undefined) {
  if (timeMs == null) {
    return { fired: [] as BlastHole[], active: [] as BlastHole[], unfired: holes };
  }
  const eps = 1e-6;
  return {
    fired: holes.filter((hole) => Number.isFinite(hole.delayMs) && (hole.delayMs as number) < timeMs - eps),
    active: holes.filter((hole) => Number.isFinite(hole.delayMs) && Math.abs((hole.delayMs as number) - timeMs) <= eps),
    unfired: holes.filter((hole) => !Number.isFinite(hole.delayMs) || (hole.delayMs as number) > timeMs + eps),
  };
}
