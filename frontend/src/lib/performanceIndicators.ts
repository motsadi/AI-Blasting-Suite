import type { BlastHole } from "../types/blast";

export interface PerformanceSummary {
  averageFragmentationMm?: number;
  maxPpvMmS?: number;
  maxAirblastDb?: number;
  highFlyrockRiskCount: number;
  moderateFlyrockRiskCount: number;
  maxChargeAtSameDelay: number;
  warnings: string[];
}

function finiteValues(values: Array<number | undefined>) {
  return values.filter((value): value is number => Number.isFinite(value));
}

function nearestSpacing(holes: BlastHole[], hole: BlastHole) {
  let best = Number.POSITIVE_INFINITY;
  holes.forEach((candidate) => {
    if (candidate.id === hole.id) return;
    const d = Math.hypot(candidate.x - hole.x, candidate.y - hole.y);
    if (d > 0) best = Math.min(best, d);
  });
  return Number.isFinite(best) ? best : 5;
}

function median(values: number[], fallback: number) {
  if (!values.length) return fallback;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? fallback;
}

function delayGroupCharge(holes: BlastHole[], delay: number | undefined) {
  if (!Number.isFinite(delay)) return 0;
  return holes
    .filter((hole) => Number.isFinite(hole.delayMs) && Math.abs((hole.delayMs as number) - (delay as number)) < 1e-6)
    .reduce((sum, hole) => sum + (Number.isFinite(hole.charge) ? (hole.charge as number) : 0), 0);
}

export function estimatePerformance(holes: BlastHole[]) {
  const charges = finiteValues(holes.map((hole) => hole.charge));
  const depths = finiteValues(holes.map((hole) => hole.depth));
  const medianCharge = median(charges, 100);
  const medianDepth = median(depths, 12);
  const projectCenter = {
    x: holes.reduce((sum, hole) => sum + hole.x, 0) / Math.max(1, holes.length),
    y: holes.reduce((sum, hole) => sum + hole.y, 0) / Math.max(1, holes.length),
  };

  const enriched = holes.map((hole) => {
    const charge = Math.max(Number.isFinite(hole.charge) ? (hole.charge as number) : medianCharge, 0.001);
    const depth = Math.max(Number.isFinite(hole.depth) ? (hole.depth as number) : medianDepth, 0.5);
    const spacing = Math.max(nearestSpacing(holes, hole), 0.5);
    const burden = Math.max(spacing * 0.9, 0.5);
    const volume = Math.max(depth * spacing * burden, 0.001);
    const powderFactor = charge / volume;
    const chargePerM = Math.max(charge / depth, 0.001);
    const scaledBurden = burden / Math.sqrt(chargePerM);
    const chargeAtDelay = delayGroupCharge(holes, hole.delayMs) || charge;
    const distanceFromCenter = Math.max(80, Math.hypot(hole.x - projectCenter.x, hole.y - projectCenter.y) + 120);
    const scaledDistance = distanceFromCenter / Math.sqrt(Math.max(chargeAtDelay, 0.001));

    // Planning estimate inspired by common scaled-distance PPV practice. Site constants must be calibrated.
    const ppv = 1143 * Math.pow(Math.max(scaledDistance, 0.1), -1.65);
    // Conceptual airblast trend: lower scaled distance and larger charge per delay increase overpressure.
    const airblast = Math.max(85, Math.min(150, 165 - 24 * Math.log10(Math.max(scaledDistance, 1))));
    // Simplified fragmentation trend: higher powder factor and confinement reduce estimated X50.
    const fragmentation = Math.max(20, Math.min(900, 120 * Math.pow(Math.max(powderFactor, 0.001), -0.38) * Math.pow(charge, 1 / 6) / Math.pow(Math.max(spacing / burden, 0.2), 0.15)));
    const confinementRisk = Math.max(0, (0.78 - scaledBurden) / 0.78);
    const chargeRisk = charge > medianCharge * 1.25 ? 0.35 : 0;
    const delayRisk = chargeAtDelay > medianCharge * 2 ? 0.25 : 0;
    const flyrockDistance = Math.max(20, Math.min(700, 95 + 130 * confinementRisk + 0.035 * charge + 25 * chargeRisk + 30 * delayRisk));
    const flyrockScore = confinementRisk + chargeRisk + delayRisk;
    const flyrockRisk = flyrockScore >= 0.65 ? "high" : flyrockScore >= 0.3 ? "moderate" : "low";
    const performanceWarnings = [
      flyrockRisk === "high" ? "Elevated flyrock screening risk from scaled burden/charge concentration." : "",
      ppv > 12 ? "Higher relative PPV estimate. Review charge per delay and sensitive locations." : "",
      airblast > 125 ? "Higher relative airblast estimate. Review confinement, stemming and charge concentration." : "",
    ].filter(Boolean);

    return {
      ...hole,
      estimatedFragmentationMm: fragmentation,
      estimatedPpvMmS: ppv,
      estimatedAirblastDb: airblast,
      estimatedFlyrockDistanceM: flyrockDistance,
      flyrockRisk,
      performanceWarnings,
    };
  });

  return enriched;
}

export function summarizePerformance(holes: BlastHole[]): PerformanceSummary {
  const frag = finiteValues(holes.map((hole) => hole.estimatedFragmentationMm));
  const ppv = finiteValues(holes.map((hole) => hole.estimatedPpvMmS));
  const air = finiteValues(holes.map((hole) => hole.estimatedAirblastDb));
  const delayCharges = new Map<number, number>();
  holes.forEach((hole) => {
    if (!Number.isFinite(hole.delayMs)) return;
    delayCharges.set(hole.delayMs as number, (delayCharges.get(hole.delayMs as number) ?? 0) + (Number.isFinite(hole.charge) ? (hole.charge as number) : 0));
  });
  const highFlyrockRiskCount = holes.filter((hole) => hole.flyrockRisk === "high").length;
  const moderateFlyrockRiskCount = holes.filter((hole) => hole.flyrockRisk === "moderate").length;
  const warnings = [
    highFlyrockRiskCount ? `${highFlyrockRiskCount} hole(s) screen as elevated flyrock risk.` : "",
    ppv.length && Math.max(...ppv) > 12 ? "Relative PPV estimate is elevated. Review charge per delay and timing concentration." : "",
    air.length && Math.max(...air) > 125 ? "Relative airblast estimate is elevated. Review confinement, stemming and charge concentration." : "",
  ].filter(Boolean);
  return {
    averageFragmentationMm: frag.length ? frag.reduce((sum, value) => sum + value, 0) / frag.length : undefined,
    maxPpvMmS: ppv.length ? Math.max(...ppv) : undefined,
    maxAirblastDb: air.length ? Math.max(...air) : undefined,
    highFlyrockRiskCount,
    moderateFlyrockRiskCount,
    maxChargeAtSameDelay: Math.max(0, ...delayCharges.values()),
    warnings,
  };
}
