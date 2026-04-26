import type { BlastHole, TimingSettings, ValidationIssue } from "../types/blast";

function numericValues(holes: BlastHole[], selector: (hole: BlastHole) => number | undefined) {
  return holes.map(selector).filter((value): value is number => Number.isFinite(value));
}

function outlierIssues(holes: BlastHole[], field: "charge" | "depth", label: string) {
  const values = numericValues(holes, (hole) => hole[field]);
  if (values.length < 4) return [] as ValidationIssue[];
  const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length;
  const std = Math.sqrt(variance);
  if (!Number.isFinite(std) || std === 0) return [];
  return holes
    .filter((hole) => Number.isFinite(hole[field]) && Math.abs((hole[field] as number) - avg) > 2.5 * std)
    .map((hole) => ({
      severity: "warning" as const,
      message: `${label} for ${hole.id} is far from the project average.`,
      holeId: hole.id,
      field,
      suggestion: "Review the imported value before relying on analysis summaries.",
    }));
}

export function validateBlastDesign(holes: BlastHole[], settings: TimingSettings, requireDelays = false): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  if (!holes.length) {
    issues.push({ severity: "error", message: "No blast-hole data has been imported.", suggestion: "Import a CSV file first." });
    return issues;
  }

  const ids = new Map<string, number>();
  holes.forEach((hole) => {
    ids.set(hole.id.toLowerCase(), (ids.get(hole.id.toLowerCase()) ?? 0) + 1);
    if (!Number.isFinite(hole.x) || !Number.isFinite(hole.y)) {
      issues.push({ severity: "error", message: `${hole.id} has missing or invalid X/Y coordinates.`, holeId: hole.id });
    }
    if (!Number.isFinite(hole.depth)) {
      issues.push({ severity: "warning", message: `${hole.id} has no depth value.`, holeId: hole.id, field: "depth" });
    }
    if (!Number.isFinite(hole.charge)) {
      issues.push({ severity: "warning", message: `${hole.id} has no charge value.`, holeId: hole.id, field: "charge" });
    }
    if (requireDelays && !Number.isFinite(hole.delayMs)) {
      issues.push({ severity: "error", message: `${hole.id} has no assigned delay.`, holeId: hole.id, field: "delayMs" });
    }
    if (Number.isFinite(hole.delayMs)) {
      if ((hole.delayMs as number) < 0) {
        issues.push({ severity: "error", message: `${hole.id} has a negative delay.`, holeId: hole.id, field: "delayMs" });
      }
      if ((hole.delayMs as number) < settings.minDelayMs || (hole.delayMs as number) > settings.maxDelayMs) {
        issues.push({
          severity: "error",
          message: `${hole.id} delay is outside the configured allowed range.`,
          holeId: hole.id,
          field: "delayMs",
          suggestion: "Adjust timing settings or manually edit the delay.",
        });
      }
    }
  });

  ids.forEach((count, id) => {
    if (count > 1) {
      issues.push({ severity: "warning", message: `Duplicate Hole ID detected: ${id}.`, field: "id" });
    }
  });

  const delays = holes.map((hole) => hole.delayMs).filter((delay): delay is number => Number.isFinite(delay));
  const delayCounts = new Map<number, number>();
  delays.forEach((delay) => delayCounts.set(delay, (delayCounts.get(delay) ?? 0) + 1));
  if ([...delayCounts.values()].some((count) => count > 1)) {
    issues.push({
      severity: "warning",
      message: "Some holes share the same delay time.",
      field: "delayMs",
      suggestion: "Review delay groups and timing concentration before export.",
    });
  }

  const xs = numericValues(holes, (hole) => hole.x);
  const ys = numericValues(holes, (hole) => hole.y);
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  if (Number.isFinite(span) && span > 10000) {
    issues.push({
      severity: "warning",
      message: "The coordinate range is very large.",
      suggestion: "Confirm the coordinate system and units are correct.",
    });
  }

  issues.push(...outlierIssues(holes, "charge", "Charge"));
  issues.push(...outlierIssues(holes, "depth", "Depth"));
  return issues;
}

export function designCompleteness(holes: BlastHole[]) {
  if (!holes.length) return 0;
  const assigned = holes.filter((hole) => Number.isFinite(hole.delayMs)).length;
  return Math.round((assigned / holes.length) * 100);
}
