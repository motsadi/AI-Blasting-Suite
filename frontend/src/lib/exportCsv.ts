import type { BlastHole, TimingPattern, ValidationIssue } from "../types/blast";
import { TIMING_PATTERN_LABELS } from "../types/blast";

const COLUMNS = [
  "Hole ID",
  "Original Hole ID",
  "X",
  "Y",
  "Z",
  "Depth",
  "Charge",
  "Row Index",
  "Column Index",
  "Timing Pattern",
  "Delay_ms",
  "Firing_Order",
  "Timing_Group",
  "Validation_Warnings",
  "Notes",
];

function csvEscape(value: unknown) {
  const text = String(value ?? "");
  return text.includes(",") || text.includes('"') || text.includes("\n") ? `"${text.replace(/"/g, '""')}"` : text;
}

export function delayAssignmentRows(holes: BlastHole[], pattern: TimingPattern) {
  return holes.map((hole) => ({
    "Hole ID": hole.id,
    "Original Hole ID": hole.originalId ?? "",
    X: hole.x,
    Y: hole.y,
    Z: hole.z ?? "",
    Depth: hole.depth ?? "",
    Charge: hole.charge ?? "",
    "Row Index": hole.rowIndex ?? "",
    "Column Index": hole.columnIndex ?? "",
    "Timing Pattern": TIMING_PATTERN_LABELS[pattern],
    Delay_ms: hole.delayMs ?? "",
    Firing_Order: hole.firingOrder ?? "",
    Timing_Group: hole.timingGroup ?? "",
    Validation_Warnings: (hole.validationWarnings ?? []).join("; "),
    Notes: "Planning/Simulation Draft",
  }));
}

export function buildDelayAssignmentCsv(holes: BlastHole[], pattern: TimingPattern) {
  const rows = delayAssignmentRows(holes, pattern);
  const lines = [COLUMNS.join(",")];
  rows.forEach((row) => {
    lines.push(COLUMNS.map((column) => csvEscape(row[column as keyof typeof row])).join(","));
  });
  return lines.join("\n");
}

export function downloadTextFile(content: string, filename: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function exportWarnings(issues: ValidationIssue[]) {
  const unassigned = issues.filter((issue) => issue.field === "delayMs" && issue.severity === "error").length;
  if (unassigned) return `${unassigned} delay assignment issue(s) remain. Export will be marked incomplete.`;
  return "";
}
