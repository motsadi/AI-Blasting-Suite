export type TimingPattern =
  | "manual"
  | "rowByRow"
  | "chevron"
  | "vCut"
  | "boxCut"
  | "pointDirectional"
  | "lineDirectional";

export type TimingDirection = "leftToRight" | "rightToLeft" | "bottomToTop" | "topToBottom";

export type ColorMode = "delay" | "row" | "charge" | "depth" | "group";

export type ValidationSeverity = "error" | "warning";

export interface BlastHole {
  id: string;
  originalId?: string;
  x: number;
  y: number;
  z?: number;
  depth?: number;
  charge?: number;
  rowIndex?: number;
  columnIndex?: number;
  delayMs?: number;
  firingOrder?: number;
  timingGroup?: string;
  selected?: boolean;
  validationWarnings?: string[];
}

export interface TimingSettings {
  startDelayMs: number;
  inRowDelayMs: number;
  rowDelayMs: number;
  minDelayMs: number;
  maxDelayMs: number;
  delayIncrementMs: number;
  rowTolerance: number;
  direction: TimingDirection;
  reverseOrder: boolean;
  applyRounding: boolean;
}

export interface BlastProject {
  projectName: string;
  importedFileName: string;
  holes: BlastHole[];
  timingPattern: TimingPattern;
  settings: TimingSettings;
  createdAt: string;
  updatedAt: string;
}

export interface ValidationIssue {
  severity: ValidationSeverity;
  message: string;
  holeId?: string;
  field?: string;
  suggestion?: string;
}

export interface ColumnMapping {
  id?: string;
  x?: string;
  y?: string;
  z?: string;
  depth?: string;
  charge?: string;
}

export interface ParsedCsv {
  rows: Array<Record<string, string>>;
  columns: string[];
  mapping: ColumnMapping;
  holes: BlastHole[];
  issues: ValidationIssue[];
}

export interface TimingPoint {
  x: number;
  y: number;
}

export interface TimingLine {
  start: TimingPoint;
  end: TimingPoint;
}

export const DEFAULT_TIMING_SETTINGS: TimingSettings = {
  startDelayMs: 0,
  inRowDelayMs: 17,
  rowDelayMs: 42,
  minDelayMs: 0,
  maxDelayMs: 10000,
  delayIncrementMs: 1,
  rowTolerance: 8,
  direction: "leftToRight",
  reverseOrder: false,
  applyRounding: true,
};

export const TIMING_PATTERN_LABELS: Record<TimingPattern, string> = {
  manual: "Manual timing",
  rowByRow: "Row-by-row",
  chevron: "Chevron",
  vCut: "V-cut",
  boxCut: "Box-cut / centre-out",
  pointDirectional: "Directional from point",
  lineDirectional: "Directional from line",
};
