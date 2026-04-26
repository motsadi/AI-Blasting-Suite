import type { BlastHole, ColumnMapping, ParsedCsv, ValidationIssue } from "../types/blast";

const ID_ALIASES = ["hole id", "holeid", "id", "hole", "hole_no", "hole number", "hole_number"];
const X_ALIASES = ["x", "easting", "east", "x coordinate", "x_coordinate"];
const Y_ALIASES = ["y", "northing", "north", "y coordinate", "y_coordinate"];
const Z_ALIASES = ["z", "rl", "elev", "elevation", "z coordinate", "z_coordinate"];
const DEPTH_ALIASES = ["depth", "hole depth", "hole_depth", "hole depth (m)", "hole_depth_m"];
const CHARGE_ALIASES = ["charge", "charge_kg", "explosive mass", "explosive_mass", "explosive mass (kg)"];

function normalizeHeader(value: string) {
  return value.trim().toLowerCase();
}

function findColumn(columns: string[], aliases: string[]) {
  const normalized = new Map(columns.map((column) => [normalizeHeader(column), column]));
  for (const alias of aliases) {
    const match = normalized.get(normalizeHeader(alias));
    if (match) return match;
  }
  return undefined;
}

export function inferColumnMapping(columns: string[]): ColumnMapping {
  return {
    id: findColumn(columns, ID_ALIASES),
    x: findColumn(columns, X_ALIASES),
    y: findColumn(columns, Y_ALIASES),
    z: findColumn(columns, Z_ALIASES),
    depth: findColumn(columns, DEPTH_ALIASES),
    charge: findColumn(columns, CHARGE_ALIASES),
  };
}

function parseCsvLine(line: string) {
  const values: string[] = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    const next = line[i + 1];
    if (ch === '"' && quoted && next === '"') {
      current += '"';
      i += 1;
    } else if (ch === '"') {
      quoted = !quoted;
    } else if (ch === "," && !quoted) {
      values.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  values.push(current);
  return values.map((value) => value.trim());
}

export function parseCsvText(text: string) {
  const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (!lines.length) return { columns: [] as string[], rows: [] as Array<Record<string, string>> };
  const columns = parseCsvLine(lines[0]).map((column, idx) => column || `Column ${idx + 1}`);
  const rows = lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    return columns.reduce<Record<string, string>>((acc, column, idx) => {
      acc[column] = values[idx] ?? "";
      return acc;
    }, {});
  });
  return { columns, rows };
}

function toNumber(value: string | undefined) {
  if (value == null || value.trim() === "") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function isMissingId(value: string | undefined) {
  const normalized = String(value ?? "").trim().toLowerCase();
  return !normalized || ["n/a", "na", "nan", "none", "null", "-"].includes(normalized);
}

export function rowsToBlastHoles(rows: Array<Record<string, string>>, mapping: ColumnMapping) {
  const issues: ValidationIssue[] = [];
  const holes: BlastHole[] = [];
  if (!mapping.x || !mapping.y) {
    issues.push({
      severity: "error",
      message: "CSV must include mappable X and Y coordinate columns.",
      suggestion: "Use the column mapping controls to select X and Y.",
    });
    return { holes, issues };
  }

  rows.forEach((row, index) => {
    const x = toNumber(row[mapping.x as string]);
    const y = toNumber(row[mapping.y as string]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      issues.push({
        severity: "error",
        message: `Row ${index + 2} has missing or non-numeric X/Y coordinates.`,
        field: "X/Y",
        suggestion: "Check coordinate values before assigning delays.",
      });
      return;
    }

    const originalId = mapping.id ? row[mapping.id] : "";
    const generatedId = `H${String(index + 1).padStart(3, "0")}`;
    const id = isMissingId(originalId) ? generatedId : String(originalId).trim();
    const warnings: string[] = [];
    if (isMissingId(originalId)) {
      warnings.push(`Original Hole ID "${originalId || "blank"}" replaced with ${generatedId}.`);
      issues.push({
        severity: "warning",
        message: `Row ${index + 2} had a blank/N/A Hole ID and was assigned ${generatedId}.`,
        holeId: generatedId,
      });
    }

    const z = mapping.z ? toNumber(row[mapping.z]) : undefined;
    const depth = mapping.depth ? toNumber(row[mapping.depth]) : undefined;
    const charge = mapping.charge ? toNumber(row[mapping.charge]) : undefined;

    holes.push({
      id,
      originalId: originalId || undefined,
      x: x as number,
      y: y as number,
      z: Number.isFinite(z) ? z : undefined,
      depth: Number.isFinite(depth) ? depth : undefined,
      charge: Number.isFinite(charge) ? charge : undefined,
      validationWarnings: warnings,
    });
  });

  return { holes, issues };
}

export function parseBlastCsv(text: string): ParsedCsv {
  const { columns, rows } = parseCsvText(text);
  const mapping = inferColumnMapping(columns);
  const { holes, issues } = rowsToBlastHoles(rows, mapping);
  return { rows, columns, mapping, holes, issues };
}
