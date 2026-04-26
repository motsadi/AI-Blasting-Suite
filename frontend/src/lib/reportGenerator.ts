import type { BlastProject, ValidationIssue } from "../types/blast";
import { TIMING_PATTERN_LABELS } from "../types/blast";

function esc(value: unknown) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch] as string));
}

export function buildPrintableReport(project: BlastProject, issues: ValidationIssue[]) {
  const delays = project.holes.map((hole) => hole.delayMs).filter((delay): delay is number => Number.isFinite(delay));
  const totalCharge = project.holes.reduce((sum, hole) => sum + (Number.isFinite(hole.charge) ? (hole.charge as number) : 0), 0);
  const rows = project.holes
    .slice()
    .sort((a, b) => (a.firingOrder ?? Number.MAX_SAFE_INTEGER) - (b.firingOrder ?? Number.MAX_SAFE_INTEGER))
    .map(
      (hole) => `<tr><td>${esc(hole.id)}</td><td>${esc(hole.originalId ?? "")}</td><td>${esc(hole.x)}</td><td>${esc(hole.y)}</td><td>${esc(hole.delayMs ?? "")}</td><td>${esc(hole.firingOrder ?? "")}</td><td>${esc(hole.timingGroup ?? "")}</td></tr>`
    )
    .join("");
  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>${esc(project.projectName)} - Delay Assignment Draft</title>
  <style>
    body{font-family:Arial,sans-serif;margin:28px;color:#111827}
    h1{margin-bottom:4px} .muted{color:#6b7280;font-size:13px}
    .card{border:1px solid #d1d5db;border-radius:12px;padding:14px;margin:14px 0}
    table{border-collapse:collapse;width:100%;font-size:12px}
    th,td{border:1px solid #d1d5db;padding:6px;text-align:left}
    th{background:#f3f4f6}
  </style>
</head>
<body>
  <h1>${esc(project.projectName)} Delay Assignment</h1>
  <div class="muted">Planning/Simulation Draft - generated ${esc(new Date().toLocaleString())}</div>
  <div class="card">
    <strong>Safety scope:</strong>
    This tool provides planning and simulation support only. Real blast performance depends on geology, burden, spacing,
    explosive type, charge distribution, confinement, stemming, timing accuracy, initiation system, and site-specific
    conditions. All designs must be reviewed and approved by qualified blasting personnel.
  </div>
  <div class="card">
    <p><strong>Imported file:</strong> ${esc(project.importedFileName || "Not specified")}</p>
    <p><strong>Timing pattern:</strong> ${esc(TIMING_PATTERN_LABELS[project.timingPattern])}</p>
    <p><strong>Holes:</strong> ${project.holes.length} | <strong>Total charge:</strong> ${totalCharge.toFixed(2)} | <strong>Duration:</strong> ${delays.length ? `${(Math.max(...delays) - Math.min(...delays)).toFixed(0)} ms` : "No delays"}</p>
    <p><strong>Settings:</strong> start ${project.settings.startDelayMs} ms, in-row ${project.settings.inRowDelayMs} ms, row ${project.settings.rowDelayMs} ms, rounding ${project.settings.applyRounding ? project.settings.delayIncrementMs : "off"}</p>
  </div>
  <div class="card">
    <h2>Validation</h2>
    <ul>${issues.length ? issues.map((issue) => `<li>${esc(issue.severity.toUpperCase())}: ${esc(issue.message)}</li>`).join("") : "<li>No validation issues reported.</li>"}</ul>
  </div>
  <h2>Hole Delay Table</h2>
  <table>
    <thead><tr><th>Hole ID</th><th>Original ID</th><th>X</th><th>Y</th><th>Delay ms</th><th>Firing order</th><th>Group</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
</body>
</html>`;
}

export function openPrintableReport(html: string) {
  const win = window.open("", "_blank");
  if (!win) return false;
  win.document.write(html);
  win.document.close();
  return true;
}
