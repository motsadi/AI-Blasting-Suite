import type { BlastProject } from "../types/blast";

export function buildProjectJson(project: BlastProject) {
  return JSON.stringify({ ...project, updatedAt: new Date().toISOString() }, null, 2);
}
