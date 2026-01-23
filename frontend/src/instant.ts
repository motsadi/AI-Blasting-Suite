import { init } from "@instantdb/react";

const appId = (import.meta.env.VITE_INSTANTDB_APP_ID as string | undefined) ?? "";
const requireAuthEnv = (import.meta.env.VITE_REQUIRE_AUTH as string | undefined) ?? null;
export const REQUIRE_AUTH =
  requireAuthEnv !== null
    ? requireAuthEnv.toLowerCase() === "true"
    : true;

// We export a lazily-initialized db to avoid hard-crashing the app when env vars are missing.
export function getDb() {
  if (!appId) {
    throw new Error("Missing VITE_INSTANTDB_APP_ID");
  }
  return init({ appId });
}

export const INSTANT_APP_ID = appId;

