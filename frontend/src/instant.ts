import { init } from "@instantdb/react";

const appId = (import.meta.env.VITE_INSTANTDB_APP_ID as string | undefined) ?? "";

// We export a lazily-initialized db to avoid hard-crashing the app when env vars are missing.
export function getDb() {
  if (!appId) {
    throw new Error("Missing VITE_INSTANTDB_APP_ID");
  }
  return init({ appId });
}

export const INSTANT_APP_ID = appId;

