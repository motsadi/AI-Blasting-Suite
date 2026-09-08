import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { LoginCard } from "./LoginCard";
import { Shell } from "./Shell";
import { getDb, REQUIRE_AUTH } from "../instant";
import { isGeoMotionStandalone } from "./geomotion/standalone";

const GeoMotionPanel = lazy(() =>
  import("./geomotion/GeoMotionPanel").then((module) => ({ default: module.GeoMotionPanel }))
);

type Session = {
  token: string;
  email: string;
};

export function App() {
  const [session, setSession] = useState<Session | null>(
    REQUIRE_AUTH ? null : { token: "local", email: "Local" }
  );
  const [booting, setBooting] = useState(REQUIRE_AUTH);
  const standaloneGeoMotion = useMemo(() => isGeoMotionStandalone(), []);

  const apiBaseUrl = useMemo(() => {
    const fromEnv = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";
    // If not configured, default to same-origin so Vercel rewrites can route `/v1/*` -> `/api/v1/*`.
    return fromEnv || window.location.origin;
  }, []);

  useEffect(() => {
    if (!REQUIRE_AUTH) return;
    // Restore prior session (refresh_token) if present.
    const token = localStorage.getItem("instant_refresh_token");
    const email = localStorage.getItem("instant_email");
    if (!token) {
      setBooting(false);
      return;
    }
    (async () => {
      try {
        const db = getDb();
        const res = await db.auth.signInWithToken(token);
        const e = res?.user?.email ?? email ?? "";
        setSession({ token, email: e });
      } catch {
        localStorage.removeItem("instant_refresh_token");
        localStorage.removeItem("instant_email");
      } finally {
        setBooting(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!standaloneGeoMotion) return;
    const previousTitle = document.title;
    document.title = "GeoMotion 3D · BlastOps";
    document.body.classList.add("geomotionStandaloneBody");
    return () => {
      document.title = previousTitle;
      document.body.classList.remove("geomotionStandaloneBody");
    };
  }, [standaloneGeoMotion]);

  function logout() {
    localStorage.removeItem("instant_refresh_token");
    localStorage.removeItem("instant_email");
    setSession(null);
  }

  if (booting) {
    return (
      <div
        className={standaloneGeoMotion ? "geomotionStandaloneLoading" : "container"}
        style={{ display: "grid", placeItems: "center" }}
      >
        Loading…
      </div>
    );
  }

  if (!session) {
    return (
      <LoginCard
        onLogin={(s) => setSession(s)}
        apiBaseUrl={apiBaseUrl}
      />
    );
  }

  if (standaloneGeoMotion) {
    return (
      <div className="geomotionStandaloneShell" data-testid="geomotion-standalone">
        <Suspense fallback={<div className="geomotionStandaloneLoading">Loading GeoMotion 3D…</div>}>
          <GeoMotionPanel
            apiBaseUrl={apiBaseUrl}
            token={session.token}
            standalone
            userEmail={session.email}
            onLogout={logout}
          />
        </Suspense>
      </div>
    );
  }

  return (
    <Shell
      apiBaseUrl={apiBaseUrl}
      session={session}
      onLogout={logout}
    />
  );
}

