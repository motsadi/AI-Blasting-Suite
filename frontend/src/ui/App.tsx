import { useEffect, useMemo, useState } from "react";
import { LoginCard } from "./LoginCard";
import { Shell } from "./Shell";
import { getDb, REQUIRE_AUTH } from "../instant";

type Session = {
  token: string;
  email: string;
};

export function App() {
  const [session, setSession] = useState<Session | null>(
    REQUIRE_AUTH ? null : { token: "local", email: "Local" }
  );
  const [booting, setBooting] = useState(REQUIRE_AUTH);

  const apiBaseUrl = useMemo(() => {
    return (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";
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

  if (booting) {
    return (
      <div className="container" style={{ display: "grid", placeItems: "center" }}>Loading…</div>
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

  return (
    <Shell
      apiBaseUrl={apiBaseUrl}
      session={session}
      onLogout={() => {
        localStorage.removeItem("instant_refresh_token");
        localStorage.removeItem("instant_email");
        setSession(null);
      }}
    />
  );
}

