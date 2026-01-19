import { useState } from "react";
import { getDb, INSTANT_APP_ID } from "../instant";

type Props = {
  apiBaseUrl: string;
  onLogin: (session: { token: string; email: string }) => void;
};

export function LoginCard({ onLogin }: Props) {
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [step, setStep] = useState<"email" | "code">("email");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function requestCode() {
    if (!email.includes("@")) return;
    setErr(null);
    setBusy(true);
    try {
      const db = getDb();
      await db.auth.sendMagicCode({ email });
      setStep("code");
    } catch (e: any) {
      setErr(String(e?.body?.message ?? e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  async function verifyCode() {
    if (!code.trim()) return;
    setErr(null);
    setBusy(true);
    try {
      const db = getDb();
      const res = await db.auth.signInWithMagicCode({ email, code: code.trim() });
      const refreshToken = res?.user?.refresh_token;
      if (!refreshToken) {
        throw new Error("No refresh_token returned from InstantDB");
      }
      localStorage.setItem("instant_refresh_token", refreshToken);
      localStorage.setItem("instant_email", res.user.email ?? email);
      onLogin({ email: res.user.email ?? email, token: refreshToken });
    } catch (e: any) {
      setErr(String(e?.body?.message ?? e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="container" style={{ display: "grid", placeItems: "center" }}>
      <div className="card" style={{ width: "min(560px, 100%)" }}>
        <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.02em" }}>Sign in</div>
        <div className="subtitle">Secure email login via magic code (InstantDB).</div>

        <div style={{ marginTop: 14, display: "grid", gap: 10 }}>
          {!INSTANT_APP_ID && (
            <div className="error">
              Missing <code>VITE_INSTANTDB_APP_ID</code>. Set it in Vercel env vars (or locally) and
              reload.
            </div>
          )}

          <label className="label">Email</label>
          <input
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@company.com"
            className="input"
          />

          {step === "code" && (
            <>
              <label className="label">Verification code</label>
              <input
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="123456"
                className="input"
              />
            </>
          )}

          {err && <div className="error">{err}</div>}

          {step === "email" ? (
            <button onClick={requestCode} className="btn btnPrimary" disabled={!email.includes("@") || busy || !INSTANT_APP_ID}>
              {busy ? "Sending…" : "Send code"}
            </button>
          ) : (
            <button onClick={verifyCode} className="btn btnPrimary" disabled={!code.trim() || busy || !INSTANT_APP_ID}>
              {busy ? "Verifying…" : "Verify & continue"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
