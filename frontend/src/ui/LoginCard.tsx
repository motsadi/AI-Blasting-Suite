import { useMemo, useState } from "react";
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
    <div style={page}>
      <div style={card}>
        <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: "-0.02em" }}>
          Sign in
        </div>
        <div style={{ color: "#64748b", marginTop: 6 }}>
          Email login with a verification code.
        </div>

        <div style={{ marginTop: 14, display: "grid", gap: 10 }}>
          {!INSTANT_APP_ID && (
            <div style={warn}>
              Missing <code>VITE_INSTANTDB_APP_ID</code>. Set it in Vercel env vars (or locally) and
              reload.
            </div>
          )}

          <label style={label}>Email</label>
          <input
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@company.com"
            style={input}
          />

          {step === "code" && (
            <>
              <label style={label}>Verification code</label>
              <input
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="123456"
                style={input}
              />
            </>
          )}

          {err && <div style={errBox}>{err}</div>}

          {step === "email" ? (
            <button onClick={requestCode} style={btnPrimary} disabled={!email.includes("@") || busy || !INSTANT_APP_ID}>
              {busy ? "Sending…" : "Send code"}
            </button>
          ) : (
            <button onClick={verifyCode} style={btnPrimary} disabled={!code.trim() || busy || !INSTANT_APP_ID}>
              {busy ? "Verifying…" : "Verify & continue"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

const page = {
  minHeight: "100vh",
  display: "grid",
  placeItems: "center",
  background: "radial-gradient(1200px 600px at 20% 20%, rgba(99,102,241,0.25), transparent), #0b1220",
  padding: 16,
} as const;

const card = {
  width: "min(560px, 100%)",
  background: "linear-gradient(135deg, #ffffff, #f8fafc)",
  border: "1px solid #e2e8f0",
  borderRadius: 18,
  padding: 18,
  boxShadow: "0 18px 50px rgba(15, 23, 42, 0.25)",
} as const;

const label = { fontSize: 12, color: "#475569", fontWeight: 700 } as const;
const input = {
  border: "1px solid #e2e8f0",
  borderRadius: 12,
  padding: "10px 12px",
  outline: "none",
  fontSize: 14,
} as const;
const btnPrimary = {
  border: "1px solid #0f172a",
  background: "#0f172a",
  color: "white",
  padding: "10px 12px",
  borderRadius: 12,
  fontWeight: 700,
  cursor: "pointer",
} as const;

const errBox = {
  background: "#fff1f2",
  color: "#9f1239",
  border: "1px solid #fecdd3",
  borderRadius: 12,
  padding: "10px 12px",
  fontSize: 13,
  whiteSpace: "pre-wrap",
} as const;

const warn = {
  background: "#fffbeb",
  color: "#92400e",
  border: "1px solid #fde68a",
  borderRadius: 12,
  padding: "10px 12px",
  fontSize: 13,
} as const;
