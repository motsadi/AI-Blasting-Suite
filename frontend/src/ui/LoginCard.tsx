import { useState } from "react";
import { getDb, INSTANT_APP_ID } from "../instant";

type Props = {
  apiBaseUrl: string;
  onLogin: (session: { token: string; email: string }) => void;
};

const DEFAULT_ALLOWED_LOGIN_EMAILS = [
  "so13000604@biust.ac.bw",
  "Ozigwa@debswana.bw",
  "Tgalefete@debswana.bw",
  "Mhiya@debswana.bw",
  "Ttshambane@debswana.bw",
  "Mgaopelo@debswana.bw",
  "SMoabi@debswana.bw",
  "MMoleofe@debswana.bw",
];

const allowedLoginEmails = new Set(
  [
    ...DEFAULT_ALLOWED_LOGIN_EMAILS,
    ...((import.meta.env.VITE_ALLOWED_LOGIN_EMAILS as string | undefined)?.split(",") ?? []),
  ]
    .map((email) => email.trim().toLowerCase())
    .filter(Boolean)
);

export function LoginCard({ onLogin }: Props) {
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [step, setStep] = useState<"email" | "code">("email");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const normalizedEmail = email.trim().toLowerCase();
  const isAllowedEmail = allowedLoginEmails.has(normalizedEmail);
  const canRequestCode = email.includes("@") && isAllowedEmail && !!INSTANT_APP_ID;
  const canVerifyCode = !!code.trim() && isAllowedEmail && !!INSTANT_APP_ID;

  async function requestCode() {
    if (!email.includes("@")) return;
    if (!isAllowedEmail) {
      setErr("This email is not authorized for this application.");
      return;
    }
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
      const signedInEmail = res.user.email ?? email;
      if (!allowedLoginEmails.has(signedInEmail.trim().toLowerCase())) {
        throw new Error("This email is not authorized for this application.");
      }
      const refreshToken = res?.user?.refresh_token;
      if (!refreshToken) {
        throw new Error("No refresh_token returned from InstantDB");
      }
      localStorage.setItem("instant_refresh_token", refreshToken);
      localStorage.setItem("instant_email", signedInEmail);
      onLogin({ email: signedInEmail, token: refreshToken });
    } catch (e: any) {
      setErr(String(e?.body?.message ?? e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="container loginPage">
      <div className="loginLayout">
        <section className="loginHero">
          <div className="loginHeroGlow loginHeroGlowPrimary" />
          <div className="loginHeroGlow loginHeroGlowSecondary" />

          <div className="loginBrandRow">
            <div className="loginBrandMark">B</div>
            <div>
              <strong>BlastOps</strong>
              <span>Mine intelligence workspace</span>
            </div>
            <div className="loginLiveBadge"><i /> Operational analytics</div>
          </div>

          <div className="loginEyebrow">FROM BLAST DESIGN TO ORE RECOVERY</div>
          <h1 className="loginHeroTitle">See the movement.<br />Protect the ore.</h1>
          <p className="loginHeroCopy">
            One focused workspace for blast predictions, cost control, wall protection and
            three-dimensional material movement.
          </p>

          <div className="loginCommandVisual" aria-hidden="true">
            <div className="loginVisualHeader">
              <span>GeoMotion 3D</span>
              <span>Post-blast movement model</span>
            </div>
            <svg viewBox="0 0 620 250" className="loginHeroSvg" role="presentation">
              <defs>
                <linearGradient id="loginGrid" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#38bdf8" />
                  <stop offset="100%" stopColor="#14b8a6" />
                </linearGradient>
                <filter id="loginGlow"><feGaussianBlur stdDeviation="5" result="blur" /></filter>
              </defs>
              <path d="M90 174 L280 78 L520 158 L330 238 Z" fill="rgba(8,20,38,.86)" stroke="rgba(125,211,252,.34)" strokeWidth="2" />
              <g opacity=".25" stroke="#7dd3fc">
                {[0,1,2,3,4,5,6].map((i) => <path key={`gx-${i}`} d={`M${90+i*31.7} ${174-i*16} L${330+i*31.7} ${238-i*16}`} />)}
                {[0,1,2,3,4,5].map((i) => <path key={`gy-${i}`} d={`M${90+i*48} ${174+i*16} L${280+i*48} ${78+i*16}`} />)}
              </g>
              <g fill="url(#loginGrid)">
                {[0,1,2,3,4].flatMap((row) => [0,1,2,3,4,5].map((column) => {
                  const x = 205 + column * 38 + row * 18;
                  const y = 95 + row * 24 - column * 7;
                  const ore = row > 1 && column > 1 && column < 5;
                  return <rect key={`${row}-${column}`} x={x} y={y} width="25" height="25" rx="3" fill={ore ? "#22c55e" : "#475569"} opacity={ore ? ".95" : ".72"} />;
                }))}
              </g>
              <g stroke="#fb923c" strokeWidth="3" strokeLinecap="round" opacity=".9">
                <path d="M236 121 l-24 -18" /><path d="M306 135 l-22 -24" /><path d="M378 149 l-18 -27" />
              </g>
              <circle cx="212" cy="103" r="7" fill="#fdba74" filter="url(#loginGlow)" />
              <circle cx="284" cy="111" r="7" fill="#fdba74" filter="url(#loginGlow)" />
              <circle cx="360" cy="122" r="7" fill="#fdba74" filter="url(#loginGlow)" />
              <text x="38" y="48" fill="#7dd3fc" fontSize="12" fontWeight="700">SOURCE</text>
              <text x="500" y="220" fill="#5eead4" fontSize="12" fontWeight="700">DESTINATION</text>
            </svg>
            <div className="loginVisualMetrics">
              <div><span>Movement</span><strong>3D vectors</strong></div>
              <div><span>Ore control</span><strong>Recovery</strong></div>
              <div><span>Risk</span><strong>Dilution</strong></div>
            </div>
          </div>

          <div className="loginCapabilityRow">
            <span>Prediction</span><span>Cost optimisation</span><span>GeoMotion 3D</span><span>Safety analytics</span>
          </div>
        </section>

        <section className="card loginCardPanel">
          <div className="loginAccessIcon">↗</div>
          <div className="loginCardTitle">Welcome back</div>
          <div className="subtitle">
            Sign in securely to continue to the blasting workspace.
          </div>

          <div className="loginProgress">
            <div className={`loginProgressStep ${step === "email" ? "loginProgressStepActive" : ""}`}>1. Enter email</div>
            <div className={`loginProgressStep ${step === "code" ? "loginProgressStepActive" : ""}`}>2. Verify code</div>
          </div>

          <div style={{ marginTop: 14, display: "grid", gap: 10 }}>
            {!INSTANT_APP_ID && (
              <div className="error">
                Missing <code>VITE_INSTANTDB_APP_ID</code>. Set it in Vercel env vars (or locally) and
                reload.
              </div>
            )}

            <label className="label">Work email</label>
            <input
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@company.com"
              className="input"
              autoComplete="email"
            />

            {step === "code" && (
              <>
                <div className="loginCodeHeader">
                  <label className="label">Verification code</label>
                  <button
                    type="button"
                    className="loginLinkButton"
                    onClick={() => {
                      setStep("email");
                      setCode("");
                    }}
                    disabled={busy}
                  >
                    Change email
                  </button>
                </div>
                <input
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  placeholder="123456"
                  className="input"
                  inputMode="numeric"
                  autoComplete="one-time-code"
                />
                <div className="subtitle">
                  We sent a login code to <strong>{email}</strong>. Enter it to continue.
                </div>
              </>
            )}

            {err && <div className="error">{err}</div>}

            {step === "email" ? (
              <button onClick={requestCode} className="btn btnPrimary loginPrimaryAction" disabled={!canRequestCode || busy}>
                {busy ? "Sending..." : "Send magic code"}
              </button>
            ) : (
              <button onClick={verifyCode} className="btn btnPrimary loginPrimaryAction" disabled={!canVerifyCode || busy}>
                {busy ? "Verifying..." : "Verify and continue"}
              </button>
            )}
          </div>

          <div className="loginTrustRow">
            <div className="loginTrustItem">Protected sign-in</div>
            <div className="loginTrustItem">No password required</div>
            <div className="loginTrustItem">Fast access for operations teams</div>
          </div>
        </section>
      </div>
    </div>
  );
}
