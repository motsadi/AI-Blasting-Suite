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
];

const allowedLoginEmails = new Set(
  ((import.meta.env.VITE_ALLOWED_LOGIN_EMAILS as string | undefined)?.split(",") ?? DEFAULT_ALLOWED_LOGIN_EMAILS)
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

          <div className="pill loginPill">AI-assisted blasting intelligence</div>
          <h1 className="loginHeroTitle">Design safer, smarter and more predictable blasts.</h1>
          <p className="loginHeroCopy">
            The AI Blasting Suite brings blast planning, vibration insight, fragmentation forecasting
            and explainable recommendations into one production-ready workspace.
          </p>

          <div className="loginHeroVisual" aria-hidden="true">
            <div className="loginBlastBadge">Blast design + AI</div>
            <svg viewBox="0 0 520 320" className="loginHeroSvg" role="presentation">
              <defs>
                <linearGradient id="blastSky" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="rgba(15, 23, 42, 0.98)" />
                  <stop offset="55%" stopColor="rgba(30, 41, 59, 0.96)" />
                  <stop offset="100%" stopColor="rgba(37, 99, 235, 0.88)" />
                </linearGradient>
                <linearGradient id="blastWave" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="rgba(56, 189, 248, 0.08)" />
                  <stop offset="50%" stopColor="rgba(96, 165, 250, 0.95)" />
                  <stop offset="100%" stopColor="rgba(56, 189, 248, 0.08)" />
                </linearGradient>
                <linearGradient id="terrain" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="rgba(249, 115, 22, 0.9)" />
                  <stop offset="100%" stopColor="rgba(245, 158, 11, 0.94)" />
                </linearGradient>
              </defs>

              <rect x="0" y="0" width="520" height="320" rx="28" fill="url(#blastSky)" />
              <circle cx="96" cy="72" r="56" fill="rgba(56, 189, 248, 0.16)" />
              <circle cx="420" cy="62" r="78" fill="rgba(59, 130, 246, 0.16)" />
              <path
                d="M0 236 C84 214 132 178 194 188 C258 198 286 236 352 226 C418 216 454 184 520 176 L520 320 L0 320 Z"
                fill="url(#terrain)"
                opacity="0.96"
              />
              <path
                d="M0 248 C96 226 136 202 196 210 C272 220 304 252 370 244 C430 236 468 210 520 202"
                fill="none"
                stroke="rgba(255,255,255,0.24)"
                strokeWidth="2"
              />

              <g opacity="0.96">
                <rect x="112" y="104" width="14" height="120" rx="7" fill="rgba(255,255,255,0.92)" />
                <rect x="176" y="96" width="14" height="128" rx="7" fill="rgba(255,255,255,0.92)" />
                <rect x="240" y="114" width="14" height="110" rx="7" fill="rgba(255,255,255,0.92)" />
                <rect x="304" y="100" width="14" height="124" rx="7" fill="rgba(255,255,255,0.92)" />
                <rect x="368" y="118" width="14" height="106" rx="7" fill="rgba(255,255,255,0.92)" />
              </g>

              <g>
                <circle cx="183" cy="146" r="58" fill="none" stroke="url(#blastWave)" strokeWidth="6" />
                <circle cx="183" cy="146" r="92" fill="none" stroke="url(#blastWave)" strokeWidth="4" opacity="0.85" />
                <circle cx="183" cy="146" r="124" fill="none" stroke="url(#blastWave)" strokeWidth="3" opacity="0.55" />
                <circle cx="183" cy="146" r="10" fill="rgba(255,255,255,0.98)" />
              </g>

              <g>
                <rect x="330" y="54" width="116" height="92" rx="18" fill="rgba(255,255,255,0.12)" stroke="rgba(255,255,255,0.24)" />
                <path d="M354 116 L378 92 L396 104 L424 74" fill="none" stroke="rgba(125,211,252,0.95)" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
                <circle cx="354" cy="116" r="6" fill="rgba(255,255,255,0.94)" />
                <circle cx="378" cy="92" r="6" fill="rgba(255,255,255,0.94)" />
                <circle cx="396" cy="104" r="6" fill="rgba(255,255,255,0.94)" />
                <circle cx="424" cy="74" r="6" fill="rgba(255,255,255,0.94)" />
                <path d="M352 132 H428" stroke="rgba(255,255,255,0.2)" strokeWidth="4" strokeLinecap="round" />
              </g>
            </svg>
          </div>

          <div className="loginFeatureGrid">
            <div className="loginFeatureCard">
              <div className="loginFeatureLabel">Prediction</div>
              <div className="loginFeatureText">Forecast vibration, fragmentation and airblast before execution.</div>
            </div>
            <div className="loginFeatureCard">
              <div className="loginFeatureLabel">Optimization</div>
              <div className="loginFeatureText">Compare scenarios and tune blast parameters with AI support.</div>
            </div>
            <div className="loginFeatureCard">
              <div className="loginFeatureLabel">Explainability</div>
              <div className="loginFeatureText">Understand which variables drive outcomes and recommended actions.</div>
            </div>
          </div>
        </section>

        <section className="card loginCardPanel">
          <div className="pill" style={{ display: "inline-flex", marginBottom: 10 }}>Secure access</div>
          <div className="loginCardTitle">Sign in to your blasting workspace</div>
          <div className="subtitle">
            Use your email to receive a one-time magic code and continue into the platform.
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
