
import os, numpy as np, pandas as pd, joblib, streamlit as st, matplotlib.pyplot as plt
import streamlit_authenticator as stauth

from utils_blaster import (
    INPUT_LABELS, slider_ranges_from_df, EmpiricalParams, empirical_predictions,
    estimate_n_rr, rr_lambda_from_xm_n, rr_cdf
)

st.set_page_config(page_title="AI Blasting Suite", layout="wide")

# --- Login (streamlit-authenticator; credentials come from Secrets) ---
def build_authenticator():
    creds = st.secrets.get("credentials", {})
    cookie  = st.secrets.get("cookie", {})
    if not creds or "usernames" not in creds:
        st.error("Credentials not configured. Add them in Streamlit Secrets. See README.")
        st.stop()
    return stauth.Authenticate(
        creds,
        cookie.get("name","blasting_cookie"),
        cookie.get("key","supersecret_key_change_me"),
        cookie.get("expiry_days", 7),
        cookie.get("preauthorized", {}),
    )

authenticator = build_authenticator()
name, auth_status, username = authenticator.login("Login", "main")
if auth_status is False:
    st.error("Invalid username or password"); st.stop()
elif auth_status is None:
    st.info("Enter your credentials to continue."); st.stop()

authenticator.logout("Logout", "sidebar")
st.title("💎 AI Blasting Suite")
st.caption("Data audit • Empirical calibration • ML predictions • Flyrock • Slope stability • Delay simulation")

# --- Optional model loaders ---
st.sidebar.header("📦 Assets")
def safe_load_joblib(p):
    try: return joblib.load(p)
    except Exception: return None

scaler   = safe_load_joblib("scaler1.joblib")
mdl_frag = safe_load_joblib("random_forest_model_Fragmentation.joblib")
mdl_ppv  = safe_load_joblib("random_forest_model_Ground Vibration.joblib")
mdl_air  = safe_load_joblib("random_forest_model_Airblast.joblib")

st.sidebar.write(f"Scaler: {'✅' if scaler else '—'}")
st.sidebar.write(f"Fragmentation RF: {'✅' if mdl_frag else '—'}")
st.sidebar.write(f"Ground Vibration RF: {'✅' if mdl_ppv else '—'}")
st.sidebar.write(f"Airblast RF: {'✅' if mdl_air else '—'}")

tabs = st.tabs(["Data","Predict (ML + Empirical)","Flyrock (Auto-Train)","Slope Stability (Auto-Train)","Delay & Blast Sim"])

# --- Data tab ---
with tabs[0]:
    st.subheader("📑 Data Management")
    up = st.file_uploader("Upload blast CSV (Orapa/Jwaneng or similar)", type=["csv"])
    if up is not None: st.session_state["data_df"] = pd.read_csv(up)
    else:
        for name in ["combinedv2Orapa.csv","combinedv2Jwaneng.csv","Backbreak.csv"]:
            if os.path.exists(name):
                st.session_state["data_df"] = pd.read_csv(name); break
    df = st.session_state.get("data_df")
    if df is not None:
        st.write("### Preview"); st.dataframe(df.head(200))
        with st.expander("Quick KPIs (set limits & HPD)"):
            ppv_lim = st.number_input("PPV limit (mm/s)", value=12.5, step=0.1)
            air_lim = st.number_input("Air limit (dB)", value=134.0, step=0.5)
            hpd     = st.number_input("Holes per delay (HPD)", value=1.0, step=1.0)
            if st.button("Compute KPIs"):
                d = df.copy()
                if "Hole depth" in d and "Stemming" in d:
                    d["Charge length"] = (pd.to_numeric(d["Hole depth"], errors="coerce")
                                          - pd.to_numeric(d["Stemming"], errors="coerce")).clip(lower=0)
                if "Linear charge" in d and "Charge length" in d:
                    d["Mass/hole"] = pd.to_numeric(d["Linear charge"], errors="coerce") * pd.to_numeric(d["Charge length"], errors="coerce")
                if "Explosive mass" in d and "Number of holes" in d:
                    d["Mass/hole"] = pd.to_numeric(d["Explosive mass"], errors="coerce") / pd.to_numeric(d["Number of holes"], errors="coerce")
                if "Mass/hole" in d: d["Q/delay"] = hpd * pd.to_numeric(d["Mass/hole"], errors="coerce")
                out = []
                if "Ground Vibration" in d:
                    pp = pd.to_numeric(d["Ground Vibration"], errors="coerce")
                    out.append(f"PPV pass rate (@{ppv_lim} mm/s): {(pp <= ppv_lim).mean()*100:.1f}%")
                if "Airblast" in d:
                    aa = pd.to_numeric(d["Airblast"], errors="coerce")
                    out.append(f"Airblast pass rate (@{air_lim} dB): {(aa <= air_lim).mean()*100:.1f}%")
                st.info("\n".join(out) if out else "Not enough columns for KPIs.")
        with st.expander("Correlation heatmap"):
            num = df.apply(pd.to_numeric, errors="coerce"); corr = num.corr(numeric_only=True)
            fig, ax = plt.subplots(); im = ax.imshow(corr, aspect="auto")
            ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); ax.set_title("Correlation")
            st.pyplot(fig, use_container_width=True)

# --- Predict tab ---
with tabs[1]:
    st.subheader("🎯 ML + Empirical Prediction")
    df = st.session_state.get("data_df", pd.DataFrame(columns=INPUT_LABELS))
    ranges = slider_ranges_from_df(df, INPUT_LABELS)
    cols = st.columns(3); current_vals = {}
    for i, name in enumerate(INPUT_LABELS):
        lo, md, hi = ranges[name]
        with cols[i % 3]:
            current_vals[name] = st.slider(name, float(lo), float(hi if hi > lo else lo + 1.0), float(md))
    st.markdown("**Empirical parameters** (USBM & Kuz–Ram)")
    pcols = st.columns(6)
    params = EmpiricalParams(
        K_ppv=pcols[0].number_input("K_ppv", value=1000.0),
        beta=pcols[1].number_input("β (PPV exponent)", value=1.60),
        K_air=pcols[2].number_input("K_air (dB)", value=170.0),
        B_air=pcols[3].number_input("B_air (dB/dec)", value=20.0),
        A_kuz=pcols[4].number_input("Rock factor A", value=22.0),
        RWS=pcols[5].number_input("RWS (%)", value=115.0),
    )
    current_vals["HPD_override"] = st.number_input("Holes per delay (for Q/delay)", value=1.0, step=1.0)
    outputs = ["Ground Vibration","Airblast","Fragmentation"]
    emp = empirical_predictions(current_vals, params, outputs)
    ml = {k: np.nan for k in outputs}
    if scaler and (mdl_frag or mdl_ppv or mdl_air):
        X = np.array([[current_vals.get(n, 0.0) for n in INPUT_LABELS]], dtype=float)
        Xs = scaler.transform(X)
        if mdl_ppv:  ml["Ground Vibration"] = float(mdl_ppv.predict(Xs)[0])
        if mdl_air:  ml["Airblast"]         = float(mdl_air.predict(Xs)[0])
        if mdl_frag: ml["Fragmentation"]    = float(mdl_frag.predict(Xs)[0])
    names = outputs
    em_vals = [emp.get(n, np.nan) for n in names]; ml_vals = [ml.get(n, np.nan) for n in names]
    fig, ax = plt.subplots(); x = np.arange(len(names)); w = 0.38
    ax.bar(x - w/2, em_vals, w, label="Empirical", alpha=0.85)
    ax.bar(x + w/2, ml_vals, w, label="ML", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Value")
    ax.set_title("Predicted outputs — Empirical vs ML"); ax.legend()
    st.pyplot(fig, use_container_width=True)
    st.markdown("**Rosin–Rammler Fragmentation (CDF)**")
    mode = st.radio("Uniformity index n", ["Estimate (Kuz–Ram)", "Manual"], horizontal=True)
    if mode.startswith("Estimate"):
        n_est = estimate_n_rr(current_vals["Burden (m)"], current_vals["Hole diameter (mm)"])
        n_use = n_est if n_est is not None else st.number_input("Manual n (fallback)", value=1.8)
        st.caption(f"n (estimated): {n_use:.3g}" if n_est is not None else "Estimate unavailable; using manual.")
    else:
        n_use = st.number_input("Manual n", value=1.8)
    Xm = emp.get("Fragmentation", np.nan)
    if np.isfinite(Xm):
        lam = rr_lambda_from_xm_n(Xm, n_use)
        xs = np.linspace(1, 2000, 400); ys = rr_cdf(xs, lam, n_use) * 100.0
        fig_rr, ax_rr = plt.subplots()
        ax_rr.plot(xs, ys); ax_rr.set_xlabel("Size (mm)"); ax_rr.set_ylabel("Passing (%)")
        ax_rr.set_title(f"RR CDF (Xm≈{Xm:.1f} mm, n≈{n_use:.2f})")
        st.pyplot(fig_rr, use_container_width=True)
    else:
        st.info("Provide inputs to compute Xm (Kuz–Ram) to draw RR curve.")

# --- Flyrock ---
with tabs[2]:
    st.subheader("🪨 Flyrock — Auto-train RF")
    upf = st.file_uploader("Upload a flyrock CSV (last numeric column = target by default)", type=["csv"], key="fr")
    if upf:
        dfr = pd.read_csv(upf)
        num = dfr.apply(pd.to_numeric, errors="coerce")
        y = num.iloc[:, -1]; X = num.iloc[:, :-1]
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]; y = y[mask]
        if X.shape[0] >= 30 and X.shape[1] >= 2:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            rf = RandomForestRegressor(n_estimators=500, random_state=42).fit(Xtr, ytr)
            st.success(f"Model trained. R² train≈{rf.score(Xtr, ytr):.3f}")
            cur = {}
            cols = list(X.columns)[:min(6, X.shape[1])]
            ccols = st.columns(3)
            for i, c in enumerate(cols):
                s = pd.to_numeric(dfr[c], errors="coerce").dropna()
                lo = float(s.quantile(0.02)) if len(s) else 0.0
                hi = float(s.quantile(0.98)) if len(s) else 1.0
                md = float(s.median()) if len(s) else 0.5
                with ccols[i % 3]:
                    cur[c] = st.slider(c, lo, hi if hi > lo else lo + 1.0, md)
            if cur:
                xstar = np.array([[cur.get(c, 0.0) for c in list(X.columns)]], dtype=float)
                yhat = float(rf.predict(xstar)[0])
                st.metric("Predicted flyrock distance", f"{yhat:,.1f}")
        else:
            st.warning("Need ≥30 rows and ≥2 numeric features after cleaning.")

# --- Slope stability ---
with tabs[3]:
    st.subheader("⛰️ Slope Stability — Auto-train RF classifier")
    ups = st.file_uploader("Upload slope CSV (gamma, c, phi, beta, H, ru, status)", type=["csv"], key="slp")
    if ups:
        dfs = pd.read_csv(ups)
        cols = {c.lower().strip(): c for c in dfs.columns}
        def pick(*names): 
            for n in names:
                if n in cols: return cols[n]
            return None
        req = dict(
            gamma=pick("gamma","unit weight"),
            c=pick("c","cohesion"),
            phi=pick("phi","friction angle"),
            beta=pick("beta","slope angle"),
            H=pick("h","height"),
            ru=pick("ru","pore pressure ratio"),
            status=pick("status","label","class")
        )
        if None in req.values():
            st.warning("Missing required columns. Rename to [gamma, c, phi, beta, H, ru, status].")
        else:
            num = dfs[[req["H"],req["beta"],req["c"],req["phi"],req["gamma"],req["ru"]]].apply(pd.to_numeric, errors="coerce")
            ylab = dfs[req["status"]].astype(str).str.strip().str.lower().map({"stable":1,"failure":0,"failed":0,"unstable":0})
            m = num.notna().all(axis=1) & ylab.notna()
            X = num[m]; y = ylab[m]
            if y.nunique() == 2 and len(y) >= 30:
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                pipe = Pipeline([("sc", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42))]).fit(Xtr, ytr)
                st.success(f"Model trained. Acc train≈{pipe.score(Xtr, ytr):.3f}")
                a,b,c,d,e,f = st.columns(6)
                H  = a.slider("H (m)", float(X[req["H"]].min()), float(X[req["H"]].max()), float(X[req["H"]].median()))
                beta = b.slider("β (deg)", float(X[req["beta"]].min()), float(X[req["beta"]].max()), float(X[req["beta"]].median()))
                cc = c.slider("c (kPa)", float(X[req["c"]].min()), float(X[req["c"]].max()), float(X[req["c"]].median()))
                phi = d.slider("φ (deg)", float(X[req["phi"]].min()), float(X[req["phi"]].max()), float(X[req["phi"]].median()))
                gam = e.slider("γ (kN/m³)", float(X[req["gamma"]].min()), float(X[req["gamma"]].max()), float(X[req["gamma"]].median()))
                ru  = f.slider("ru (–)", float(X[req["ru"]].min()), float(X[req["ru"]].max()), float(X[req["ru"]].median()))
                xstar = np.array([[H,beta,cc,phi,gam,ru]])
                prob = float(pipe.predict_proba(xstar)[0,1])
                st.metric("P(Stable)", f"{prob*100:.1f}%")
            else:
                st.warning("Dataset must contain both classes and ≥30 valid rows.")

# --- Delay & Blast Sim ---
with tabs[4]:
    st.subheader("⏱️ Delay Prediction & Plan-View Simulation")
    upld = st.file_uploader("Upload hole CSV (X,Y,Depth,Charge[,Z][,Delay])", type=["csv"], key="delayup")
    if upld:
        dh = pd.read_csv(upld)
        cols = {c.lower().strip(): c for c in dh.columns}
        def getc(*names): 
            for n in names:
                if n in cols: return cols[n]
            return None
        Xc = getc("x"); Yc = getc("y")
        Dc = getc("depth","hole depth (m)","hole_depth")
        Cc = getc("charge","explosive mass","charge_kg")
        Zc = getc("z","elev","elevation","rl")
        Tc = getc("delay","predicted delay (ms)")
        if None in (Xc,Yc,Dc,Cc):
            st.warning("CSV must include at least X, Y, Depth, Charge columns.")
        else:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            keep = [c for c in [Dc,Cc,Xc,Yc,Zc] if c]
            num = dh[keep].apply(pd.to_numeric, errors="coerce").dropna()
            X = num.values
            if Tc and Tc in dh.columns:
                y = pd.to_numeric(dh[Tc], errors="coerce").dropna()
                X = X[:len(y)]
            else:
                y = np.clip(10 + 0.02*X[:,0] + 0.0005*X[:,2], 5, 250)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(Xtr)
            mdl = RandomForestRegressor(n_estimators=200, random_state=42).fit(sc.transform(Xtr), ytr)

            Xs = sc.transform(X)
            yhat = mdl.predict(Xs)
            dfv = pd.DataFrame({
                "X": pd.to_numeric(dh[Xc], errors="coerce"),
                "Y": pd.to_numeric(dh[Yc], errors="coerce"),
                "Delay": yhat
            }).dropna()

            st.write("### Plan view (colour = delay)")
            fig, ax = plt.subplots()
            sca = ax.scatter(dfv["X"], dfv["Y"], c=dfv["Delay"], s=60)
            ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            cb = fig.colorbar(sca, ax=ax); cb.set_label("Predicted Delay (ms)")
            st.pyplot(fig, use_container_width=True)

            st.write("### Time scrubber")
            steps = np.unique(dfv["Delay"].values)
            k = st.slider("Step index", 0, len(steps)-1, 0)
            t = steps[k]
            fired = dfv[dfv["Delay"] <= t]; wait = dfv[dfv["Delay"] > t]
            fig2, ax2 = plt.subplots()
            ax2.scatter(wait["X"], wait["Y"], s=42, c="#9aa0a6", label="Waiting")
            cm = ax2.scatter(fired["X"], fired["Y"], c=fired["Delay"], s=70, label="Fired")
            ax2.set_aspect("equal", adjustable="box"); ax2.legend(); ax2.set_title(f"t = {t:.1f} ms")
            cb2 = fig2.colorbar(cm, ax=ax2); cb2.set_label("Delay (ms)")
            st.pyplot(fig2, use_container_width=True)
