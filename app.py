import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="CROPIX AI", page_icon="🌾", layout="wide")

COLS = dict(
    crop="Crop",
    year="Crop_Year",
    state="State",
    season="Season",          
    area="Area",
    prod="Production",
    rain="Annual_Rainfall",
    fert="Fertilizer",
    pest="Pesticide",
    yld="Yield",
)
C = COLS



# DATA LOADING

@st.cache_data
def load_data(source):
    df = pd.read_csv(source)
    df.columns = df.columns.str.strip()
    for c in (C["crop"], C["state"], C["season"]):
        if c and c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    required = [C[k] for k in ("crop", "year", "state", "area", "prod", "rain", "fert", "pest", "yld")]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return df, missing
    for c in (C["year"], C["area"], C["prod"], C["rain"], C["fert"], C["pest"], C["yld"]):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[C["crop"], C["state"], C["year"], C["yld"]])
    df[C["year"]] = df[C["year"]].astype(int)
    area = df[C["area"]].replace(0, np.nan)
    df["Fertilizer_per_ha"] = df[C["fert"]] / area
    df["Pesticide_per_ha"] = df[C["pest"]] / area
    return df, []


PALETTE = ["#2e9e5b", "#f4a300", "#17a2b8", "#e4572e", "#8e44ad", "#7cb518", "#d81b60", "#3949ab", "#00897b", "#fb8c00"]
px.defaults.color_discrete_sequence = PALETTE
px.defaults.color_continuous_scale = px.colors.sequential.Greens

st.markdown(
    """<style>
    /* banner */
    .banner{background:linear-gradient(120deg,#145a32 0%,#2e9e5b 55%,#a3d977 100%);padding:22px 28px;border-radius:16px;margin-bottom:16px;box-shadow:0 3px 12px rgba(20,90,50,.35)}
    .banner-title{color:#fff;font-size:2.5rem;font-weight:800;letter-spacing:.04em;line-height:1.1}
    .banner-sub{color:#eafbe7;font-size:1.05rem;margin-top:4px}
    /* tabs */
    .stTabs [data-baseweb="tab-list"]{gap:6px;flex-wrap:wrap;border-bottom:none}
    .stTabs [data-baseweb="tab"]{background:rgba(46,158,91,.14);border-radius:10px;padding:8px 16px;font-weight:600}
    .stTabs [aria-selected="true"]{background:#2e9e5b}
    .stTabs [aria-selected="true"] *{color:#fff !important}
    .stTabs [data-baseweb="tab-highlight"]{background:#f4a300;height:3px}
    .stTabs [data-baseweb="tab-border"]{background:transparent}
    /* metric cards (each column gets its own colour) */
    [data-testid="stMetric"]{background:rgba(46,158,91,.10);border-left:6px solid #2e9e5b;border-radius:10px;padding:12px 16px}
    [data-testid="stHorizontalBlock"]>div:nth-child(2) [data-testid="stMetric"]{border-left-color:#f4a300;background:rgba(244,163,0,.10)}
    [data-testid="stHorizontalBlock"]>div:nth-child(3) [data-testid="stMetric"]{border-left-color:#17a2b8;background:rgba(23,162,184,.10)}
    [data-testid="stHorizontalBlock"]>div:nth-child(4) [data-testid="stMetric"]{border-left-color:#e4572e;background:rgba(228,87,46,.10)}
    /* sidebar + buttons */
    section[data-testid="stSidebar"]{border-right:4px solid #2e9e5b}
    section[data-testid="stSidebar"] h2{color:#2e9e5b}
    .stButton button,.stDownloadButton button{background:#2e9e5b;color:#fff;border:none;border-radius:8px;font-weight:600}
    .stButton button:hover,.stDownloadButton button:hover{background:#1e7d45;color:#fff}
    h3{border-left:6px solid #2e9e5b;padding-left:10px}
    /* KPI tiles */
    .kpi{border-radius:12px;padding:14px 10px;text-align:center;color:#fff;box-shadow:0 2px 8px rgba(0,0,0,.18)}
    .kpi-label{font-size:.8rem;letter-spacing:.1em;font-weight:700;opacity:.92}
    .kpi-value{font-size:2rem;font-weight:800;line-height:1.2}
    .kpi-teal{background:linear-gradient(135deg,#128a9c,#17a2b8)}
    .kpi-red{background:linear-gradient(135deg,#c0392b,#e4572e)}
    .kpi-amber{background:linear-gradient(135deg,#d68910,#f4a300)}
    .kpi-green{background:linear-gradient(135deg,#1e8449,#2e9e5b)}
    .kpi-purple{background:linear-gradient(135deg,#6c3483,#8e44ad)}
    /* hero cards + risk panels */
    .hero{padding:22px 26px;border-radius:14px;color:#fff;margin-bottom:12px;box-shadow:0 2px 10px rgba(0,0,0,.25)}
    .hero-high{background:linear-gradient(135deg,#c0392b,#e74c3c)}
    .hero-low{background:linear-gradient(135deg,#1e8449,#2ecc71)}
    .hero-label{font-size:.85rem;letter-spacing:.12em;font-weight:700;opacity:.9}
    .hero-name{font-size:2.4rem;font-weight:800;line-height:1.2;margin:4px 0}
    .hero-sub{font-size:1rem;opacity:.95}
    .panel{padding:14px 18px;border-radius:12px;border:2px solid;margin-bottom:10px}
    .panel h4{margin:0 0 10px 0}
    .panel-high{border-color:#d9534f;background:rgba(217,83,79,.10)}
    .panel-low{border-color:#2e9e5b;background:rgba(46,158,91,.10)}
    .badge{display:inline-block;color:#fff;padding:4px 12px;margin:3px;border-radius:14px;font-weight:600;font-size:.9rem}
    .badge-high{background:#d9534f}
    .badge-low{background:#2e9e5b}
    </style>""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="banner"><div class="banner-title">🌾 CROPIX AI</div>'
    '<div class="banner-sub">Crop yield variability, risk profiling and prediction</div></div>',
    unsafe_allow_html=True,
)

uploaded = st.sidebar.file_uploader("Upload crop_yield.csv (optional)", type="csv")
uploaded_prof = st.sidebar.file_uploader("Upload Cropix_Risk_Profiles_v2.xlsx (optional)", type="xlsx")
try:
    raw, missing = load_data(uploaded if uploaded else "crop_yield.csv")
except FileNotFoundError:
    st.error("crop_yield.csv not found next to app.py. Upload it from the sidebar.")
    st.stop()
if missing:
    st.error(f"These columns are not in your CSV: {missing}. Edit the COLS mapping at the top of app.py.")
    st.write("Columns found:", list(raw.columns))
    st.stop()

# ------------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------------
st.sidebar.header("Filters")
all_states = sorted(raw[C["state"]].unique())
all_crops = sorted(raw[C["crop"]].unique())
states = st.sidebar.multiselect("State", all_states, default=all_states)
crops = st.sidebar.multiselect("Crop", all_crops, default=all_crops)
y_min, y_max = int(raw[C["year"]].min()), int(raw[C["year"]].max())
yr = st.sidebar.slider("Year range", y_min, y_max, (y_min, y_max))
f = raw[raw[C["state"]].isin(states) & raw[C["crop"]].isin(crops) & raw[C["year"]].between(*yr)]
if C["season"] and C["season"] in raw.columns:
    seasons = sorted(raw[C["season"]].unique())
    sel_seasons = st.sidebar.multiselect("Season", seasons, default=seasons)
    f = f[f[C["season"]].isin(sel_seasons)]

st.sidebar.header("Risk settings")
min_years = st.sidebar.slider("Min. years of data per state-crop", 3, 15, 5)
pct = st.sidebar.slider("High risk if CV is above this percentile", 10, 90, 50)

if f.empty:
    st.warning("No data for these filters.")
    st.stop()


# ------------------------------------------------------------------
# RISK ANALYSIS
# CV is computed over TIME within each (state, crop) and then averaged per
# crop, so differences between states don't get mixed up with year-to-year
# volatility.
# ------------------------------------------------------------------
def risk_tables(d):
    yearly = d.groupby([C["state"], C["crop"], C["year"]])[C["yld"]].mean().reset_index()
    sc = yearly.groupby([C["state"], C["crop"]])[C["yld"]].agg(Mean="mean", Std="std", Years="count").reset_index()
    sc = sc[sc["Years"] >= min_years].copy()
    sc["CV"] = sc["Std"] / sc["Mean"]
    crop_tbl = (
        sc.groupby(C["crop"])
        .agg(Avg_CV=("CV", "mean"), Mean_Yield=("Mean", "mean"), States=(C["state"], "nunique"))
        .reset_index()
        .sort_values("Avg_CV", ascending=False)
    )
    thr = crop_tbl["Avg_CV"].quantile(pct / 100)
    crop_tbl["Risk"] = np.where(crop_tbl["Avg_CV"] > thr, "High Risk", "Low Risk")
    return sc, crop_tbl, thr


sc_tbl, crop_tbl, cv_thr = risk_tables(f)

# ------------------------------------------------------------------
# HEADLINE METRICS
# ------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Records", f"{len(f):,}")
m2.metric("States / Crops", f"{f[C['state']].nunique()} / {f[C['crop']].nunique()}")
m3.metric("Avg yield", f"{f[C['yld']].mean():.2f}")
if not crop_tbl.empty:
    top = crop_tbl.iloc[0]
    m4.metric("Most variable crop", top[C["crop"]], f"CV {top['Avg_CV']:.2f}", delta_color="off")

tab_data, tab_year, tab_state, tab_risk, tab_prof, tab_inputs, tab_model = st.tabs(
    ["📄 Data", "📅 Year-wise", "🗺️ State-wise", "⚠️ Crop risk", "🧾 Risk profiles", "🧪 Rain / Fertilizer / Pesticide", "🤖 LightGBM model"]
)

# ---------------------------- DATA --------------------------------
with tab_data:
    st.dataframe(f, use_container_width=True, height=450)
    st.download_button("Download filtered data", f.to_csv(index=False), "filtered_data.csv")

# ---------------------------- YEAR-WISE ---------------------------
with tab_year:
    yearly = (
        f.groupby(C["year"])
        .agg(
            Avg_Yield=(C["yld"], "mean"),
            Total_Production=(C["prod"], "sum"),
            Avg_Rainfall=(C["rain"], "mean"),
            Total_Fertilizer=(C["fert"], "sum"),
            Total_Pesticide=(C["pest"], "sum"),
            Records=(C["yld"], "count"),
        )
        .reset_index()
    )
    crops_per_year = f.groupby(C["year"])[C["crop"]].nunique().rename("Crops_Grown")
    top_crop_year = (
        f.groupby([C["year"], C["crop"]])[C["yld"]].mean().reset_index()
        .sort_values(C["yld"], ascending=False).drop_duplicates(C["year"])
        .set_index(C["year"])[C["crop"]].rename("Top_Yield_Crop")
    )
    yearly = yearly.set_index(C["year"]).join([crops_per_year, top_crop_year]).reset_index()

    st.subheader("Year summary")
    st.dataframe(yearly, use_container_width=True, hide_index=True)
    st.download_button("Download year summary", yearly.to_csv(index=False), "yearwise.csv")

    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(yearly, x=C["year"], y="Avg_Yield", markers=True, hover_data=["Top_Yield_Crop", "Crops_Grown"], title="Average yield by year"), use_container_width=True)
    c2.plotly_chart(px.bar(yearly, x=C["year"], y="Total_Production", hover_data=["Top_Yield_Crop"], title="Total production by year"), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.line(yearly, x=C["year"], y="Avg_Rainfall", markers=True, title="Average annual rainfall by year"), use_container_width=True)
    c4.plotly_chart(px.line(yearly, x=C["year"], y=["Total_Fertilizer", "Total_Pesticide"], markers=True, title="Fertilizer & pesticide by year"), use_container_width=True)

    # ---- crops inside a chosen year ----
    st.divider()
    st.subheader("🌱 Crops in a selected year")
    years_avail = sorted(f[C["year"]].unique())
    year_pick = st.select_slider("Year", options=years_avail, value=years_avail[-1])
    ycrop = (
        f[f[C["year"]] == year_pick]
        .groupby(C["crop"])
        .agg(
            Avg_Yield=(C["yld"], "mean"),
            Total_Production=(C["prod"], "sum"),
            Total_Area=(C["area"], "sum"),
            Avg_Rainfall=(C["rain"], "mean"),
            Fert_per_ha=("Fertilizer_per_ha", "mean"),
            Pest_per_ha=("Pesticide_per_ha", "mean"),
        )
        .reset_index()
        .sort_values("Avg_Yield", ascending=False)
    )
    st.caption(f"{len(ycrop)} crops recorded in {year_pick}")
    cl, cr = st.columns([1, 1])
    cl.dataframe(ycrop, use_container_width=True, hide_index=True, height=450)
    cr.plotly_chart(
        px.bar(ycrop.head(20), x="Avg_Yield", y=C["crop"], orientation="h", color="Avg_Yield",
               color_continuous_scale="Greens", title=f"Top 20 crops by yield in {year_pick}").update_yaxes(autorange="reversed"),
        use_container_width=True,
    )

    with st.expander("Full year × crop table"):
        yc = f.groupby([C["year"], C["crop"]]).agg(
            Avg_Yield=(C["yld"], "mean"), Total_Production=(C["prod"], "sum"),
            Avg_Rainfall=(C["rain"], "mean"), Total_Fertilizer=(C["fert"], "sum"), Total_Pesticide=(C["pest"], "sum"),
        ).reset_index()
        st.dataframe(yc, use_container_width=True, hide_index=True)
        st.download_button("Download year × crop table", yc.to_csv(index=False), "year_crop.csv")
        piv = yc[yc[C["crop"]].isin(f[C["crop"]].value_counts().head(20).index)].pivot(index=C["crop"], columns=C["year"], values="Avg_Yield")
        st.plotly_chart(px.imshow(piv, aspect="auto", color_continuous_scale="YlGn", title="Yield heatmap: crop × year (top 20 crops by records)"), use_container_width=True)

# ---------------------------- STATE-WISE --------------------------
with tab_state:
    by_state = (
        f.groupby(C["state"])
        .agg(
            Avg_Yield=(C["yld"], "mean"),
            Total_Production=(C["prod"], "sum"),
            Avg_Rainfall=(C["rain"], "mean"),
            Avg_Fert_per_ha=("Fertilizer_per_ha", "mean"),
            Avg_Pest_per_ha=("Pesticide_per_ha", "mean"),
            Records=(C["yld"], "count"),
        )
        .reset_index()
        .sort_values("Avg_Yield", ascending=False)
    )
    st.dataframe(by_state, use_container_width=True)
    st.download_button("Download state-wise table", by_state.to_csv(index=False), "statewise.csv")

    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(by_state, x=C["state"], y="Avg_Yield", color="Avg_Yield", color_continuous_scale="YlGn", title="Average yield by state").update_coloraxes(showscale=False), use_container_width=True)
    c2.plotly_chart(px.bar(by_state, x=C["state"], y="Total_Production", color="Total_Production", color_continuous_scale="Tealgrn", title="Total production by state").update_coloraxes(showscale=False), use_container_width=True)

    st.subheader("Drill into a state")
    s_pick = st.selectbox("State", sorted(f[C["state"]].unique()))
    s_df = f[f[C["state"]] == s_pick]
    s_year = s_df.groupby([C["year"], C["crop"]])[C["yld"]].mean().reset_index()
    st.plotly_chart(px.line(s_year, x=C["year"], y=C["yld"], color=C["crop"], title=f"{s_pick}: yield by crop over time"), use_container_width=True)
    st.dataframe(s_df.groupby(C["year"])[[C["yld"], C["rain"], C["fert"], C["pest"]]].mean().reset_index(), use_container_width=True)

# ---------------------------- RISK --------------------------------
def badges(names, cls):
    return "".join(f'<span class="badge {cls}">{html.escape(str(n))}</span>' for n in names)


with tab_risk:
    if crop_tbl.empty:
        st.info("Not enough years of data per state-crop. Lower the minimum years in the sidebar.")
    else:
        top = crop_tbl.iloc[0]
        stable = crop_tbl.iloc[-1]
        high = crop_tbl[crop_tbl["Risk"] == "High Risk"]
        low = crop_tbl[crop_tbl["Risk"] == "Low Risk"]

        h1, h2 = st.columns(2)
        h1.markdown(
            f"""<div class="hero hero-high"><div class="hero-label">🔥 MOST VARIABLE CROP</div>
            <div class="hero-name">{html.escape(str(top[C['crop']]))}</div>
            <div class="hero-sub">CV {top['Avg_CV']:.2f} &nbsp;·&nbsp; avg yield {top['Mean_Yield']:.2f} &nbsp;·&nbsp; {int(top['States'])} states</div></div>""",
            unsafe_allow_html=True,
        )
        h2.markdown(
            f"""<div class="hero hero-low"><div class="hero-label">🛡️ MOST STABLE CROP</div>
            <div class="hero-name">{html.escape(str(stable[C['crop']]))}</div>
            <div class="hero-sub">CV {stable['Avg_CV']:.2f} &nbsp;·&nbsp; avg yield {stable['Mean_Yield']:.2f} &nbsp;·&nbsp; {int(stable['States'])} states</div></div>""",
            unsafe_allow_html=True,
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("🔴 High-risk crops", len(high))
        k2.metric("🟢 Low-risk crops", len(low))
        k3.metric("Risk cutoff (CV)", f"{cv_thr:.3f}")
        st.caption("CV = std ÷ mean of yield across years, computed per state-crop then averaged per crop. Higher = less predictable.")

        st.markdown("### Risk classification")
        g1, g2 = st.columns(2)
        g1.markdown(f'<div class="panel panel-high"><h4>🔴 High risk ({len(high)})</h4>{badges(high[C["crop"]], "badge-high")}</div>', unsafe_allow_html=True)
        g2.markdown(f'<div class="panel panel-low"><h4>🟢 Low risk ({len(low)})</h4>{badges(low[C["crop"]], "badge-low")}</div>', unsafe_allow_html=True)

        st.markdown("### Ranking table")
        choice = st.radio("Show", ["All", "High Risk", "Low Risk"], horizontal=True)
        tbl = crop_tbl if choice == "All" else crop_tbl[crop_tbl["Risk"] == choice]
        tbl = tbl.copy()
        tbl["Risk"] = tbl["Risk"].map({"High Risk": "🔴 High Risk", "Low Risk": "🟢 Low Risk"})
        st.dataframe(
            tbl, use_container_width=True, hide_index=True,
            column_config={
                "Avg_CV": st.column_config.ProgressColumn("Variability (CV)", format="%.3f", min_value=0.0, max_value=float(crop_tbl["Avg_CV"].max())),
                "Mean_Yield": st.column_config.NumberColumn("Mean yield", format="%.2f"),
                "States": st.column_config.NumberColumn("States"),
            },
        )
        st.download_button("Download risk ranking", crop_tbl.to_csv(index=False), "crop_variability_ranking.csv")

        st.markdown("### Charts")
        color_map = {"High Risk": "#d9534f", "Low Risk": "#2e9e5b"}
        n = st.slider("Crops to show in bar chart", 5, len(crop_tbl), min(15, len(crop_tbl))) if len(crop_tbl) > 5 else len(crop_tbl)
        bar = px.bar(
            crop_tbl.head(n), x="Avg_CV", y=C["crop"], color="Risk", orientation="h", text_auto=".2f",
            color_discrete_map=color_map, title=f"Top {n} most variable crops", height=max(380, 30 * n),
        ).update_yaxes(autorange="reversed")
        bar.add_vline(x=cv_thr, line_dash="dash", line_color="gray", annotation_text="risk cutoff")
        st.plotly_chart(bar, use_container_width=True)

        sc_fig = px.scatter(
            crop_tbl, x="Mean_Yield", y="Avg_CV", color="Risk", size="States", hover_name=C["crop"],
            color_discrete_map=color_map, title="Yield vs variability (bubble size = number of states)", height=480,
        )
        sc_fig.add_hline(y=cv_thr, line_dash="dash", line_color="gray", annotation_text="risk cutoff")
        st.plotly_chart(sc_fig, use_container_width=True)

        st.markdown("### State × crop volatility")
        pivot = sc_tbl[sc_tbl[C["crop"]].isin(crop_tbl.head(15)[C["crop"]])].pivot(index=C["state"], columns=C["crop"], values="CV")
        st.plotly_chart(
            px.imshow(pivot, aspect="auto", color_continuous_scale="RdYlGn_r", title="CV by state and crop (red = riskier)", height=550),
            use_container_width=True,
        )

# ---------------------------- RISK PROFILES (crop x state) --------
PROFILE_FILE = "Cropix_Risk_Profiles_v2.xlsx"
LEVEL_ORDER = ["High", "Medium", "Low"]
LEVEL_COLORS = {"High": "#e4572e", "Medium": "#f4a300", "Low": "#2e9e5b"}
LEVEL_ICON = {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}
P = dict(
    rank="Rank", crop="Crop", state="State", years="Years of Data", yld="Mean Yield",
    trend="Long-term Trend (%/yr)", fluct="Short-term Fluctuation",
    pyd="Prob. Yield Drop >20%", ppd="Prob. Production Drop >20%",
    worst="Worst-case (5th pct vs mean)", rain="Rainfall Dependency",
    fert="Fertilizer per ha", pest="Pesticide per ha",
    fover="Fertilizer Over-use Share", pover="Pesticide Over-use Share",
    tot="Total Input per ha", score="Risk Score", level="Risk Level",
)
PCT_KEYS = ("pyd", "ppd", "worst", "fover", "pover")
METRIC_KEYS = ("trend", "fluct", "pyd", "ppd", "worst", "rain", "fert", "pest", "fover", "pover", "tot")


@st.cache_data
def load_profiles(source):
    d = pd.read_excel(source, sheet_name=0)
    d.columns = d.columns.str.strip()
    return d


def kpi(label, value, cls):
    return f'<div class="kpi {cls}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'


def hero_card(label, row, cls):
    name = html.escape(f"{row[P['crop']]} · {row[P['state']]}")
    return (
        f'<div class="hero {cls}"><div class="hero-label">{label}</div>'
        f'<div class="hero-name" style="font-size:1.9rem">{name}</div>'
        f'<div class="hero-sub">Risk score {row[P["score"]]:.1f} &nbsp;·&nbsp; {row[P["level"]]} &nbsp;·&nbsp; rank #{int(row[P["rank"]])} &nbsp;·&nbsp; {int(row[P["years"]])} years of data</div></div>'
    )


with tab_prof:
    prof, prof_err = None, None
    try:
        prof = load_profiles(uploaded_prof if uploaded_prof else PROFILE_FILE)
    except FileNotFoundError:
        prof_err = f"{PROFILE_FILE} not found next to app.py. Put it there, or upload it from the sidebar."
    except ImportError:
        prof_err = "Reading .xlsx files needs openpyxl. Run: python -m pip install openpyxl"
    if prof is not None:
        missing_p = [c for c in P.values() if c not in prof.columns]
        if missing_p:
            prof_err = f"These columns are missing from the workbook: {missing_p}. Edit the P mapping in app.py."
            prof = None
    if prof_err:
        st.info(prof_err)
    else:
        st.markdown("### 🧾 Crop × State risk profiles")
        st.caption("Filters on this tab apply to this tab only. Levels follow the score in your file: Low below ~33, Medium ~33 to 66, High above ~66.")

        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1.5])
        p_states = fc1.multiselect("State", sorted(prof[P["state"]].unique()), placeholder="All states", key="p_states")
        p_crops = fc2.multiselect("Crop", sorted(prof[P["crop"]].unique()), placeholder="All crops", key="p_crops")
        p_levels = fc3.multiselect("Risk level", LEVEL_ORDER, default=LEVEL_ORDER, key="p_levels")
        yr_lo, yr_hi = int(prof[P["years"]].min()), int(prof[P["years"]].max())
        min_yrs_p = fc4.slider("Min. years of data", yr_lo, yr_hi, yr_lo, key="p_years")

        pf = prof[prof[P["level"]].isin(p_levels) & (prof[P["years"]] >= min_yrs_p)]
        if p_states:
            pf = pf[pf[P["state"]].isin(p_states)]
        if p_crops:
            pf = pf[pf[P["crop"]].isin(p_crops)]
        pf = pf.copy()

        if pf.empty:
            st.warning("No profiles match these filters.")
        else:
            pf["Profile"] = pf[P["crop"]] + " – " + pf[P["state"]]
            counts = {l: int((pf[P["level"]] == l).sum()) for l in LEVEL_ORDER}

            k = st.columns(5)
            k[0].markdown(kpi("PROFILES", f"{len(pf):,}", "kpi-teal"), unsafe_allow_html=True)
            k[1].markdown(kpi("🔴 HIGH RISK", counts["High"], "kpi-red"), unsafe_allow_html=True)
            k[2].markdown(kpi("🟡 MEDIUM RISK", counts["Medium"], "kpi-amber"), unsafe_allow_html=True)
            k[3].markdown(kpi("🟢 LOW RISK", counts["Low"], "kpi-green"), unsafe_allow_html=True)
            k[4].markdown(kpi("AVG RISK SCORE", f"{pf[P['score']].mean():.1f}", "kpi-purple"), unsafe_allow_html=True)

            ranked = pf.sort_values(P["score"], ascending=False)
            h1, h2 = st.columns(2)
            h1.markdown(hero_card("🔥 HIGHEST-RISK PROFILE", ranked.iloc[0], "hero-high"), unsafe_allow_html=True)
            h2.markdown(hero_card("🛡️ SAFEST PROFILE", ranked.iloc[-1], "hero-low"), unsafe_allow_html=True)

            # ---- table ----
            st.markdown("### Full risk profile table")
            disp = pf.sort_values(P["rank"]).drop(columns="Profile").copy()
            for key in PCT_KEYS:
                disp[P[key]] = disp[P[key]] * 100
            disp[P["level"]] = disp[P["level"]].map(LEVEL_ICON)
            NC = st.column_config.NumberColumn
            st.dataframe(
                disp, hide_index=True, use_container_width=True, height=480,
                column_config={
                    P["rank"]: NC("Rank", format="%d"),
                    P["yld"]: NC(P["yld"], format="%.2f"),
                    P["trend"]: NC(P["trend"], format="%.2f"),
                    P["fluct"]: NC(P["fluct"], format="%.3f"),
                    P["pyd"]: NC("Prob. Yield Drop >20% (%)", format="%.1f"),
                    P["ppd"]: NC("Prob. Production Drop >20% (%)", format="%.1f"),
                    P["worst"]: NC("Worst-case, 5th pct vs mean (%)", format="%.1f"),
                    P["rain"]: NC(P["rain"], format="%.3f"),
                    P["fert"]: NC(P["fert"], format="%.1f"),
                    P["pest"]: NC(P["pest"], format="%.3f"),
                    P["fover"]: NC("Fertilizer Over-use Share (%)", format="%.1f"),
                    P["pover"]: NC("Pesticide Over-use Share (%)", format="%.1f"),
                    P["tot"]: NC(P["tot"], format="%.1f"),
                    P["score"]: st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%.1f"),
                },
            )
            st.download_button("Download filtered profiles", pf.drop(columns="Profile").to_csv(index=False), "risk_profiles_filtered.csv")
            st.caption("Mean Yield is in each crop's own unit, so compare it within a crop, not across crops.")

            # ---- overview charts ----
            st.markdown("### Overview")
            o1, o2 = st.columns([1, 2])
            lvl_df = pd.DataFrame({"Level": LEVEL_ORDER, "Profiles": [counts[l] for l in LEVEL_ORDER]})
            o1.plotly_chart(
                px.pie(lvl_df, names="Level", values="Profiles", hole=0.55, color="Level", color_discrete_map=LEVEL_COLORS, title="Risk level mix"),
                use_container_width=True,
            )
            n_top = st.slider("Profiles to show in the ranking", 5, min(30, len(pf)), min(15, len(pf)), key="p_ntop") if len(pf) > 5 else len(pf)
            o2.plotly_chart(
                px.bar(
                    ranked.head(n_top), x=P["score"], y="Profile", color=P["level"], orientation="h", text_auto=".1f",
                    color_discrete_map=LEVEL_COLORS, category_orders={P["level"]: LEVEL_ORDER},
                    title=f"Top {n_top} riskiest crop-state profiles", height=max(380, 28 * n_top),
                ).update_yaxes(autorange="reversed"),
                use_container_width=True,
            )

            a1, a2 = st.columns(2)
            crop_avg = pf.groupby(P["crop"])[P["score"]].mean().sort_values(ascending=False).reset_index()
            state_avg = pf.groupby(P["state"])[P["score"]].mean().sort_values(ascending=False).reset_index()
            a1.plotly_chart(
                px.bar(crop_avg.head(15), x=P["score"], y=P["crop"], orientation="h", color=P["score"], color_continuous_scale="RdYlGn_r",
                       range_color=[0, 100], title="Average risk score by crop (top 15)").update_yaxes(autorange="reversed").update_coloraxes(showscale=False),
                use_container_width=True,
            )
            a2.plotly_chart(
                px.bar(state_avg.head(15), x=P["score"], y=P["state"], orientation="h", color=P["score"], color_continuous_scale="RdYlGn_r",
                       range_color=[0, 100], title="Average risk score by state (top 15)").update_yaxes(autorange="reversed").update_coloraxes(showscale=False),
                use_container_width=True,
            )

            top_crops = crop_avg.head(20)[P["crop"]]
            heat = pf[pf[P["crop"]].isin(top_crops)].pivot_table(index=P["state"], columns=P["crop"], values=P["score"], aggfunc="mean")
            st.plotly_chart(
                px.imshow(heat, aspect="auto", color_continuous_scale="RdYlGn_r", zmin=0, zmax=100, height=600,
                          title="Risk score by state and crop (20 riskiest crops; red = riskier, blank = no profile)"),
                use_container_width=True,
            )

            # ---- what goes with risk ----
            st.markdown("### What goes with higher risk?")
            metric_cols = [P[key] for key in METRIC_KEYS]
            d1, d2 = st.columns(2)
            if len(pf) > 2:
                corr = pf[metric_cols + [P["score"]]].corr()[P["score"]].drop(P["score"]).dropna().sort_values()
                d1.plotly_chart(
                    px.bar(x=corr.values, y=corr.index, orientation="h", color=corr.values, color_continuous_scale="RdYlGn_r", range_color=[-1, 1],
                           labels={"x": "Correlation with risk score", "y": ""}, title="Correlation with risk score").update_coloraxes(showscale=False),
                    use_container_width=True,
                )
            x_p = d2.selectbox("Compare risk score with", metric_cols, key="p_x")
            d2.plotly_chart(
                px.scatter(pf, x=x_p, y=P["score"], color=P["level"], hover_name="Profile", opacity=0.75,
                           color_discrete_map=LEVEL_COLORS, category_orders={P["level"]: LEVEL_ORDER}, title=f"Risk score vs {x_p}"),
                use_container_width=True,
            )
            st.caption("Correlation shows what moves together across these profiles. It doesn't prove that one thing causes another.")

            # ---- drill-down ----
            st.markdown("### Drill down")
            slim_cols = [P["rank"], P["state"], P["crop"], P["score"], P["level"], P["pyd"], P["fluct"]]
            slim_cfg = {
                P["score"]: st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%.1f"),
                P["pyd"]: NC("Prob. Yield Drop >20% (%)", format="%.1f"),
                P["fluct"]: NC(P["fluct"], format="%.3f"),
            }

            def slim(d):
                d = d.sort_values(P["score"], ascending=False)[slim_cols].copy()
                d[P["pyd"]] = d[P["pyd"]] * 100
                d[P["level"]] = d[P["level"]].map(LEVEL_ICON)
                return d

            e1, e2 = st.columns(2)
            c_pick = e1.selectbox("Crop", sorted(pf[P["crop"]].unique()), key="p_crop_pick")
            cd = pf[pf[P["crop"]] == c_pick]
            e1.plotly_chart(
                px.bar(cd.sort_values(P["score"], ascending=False), x=P["state"], y=P["score"], color=P["level"], color_discrete_map=LEVEL_COLORS,
                       category_orders={P["level"]: LEVEL_ORDER}, title=f"{c_pick}: risk score by state"),
                use_container_width=True,
            )
            e1.dataframe(slim(cd).drop(columns=[P["crop"]]), hide_index=True, use_container_width=True, column_config=slim_cfg)

            s_pick_p = e2.selectbox("State ", sorted(pf[P["state"]].unique()), key="p_state_pick")
            sd = pf[pf[P["state"]] == s_pick_p]
            e2.plotly_chart(
                px.bar(sd.sort_values(P["score"], ascending=False), x=P["crop"], y=P["score"], color=P["level"], color_discrete_map=LEVEL_COLORS,
                       category_orders={P["level"]: LEVEL_ORDER}, title=f"{s_pick_p}: risk score by crop"),
                use_container_width=True,
            )
            e2.dataframe(slim(sd).drop(columns=[P["state"]]), hide_index=True, use_container_width=True, column_config=slim_cfg)

# ---------------------------- RAIN / FERTILIZER / PESTICIDE -------
def band_table(d, col, n=5):
    """Average yield inside bands of `col`. Bands are equal-sized groups of records, cut from the CSV values."""
    d = d.dropna(subset=[col, C["yld"]])
    if d[col].nunique() < 3:
        return pd.DataFrame()
    bands = pd.qcut(d[col], q=min(n, d[col].nunique()), duplicates="drop")
    out = d.groupby(bands, observed=True).agg(Avg_Yield=(C["yld"], "mean"), Records=(C["yld"], "count")).reset_index()

    def fmt(v):
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"

    out["Band"] = [f"{fmt(iv.left)} – {fmt(iv.right)}" for iv in out[col]]
    return out[["Band", "Avg_Yield", "Records"]]


def dual_line(d, y_left, y_right, left_name, right_name, title):
    """Two lines over the years, each with its own axis, so a big number doesn't flatten a small one."""
    g = d.groupby(C["year"]).agg(L=(y_left, "mean"), R=(y_right, "mean")).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=g[C["year"]], y=g["L"], name=left_name, mode="lines+markers", line=dict(color="#2e9e5b", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=g[C["year"]], y=g["R"], name=right_name, mode="lines+markers", line=dict(color="#17a2b8", width=3, dash="dot")), secondary_y=True)
    fig.update_yaxes(title_text=left_name, secondary_y=False)
    fig.update_yaxes(title_text=right_name, secondary_y=True)
    fig.update_xaxes(title_text="Year")
    fig.update_layout(title=title, hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    return fig


def share_pie(d, value_col, group_col, title, top=8):
    s = d.groupby(group_col)[value_col].sum().sort_values(ascending=False)
    s = s[s > 0]
    if len(s) > top:
        s = pd.concat([s.iloc[:top], pd.Series({"All other": s.iloc[top:].sum()})])
    pdf = pd.DataFrame({"Name": s.index, "Total": s.values})
    fig = px.pie(pdf, names="Name", values="Total", hole=0.4, title=title)
    fig.update_traces(textinfo="percent", textposition="inside")
    return fig


with tab_inputs:
    st.markdown("### 🌧️ Rainfall, 🧪 fertilizer and 🐛 pesticide")
    st.caption(
        "Every chart here is built from your CSV columns (Annual_Rainfall, Fertilizer, Pesticide, Area, Yield) and follows the sidebar filters. "
        "Fertilizer per ha = Fertilizer ÷ Area, and pesticide per ha = Pesticide ÷ Area."
    )

    # ---- why one crop can flatten every other crop ----
    scale = (
        f.groupby(C["crop"])
        .agg(Median_Yield=(C["yld"], "median"), Median_Production=(C["prod"], "median"), Median_Area=(C["area"], "median"), Records=(C["yld"], "count"))
        .sort_values("Median_Yield", ascending=False)
        .reset_index()
    )
    if len(scale) > 1:
        top_c = scale.iloc[0][C["crop"]]
        rest_median = scale.iloc[1:]["Median_Yield"].median()
        ratio = scale.iloc[0]["Median_Yield"] / rest_median if rest_median else float("nan")
        with st.expander(f"Why does {top_c} dominate yield charts?", expanded=True):
            st.write(
                f"In the data you loaded, **{top_c}** has a median Yield of **{scale.iloc[0]['Median_Yield']:,.2f}**, while the median of the other crops' medians is "
                f"**{rest_median:,.2f}** (about **{ratio:,.0f}×** smaller). On one shared axis that squeezes every other crop into a flat line near zero, "
                f"even though rice, maize and the rest do respond to rainfall. Yield here is Production ÷ Area, so a crop whose Production is counted in a different unit "
                f"(for example, a count of items instead of tonnes) gets a far bigger Yield. Check the Production unit of {top_c} in your source. "
                f"To keep the comparison fair, the charts below show **one crop at a time**."
            )
            st.dataframe(scale.head(8), hide_index=True, use_container_width=True)

    # ---- choose the crop ----
    counts = f[C["crop"]].value_counts()
    crop_sel = st.selectbox("Choose a crop (listed by number of records)", counts.index.tolist(), index=0)
    d = f[f[C["crop"]] == crop_sel]
    st.caption(f"{len(d):,} records for {crop_sel}, years {d[C['year']].min()}–{d[C['year']].max()}, {d[C['state']].nunique()} states.")
    top15 = counts.head(15).index

    def factor_section(col, label):
        st.plotly_chart(dual_line(d, C["yld"], col, "Average yield", f"Average {label}", f"{crop_sel}: yield and {label} by year"), use_container_width=True)
        a, b = st.columns(2)
        bt = band_table(d, col)
        if bt.empty:
            a.info(f"Not enough different {label} values for {crop_sel} to make bands.")
        else:
            a.plotly_chart(
                px.bar(bt, x="Band", y="Avg_Yield", color="Avg_Yield", color_continuous_scale="YlGn", text_auto=".2f", hover_data=["Records"],
                       title=f"{crop_sel}: average yield by {label} band", labels={"Band": f"{label} band (low → high)"}).update_coloraxes(showscale=False),
                use_container_width=True,
            )
        by = f[f[C["crop"]].isin(top15)].groupby(C["crop"])[col].mean().sort_values(ascending=False).reset_index()
        b.plotly_chart(
            px.bar(by, x=col, y=C["crop"], orientation="h", color=col, color_continuous_scale="Teal", text_auto=".2f",
                   title=f"Average {label} for the 15 most-recorded crops", height=460).update_yaxes(autorange="reversed").update_coloraxes(showscale=False),
            use_container_width=True,
        )
        pair = d[[col, C["yld"]]].dropna()
        r = pair[col].corr(pair[C["yld"]]) if len(pair) > 2 else float("nan")
        if pd.notna(r):
            st.caption(f"Correlation between {label} and yield for {crop_sel}: **{r:.2f}** (−1 to +1; near 0 means no straight-line link in this data). This shows a link in the data, not proof of cause.")

    t_rain, t_fert, t_pest, t_share = st.tabs(["🌧️ Rainfall", "🧪 Fertilizer", "🐛 Pesticide", "🥧 Shares"])
    with t_rain:
        factor_section(C["rain"], "annual rainfall")
    with t_fert:
        factor_section("Fertilizer_per_ha", "fertilizer per ha")
    with t_pest:
        factor_section("Pesticide_per_ha", "pesticide per ha")
    with t_share:
        split = st.radio("Split by", ["Crop", "State"], horizontal=True)
        gcol = C["crop"] if split == "Crop" else C["state"]
        p1, p2 = st.columns(2)
        p1.plotly_chart(share_pie(f, C["fert"], gcol, f"Share of total fertilizer, by {split.lower()}"), use_container_width=True)
        p2.plotly_chart(share_pie(f, C["pest"], gcol, f"Share of total pesticide, by {split.lower()}"), use_container_width=True)
        st.caption("Top 8 slices shown; the rest are grouped as 'All other'. Totals come from the CSV rows that match the sidebar filters.")


# ---------------------------- MODEL -------------------------------
@st.cache_resource
def train(data):
    cat_cols = [C["crop"], C["state"]] + ([C["season"]] if C["season"] and C["season"] in data.columns else [])
    num_cols = [C["year"], C["area"], C["rain"], C["fert"], C["pest"]]
    feats = cat_cols + num_cols
    X = data[feats].copy()
    cats = {c: sorted(X[c].unique()) for c in cat_cols}
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    y = data[C["yld"]]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
    mdl.fit(Xtr, ytr)
    pred = mdl.predict(Xva)
    rmse = float(np.sqrt(mean_squared_error(yva, pred)))
    r2 = float(r2_score(yva, pred))
    imp = pd.DataFrame({"Feature": feats, "Importance": mdl.feature_importances_}).sort_values("Importance", ascending=False)
    return mdl, feats, cat_cols, cats, rmse, r2, imp


with tab_model:
    mdl, feats, cat_cols, cats, rmse, r2, imp = train(raw)
    a, b = st.columns(2)
    a.metric("Validation RMSE", f"{rmse:.3f}")
    b.metric("Validation R²", f"{r2:.3f}")
    st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", title="What drives yield predictions").update_yaxes(autorange="reversed"), use_container_width=True)

    st.subheader("Predict yield")
    c1, c2, c3 = st.columns(3)
    inp = {
        C["crop"]: c1.selectbox("Crop ", cats[C["crop"]]),
        C["state"]: c2.selectbox("State ", cats[C["state"]]),
        C["year"]: c3.number_input("Year", y_min, y_max + 10, y_max),
        C["area"]: c1.number_input("Area", 0.0, value=float(raw[C["area"]].median())),
        C["rain"]: c2.number_input("Annual rainfall", 0.0, value=float(raw[C["rain"]].median())),
        C["fert"]: c3.number_input("Fertilizer (total)", 0.0, value=float(raw[C["fert"]].median())),
        C["pest"]: c1.number_input("Pesticide (total)", 0.0, value=float(raw[C["pest"]].median())),
    }
    if C["season"] in cat_cols:
        inp[C["season"]] = c2.selectbox("Season ", cats[C["season"]])
    row = pd.DataFrame([inp])[feats]
    for c in cat_cols:
        row[c] = pd.Categorical(row[c], categories=cats[c])
    if st.button("Predict"):
        st.success(f"Predicted yield: {mdl.predict(row)[0]:.3f}")
