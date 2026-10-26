"""
Customer Segmentation Dashboard
-------------------------------
Run with:  streamlit run app.py

Uses real K-Means clustering (scikit-learn) on a built-in sample dataset
or on a CSV file uploaded by the user. No API keys or databases needed.
"""

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SAMPLE_CSV = Path(__file__).parent / "data" / "customers.csv"
RANDOM_STATE = 42
MIN_ROWS = 10
# Scaling limits: bigger data switches to faster methods automatically
LARGE_DATA_THRESHOLD = 50_000   # use MiniBatchKMeans above this many rows
ELBOW_SAMPLE = 20_000           # rows used for the elbow / silhouette sweep
MAX_PLOT_POINTS = 5_000         # points drawn in the scatter plot
MAX_TABLE_ROWS = 1_000          # rows shown in the data table

st.set_page_config(page_title=" Customer Segmentation", page_icon="🎯", layout="wide")

# ---------------------------------------------------------------- styling ---
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
    .kpi-grid {display: grid; gap: 14px; margin: 0.5rem 0 1.2rem 0;
               grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));}
    .kpi-card {border-radius: 14px; padding: 16px 20px; color: #fff;
               box-shadow: 0 4px 14px rgba(0,0,0,.15);}
    .kpi-label {font-size: .8rem; opacity: .9; text-transform: uppercase; letter-spacing: .06em;}
    .kpi-value {font-size: 1.9rem; font-weight: 700; line-height: 1.25;}
    .kpi-sub {font-size: .75rem; opacity: .85;}
    .c1 {background: linear-gradient(135deg,#6366f1,#8b5cf6);}
    .c2 {background: linear-gradient(135deg,#0ea5e9,#06b6d4);}
    .c3 {background: linear-gradient(135deg,#f59e0b,#ef4444);}
    .c4 {background: linear-gradient(135deg,#10b981,#14b8a6);}
    .c5 {background: linear-gradient(135deg,#ec4899,#f43f5e);}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------ data helpers ---
def generate_sample_data(n=400, seed=RANDOM_STATE):
    """Create a realistic customer dataset made of 5 hidden behaviour groups (vectorised, fast)."""
    rng = np.random.default_rng(seed)
    # share, age (mean, sd), income k$ (mean, sd), spending (mean, sd), purchases/yr
    groups = [
        (0.20, (24, 4), (32, 8), (78, 9), 34),   # young trendsetters
        (0.20, (34, 5), (88, 10), (82, 8), 26),  # affluent big spenders
        (0.15, (44, 7), (90, 10), (20, 8), 8),   # wealthy but cautious
        (0.20, (45, 9), (28, 7), (22, 8), 9),    # budget conscious
        (0.25, (40, 10), (55, 8), (50, 8), 18),  # mainstream
    ]
    idx = rng.choice(len(groups), size=n, p=[g[0] for g in groups])
    age = np.array([g[1] for g in groups])[idx]
    inc = np.array([g[2] for g in groups])[idx]
    spend = np.array([g[3] for g in groups])[idx]
    buys = np.array([g[4] for g in groups])[idx]
    return pd.DataFrame({
        "CustomerID": np.arange(1, n + 1),
        "Gender": rng.choice(["Female", "Male"], size=n),
        "Age": rng.normal(age[:, 0], age[:, 1]).clip(18, 70).round().astype(int),
        "Annual Income (k$)": rng.normal(inc[:, 0], inc[:, 1]).clip(10, 140).round(1),
        "Spending Score (1-100)": rng.normal(spend[:, 0], spend[:, 1]).clip(1, 100).round().astype(int),
        "Purchases per Year": rng.poisson(buys).astype(int),
    })


@st.cache_data(show_spinner=False)
def load_sample(n=400):
    """Use data/customers.csv for the default size, otherwise generate the sample in code."""
    if n == 400 and SAMPLE_CSV.exists():
        try:
            return pd.read_csv(SAMPLE_CSV)
        except Exception:
            pass
    return generate_sample_data(n)


def parse_csv_bytes(raw, encoding):
    """Fast C parser first; fall back to delimiter sniffing (; or tab) if only 1 column came out."""
    df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
    if df.shape[1] == 1:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=None, engine="python")
        except Exception:
            pass
    return df


def read_uploaded_csv(file):
    """Read an uploaded CSV, raising a friendly ValueError on any problem."""
    raw = file.getvalue()
    if not raw.strip():
        raise ValueError("The uploaded file is empty.")
    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = parse_csv_bytes(raw, encoding)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("The file has no readable columns.")
        except (pd.errors.ParserError, csv.Error) as exc:
            raise ValueError(f"Could not parse the file as CSV ({exc}).")
    if df is None:
        raise ValueError("Could not decode the file. Please save it as UTF-8 CSV.")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        raise ValueError("The file contains headers but no data rows.")
    return df


def numeric_candidates(df):
    """Columns that are usable for clustering (numeric, varying, not an ID)."""
    cols = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().mean() < 0.9 or s.nunique(dropna=True) < 2:
            continue  # mostly text or constant
        name = str(col).lower().replace(" ", "").replace("_", "")
        if name.endswith("id") and s.is_unique:
            continue  # identifier, not a feature
        cols.append(col)
    return cols


def find_column(df, keyword):
    for col in df.columns:
        if keyword in str(col).lower() and pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def prepare_features(df, features, strategy):
    """Convert to numbers, handle missing values and return (X, n_filled_or_dropped)."""
    X = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_rows = int(X.isna().any(axis=1).sum())
    if strategy == "Drop rows with missing values":
        X = X.dropna()
    else:
        X = X.fillna(X.median())
    return X, missing_rows


# ---------------------------------------------------------------- ML logic ---
def make_model(n_rows, k, seed):
    """Exact K-Means for normal data, MiniBatchKMeans (much faster) for very large data."""
    if n_rows > LARGE_DATA_THRESHOLD:
        return MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=4096, random_state=seed)
    return KMeans(n_clusters=k, n_init=10, random_state=seed)


@st.cache_data(show_spinner=False)
def run_kmeans(values, k, seed):
    """Scale -> K-Means -> silhouette (on a sample). Returns 1-based labels and the score."""
    scaled = StandardScaler().fit_transform(values)
    labels = make_model(len(values), k, seed).fit(scaled).labels_
    score = None
    if 1 < len(set(labels)) < len(values):
        score = silhouette_score(scaled, labels, sample_size=min(5000, len(values)), random_state=seed)
    return labels + 1, score


@st.cache_data(show_spinner=False)
def compute_elbow(values, max_k, seed):
    """Inertia + silhouette for k = 2..max_k, computed on a sample when data is large."""
    if len(values) > ELBOW_SAMPLE:
        rng = np.random.default_rng(seed)
        values = values[rng.choice(len(values), ELBOW_SAMPLE, replace=False)]
    scaled = StandardScaler().fit_transform(values)
    rows = []
    for k in range(2, max_k + 1):
        model = make_model(len(scaled), k, seed).fit(scaled)
        sil = silhouette_score(scaled, model.labels_, sample_size=min(3000, len(scaled)), random_state=seed)
        rows.append({"k": k, "Inertia": model.inertia_, "Silhouette": sil})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def to_csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8")


def describe_clusters(X, labels):
    """Compare each cluster with the overall average to produce plain-English text."""
    overall_mean = X.mean()
    overall_std = X.std(ddof=0).replace(0, 1)
    means = X.groupby(labels).mean()
    z = (means - overall_mean) / overall_std
    names, texts = {}, {}
    for cl, row in z.iterrows():
        high = row[row >= 0.5].sort_values(ascending=False).head(3)
        low = row[row <= -0.5].sort_values().head(3)
        parts = []
        if len(high):
            parts.append("Higher than average " + ", ".join(f"**{f}**" for f in high.index))
        if len(low):
            parts.append("lower than average " + ", ".join(f"**{f}**" for f in low.index))
        texts[cl] = ("; ".join(parts) + ".") if parts else "Close to the overall average on every feature."
        strongest = row.abs().idxmax()
        names[cl] = f"{'High' if row[strongest] > 0 else 'Low'} {strongest}" if row.abs().max() >= 0.5 else "Average"
    return means, z, names, texts


def kpi_card(label, value, sub, css):
    return (f'<div class="kpi-card {css}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


# ----------------------------------------------------------------- sidebar ---
st.sidebar.title("🎯 Segmentation")
st.sidebar.caption("Configure the data and the model.")

source = st.sidebar.radio("Data source", ["Sample dataset", "Upload CSV"])
df = None
sample_n = 400
if source == "Sample dataset":
    sample_n = st.sidebar.select_slider("Sample size (rows)", [400, 5_000, 50_000, 250_000], value=400,
                                        format_func=lambda v: f"{v:,}",
                                        help="Try larger sizes to see the app scale.")
if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = read_uploaded_csv(uploaded)
        except ValueError as err:
            st.sidebar.error(f"Upload failed: {err}")
        except Exception as err:  # last-resort safety net
            st.sidebar.error(f"Unexpected error while reading the file: {err}")
    if df is None:
        st.sidebar.info("Showing the sample dataset until a valid file is uploaded.")
if df is None:
    df = load_sample(sample_n)

candidates = numeric_candidates(df)
if len(candidates) < 2:
    st.title("Customer Segmentation Dashboard")
    st.error("This dataset needs at least **two numeric columns** with varying values. "
             "Check that your numbers are not stored as text.")
    st.dataframe(df.head(20))
    st.stop()

default_features = [c for c in candidates if find_column(df, "income") == c or find_column(df, "spend") == c]
features = st.sidebar.multiselect("Features to cluster on", candidates,
                                  default=default_features if len(default_features) >= 2 else candidates[:3])
k = st.sidebar.slider("Number of clusters (k)", 2, 10, 5)
missing_strategy = st.sidebar.selectbox("Missing values",
                                        ["Fill with column median", "Drop rows with missing values"])
view_mode = st.sidebar.radio("Scatter plot axes", ["Chosen features", "PCA (2D projection)"],
                             help="PCA squeezes all selected features into 2 axes.")
seed = st.sidebar.number_input("Random seed", 0, 9999, RANDOM_STATE)

# ------------------------------------------------------------------ header ---
st.title("Customer Segmentation Dashboard")
st.caption("K-Means clustering with StandardScaler preprocessing · built with Streamlit, scikit-learn & Plotly")

# KPI cards (computed on the full dataset)
income_col, spend_col = find_column(df, "income"), find_column(df, "spend")
cards = [kpi_card("Total customers", f"{len(df):,}", f"{df.shape[1]} columns", "c1")]
if income_col:
    cards.append(kpi_card("Avg annual income", f"{df[income_col].mean():,.1f}", income_col, "c2"))
if spend_col:
    cards.append(kpi_card("Avg spending score", f"{df[spend_col].mean():,.1f}", spend_col, "c3"))

# ------------------------------------------------------------- validation ---
if len(features) < 2:
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.warning("Select **at least two features** in the sidebar to start clustering.")
    st.stop()

X, missing_rows = prepare_features(df, features, missing_strategy)
if len(X) < MIN_ROWS or len(X) <= k:
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.error(f"Not enough rows: {len(X)} usable row(s) for k = {k}. "
             f"Need at least {max(MIN_ROWS, k + 1)}. Lower k or choose features with fewer gaps.")
    st.stop()

# --------------------------------------------------------------- clustering ---
with st.spinner("Clustering..."):
    labels, sil = run_kmeans(X.to_numpy(dtype="float32"), k, int(seed))

result = df.loc[X.index].copy()
result["Cluster"] = labels
means, z, names, texts = describe_clusters(X, labels)
sizes = pd.Series(labels).value_counts().sort_index()

sil_text = f"{sil:.3f}" if sil is not None else "n/a"
if sil is None:
    sil_note = "Not computable"
elif sil > 0.5:
    sil_note = "Strong structure"
elif sil > 0.25:
    sil_note = "Reasonable structure"
else:
    sil_note = "Weak structure"
cards.append(kpi_card("Clusters", str(k), f"Largest: {sizes.max():,} customers", "c4"))
cards.append(kpi_card("Silhouette score", sil_text, sil_note, "c5"))
st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

if len(X) > LARGE_DATA_THRESHOLD:
    st.caption(f"⚡ Large dataset ({len(X):,} rows): using MiniBatchKMeans; the plot shows a random sample of {MAX_PLOT_POINTS:,} points.")
if missing_rows:
    action = "removed" if missing_strategy.startswith("Drop") else "filled with the median"
    st.info(f"{missing_rows} row(s) had missing values in the selected features and were {action}.")

# Colour map shared by all charts so each cluster keeps its colour
palette = px.colors.qualitative.Bold + px.colors.qualitative.Vivid
cluster_labels = [f"Cluster {c}" for c in sizes.index]
colors = {name: palette[i % len(palette)] for i, name in enumerate(cluster_labels)}

tab_plot, tab_profile, tab_elbow, tab_data = st.tabs(
    ["📊 Cluster Plot", "🧩 Segment Profiles", "📈 Choosing k", "🗂 Data & Download"])

# -------------------------------------------------------------- tab: plot ---
with tab_plot:
    plot_df = X[features].copy()
    plot_df["Cluster"] = pd.Series(labels, index=X.index).map(lambda c: f"Cluster {c}")
    if view_mode.startswith("PCA"):
        comps = PCA(n_components=2, random_state=int(seed)).fit_transform(StandardScaler().fit_transform(X))
        plot_df["PC1"], plot_df["PC2"] = comps[:, 0], comps[:, 1]
        x_col, y_col, hover = "PC1", "PC2", features
    else:
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("X axis", features, index=0)
        y_col = c2.selectbox("Y axis", features, index=1 if features[1] != x_col else 0)
        hover = features
    centers = plot_df.groupby("Cluster")[[x_col, y_col]].mean()  # centroids use ALL rows
    view = plot_df.sample(MAX_PLOT_POINTS, random_state=int(seed)) if len(plot_df) > MAX_PLOT_POINTS else plot_df
    fig = px.scatter(view.sort_values("Cluster"), x=x_col, y=y_col, color="Cluster",
                     color_discrete_map=colors, opacity=0.8, hover_data=hover,
                     render_mode="webgl", title="Customer clusters")
    fig.update_traces(marker=dict(size=8))
    fig.add_trace(go.Scatter(x=centers[x_col], y=centers[y_col], mode="markers+text",
                             text=centers.index.str.replace("Cluster ", "C"), textposition="top center",
                             name="Centroids",
                             marker=dict(symbol="diamond", size=15, color="black", line=dict(width=2, color="white"))))
    fig.update_layout(height=560, legend_title_text="", margin=dict(t=60, b=10))
    st.plotly_chart(fig)

# ----------------------------------------------------------- tab: profiles ---
with tab_profile:
    left, right = st.columns([1, 2])
    with left:
        size_df = pd.DataFrame({"Cluster": cluster_labels, "Customers": sizes.values})
        bar = px.bar(size_df, x="Cluster", y="Customers", color="Cluster", color_discrete_map=colors,
                     text="Customers", title="Cluster sizes")
        bar.update_layout(showlegend=False, height=380)
        st.plotly_chart(bar)
    with right:
        heat = px.imshow(z.round(2).set_axis(cluster_labels, axis=0), color_continuous_scale="RdBu_r",
                         zmin=-2, zmax=2, text_auto=True, aspect="auto",
                         title="Cluster vs. overall average (standard deviations)")
        heat.update_layout(height=380)
        st.plotly_chart(heat)

    st.subheader("Average feature values")
    table = means.round(2).copy()
    table.insert(0, "Customers", sizes)
    table.insert(1, "Share", (sizes / sizes.sum() * 100).round(1).astype(str) + "%")
    table.index = [f"Cluster {c}" for c in table.index]
    st.dataframe(table)

    st.subheader("What defines each segment?")
    cols = st.columns(min(3, k))
    for i, cl in enumerate(sizes.index):
        with cols[i % len(cols)].container(border=True):
            st.markdown(f"#### Cluster {cl} · {names[cl]}")
            st.caption(f"{sizes[cl]:,} customers ({sizes[cl] / sizes.sum():.0%})")
            st.markdown(texts[cl])

# -------------------------------------------------------------- tab: elbow ---
with tab_elbow:
    max_k = min(10, len(X) - 1)
    elbow = compute_elbow(X.to_numpy(dtype="float32"), max_k, int(seed))
    best_k = int(elbow.loc[elbow["Silhouette"].idxmax(), "k"])
    st.markdown(f"Look for the **bend** in the inertia curve, and for a **high** silhouette score. "
                f"The best silhouette is at **k = {best_k}**; you currently use **k = {k}**.")
    e1, e2 = st.columns(2)
    for col, metric, title in [(e1, "Inertia", "Elbow method (inertia)"),
                               (e2, "Silhouette", "Silhouette score by k")]:
        f = px.line(elbow, x="k", y=metric, markers=True, title=title)
        f.add_vline(x=k, line_dash="dash", line_color="#ef4444", annotation_text="selected")
        f.update_traces(line=dict(width=3), marker=dict(size=9))
        f.update_xaxes(dtick=1)
        col.plotly_chart(f)

# --------------------------------------------------------------- tab: data ---
with tab_data:
    st.dataframe(result.head(MAX_TABLE_ROWS))
    if len(result) > MAX_TABLE_ROWS:
        st.caption(f"Showing the first {MAX_TABLE_ROWS:,} rows. The download contains all {len(result):,}.")
    st.download_button("⬇️ Download data with cluster labels",
                       data=to_csv_bytes(result),
                       file_name="customers_with_clusters.csv", mime="text/csv")
    st.caption(f"{len(result):,} rows · {len(features)} features used: {', '.join(features)}")
