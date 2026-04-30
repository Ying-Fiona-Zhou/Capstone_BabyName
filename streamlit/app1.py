from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "notebooks" / "data.csv"
DEPLOY_DATA_PATH = PROJECT_ROOT / "data_source" / "app_data.pkl.gz"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"
END_LETTERS = ("a", "e", "i")
BLUE_PALETTE = ["#f8fbfe", "#f3f8fc", "#e8f1f8", "#d8e6f1", "#bdd2e2", "#6f95b3", "#173954"]


st.set_page_config(
    page_title="Baby Name Prediction Studio",
    page_icon="Baby Name",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    if DEPLOY_DATA_PATH.exists():
        dataset = pd.read_pickle(DEPLOY_DATA_PATH, compression="gzip")
    else:
        dataset = pd.read_csv(DATA_PATH, index_col=0)
    dataset["Name"] = dataset["Name"].astype(str).str.strip()
    dataset["Gender"] = dataset["Gender"].astype(str).str.upper()
    dataset = dataset.sort_values(["Name", "Gender", "Year"]).reset_index(drop=True)
    return dataset


@st.cache_resource
def load_model_assets():
    import joblib

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def normalize_name(name: str) -> str:
    cleaned = "".join(ch for ch in name.strip() if ch.isalpha() or ch in (" ", "-", "'"))
    return cleaned.title()


def count_vowels(name: str) -> int:
    vowels = "aeiou"
    return sum(1 for char in name.lower() if char in vowels)


def ends_with_supported_letters(name: str) -> int:
    return int(name.lower().endswith(END_LETTERS))


def name_history(data: pd.DataFrame, name: str, gender: str) -> pd.DataFrame:
    return data[(data["Name"] == name) & (data["Gender"] == gender)].copy()


def build_prediction_features(
    data: pd.DataFrame,
    name: str,
    gender: str,
    target_year: int,
    is_famous: int,
) -> tuple[pd.DataFrame, dict]:
    history = name_history(data, name, gender)
    history = history[history["Year"] <= target_year]

    gender_slice = data[(data["Gender"] == gender) & (data["Year"] <= target_year)]
    default_ratio = float(gender_slice["Gender_Name_Ratio"].median()) if not gender_slice.empty else 0.5

    if history.empty:
        rolling_ratio = default_ratio
        latest_count = 0
        latest_ratio = 0.0
        latest_year = None
    else:
        rolling_ratio = float(history["Gender_Name_Ratio"].tail(5).mean())
        latest_count = int(history.iloc[-1]["Count"])
        latest_ratio = float(history.iloc[-1]["Name_Ratio"])
        latest_year = int(history.iloc[-1]["Year"])

    feature_row = pd.DataFrame(
        [
            {
                "Year": target_year - 1880,
                "Is_Famous": int(is_famous),
                "Gender_Binary": 1 if gender == "M" else 0,
                "Rolling_Average_Gender_Ratio_5_Years": rolling_ratio,
                "Vowel_Count": count_vowels(name),
                "Ends_With_Specified_Letters": ends_with_supported_letters(name),
            }
        ]
    )

    summary = {
        "latest_count": latest_count,
        "latest_ratio": latest_ratio,
        "latest_year": latest_year,
        "rolling_ratio": rolling_ratio,
        "history_rows": len(history),
    }
    return feature_row, summary


def render_trend_plot(history: pd.DataFrame, metric: str, title_name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BLUE_PALETTE[0])
    ax.set_facecolor(BLUE_PALETTE[0])
    ax.plot(history["Year"], history[metric], color=BLUE_PALETTE[6], linewidth=2.6)
    ax.scatter(history["Year"], history[metric], color=BLUE_PALETTE[6], s=20)
    ax.set_title(f"{title_name} {metric.replace('_', ' ')} Over Time", fontsize=15, color="#173954")
    ax.set_xlabel("Year")
    ax.set_ylabel(metric.replace("_", " "))
    ax.tick_params(colors="#35556f")
    for spine in ax.spines.values():
        spine.set_color(BLUE_PALETTE[2])
    ax.grid(alpha=0.28, color=BLUE_PALETTE[3])
    st.pyplot(fig)
    plt.close(fig)


def render_dataframe(df: pd.DataFrame) -> None:
    styled = (
        df.style.hide(axis="index")
        .set_table_styles(
            [
                {
                    "selector": "table",
                    "props": [
                        ("background-color", BLUE_PALETTE[0]),
                        ("color", "#24455f"),
                        ("border-collapse", "collapse"),
                        ("width", "100%"),
                    ],
                },
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", BLUE_PALETTE[3]),
                        ("color", "#173954"),
                        ("font-weight", "700"),
                        ("border", f"1px solid {BLUE_PALETTE[4]}"),
                        ("padding", "10px 14px"),
                    ],
                },
                {
                    "selector": "tbody td",
                    "props": [
                        ("border", f"1px solid {BLUE_PALETTE[2]}"),
                        ("color", "#24455f"),
                        ("padding", "10px 14px"),
                    ],
                },
                {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", BLUE_PALETTE[0])]},
                {"selector": "tbody tr:nth-child(even)", "props": [("background-color", BLUE_PALETTE[1])]},
            ]
        )
    )
    st.table(styled)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --blue-1: #f8fbfe;
            --blue-2: #f3f8fc;
            --blue-3: #e8f1f8;
            --blue-4: #d8e6f1;
            --blue-5: #bdd2e2;
            --blue-6: #6f95b3;
            --blue-7: #173954;
            --page-bg: var(--blue-2);
            --panel-bg: rgba(251, 253, 255, 0.98);
            --panel-border: rgba(23, 57, 84, 0.10);
            --heading: #173954;
            --body-text: #24455f;
            --muted-text: #59728a;
            --accent: #6f95b3;
            --accent-soft: #e8f1f8;
            --accent-deep: #173954;
            --mint-soft: #f3f8fc;
        }
        .stApp {
            background:
              radial-gradient(circle at top left, rgba(189, 210, 226, 0.20), transparent 26%),
              radial-gradient(circle at top right, rgba(111, 149, 179, 0.08), transparent 20%),
              linear-gradient(180deg, var(--blue-1) 0%, var(--page-bg) 100%);
            color: var(--body-text);
        }
        .block-container {
            padding-top: 2.25rem;
            padding-bottom: 3.5rem;
            max-width: 1200px;
        }
        body, p, li, label, span, div {
            color: var(--body-text);
        }
        h1, h2, h3 {
            color: var(--heading);
            letter-spacing: -0.02em;
            font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }
        .hero-card, .soft-card, .section-card {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 28px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 12px 30px rgba(47, 95, 132, 0.06);
            backdrop-filter: blur(8px);
        }
        .hero-card {
            padding: 1.8rem;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.9fr);
            gap: 1.25rem;
            align-items: stretch;
        }
        .hero-title {
            font-size: clamp(2.2rem, 5vw, 4rem);
            line-height: 1.02;
            margin: 0.35rem 0 0.9rem;
            font-weight: 700;
        }
        .hero-copy {
            max-width: 44rem;
            font-size: 1.02rem;
            line-height: 1.75;
            color: var(--muted-text);
        }
        .hero-stat-block {
            background: linear-gradient(180deg, #fbfdff 0%, var(--blue-3) 100%);
            border: 1px solid rgba(23, 57, 84, 0.10);
            border-radius: 24px;
            padding: 1.25rem;
        }
        .hero-stat-kicker {
            color: var(--accent-deep);
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 700;
        }
        .hero-stat-value {
            font-size: 2.6rem;
            line-height: 1;
            margin: 0.65rem 0 0.4rem;
            color: var(--heading);
            font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            font-weight: 700;
        }
        .hero-stat-note {
            color: var(--muted-text);
            line-height: 1.65;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 1.15rem;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.65rem 0.9rem;
            border-radius: 999px;
            background: rgba(240, 248, 254, 0.95);
            border: 1px solid rgba(111, 149, 179, 0.18);
            color: var(--heading);
            font-weight: 600;
            font-size: 0.92rem;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.76rem;
            color: var(--accent);
            font-weight: 700;
        }
        .note {
            color: var(--muted-text);
            font-size: 1rem;
            line-height: 1.7;
        }
        .section-intro {
            margin: 0.1rem 0 1rem;
            color: var(--muted-text);
            font-size: 1rem;
            line-height: 1.7;
            max-width: 48rem;
        }
        .subtle-label {
            color: var(--accent-deep);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.74rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        div[data-testid="stMetric"] {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            padding: 1rem;
            border-radius: 18px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
        }
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricValue"] {
            color: var(--heading);
        }
        div[data-testid="stMetricDelta"] svg,
        div[data-testid="stMetricDelta"] {
            color: var(--accent-deep);
        }
        div[data-baseweb="tab-list"] {
            gap: 0.5rem;
            margin: 1.1rem 0 1.25rem;
        }
        button[data-baseweb="tab"] {
            background: rgba(247, 252, 255, 0.92);
            border: 1px solid rgba(111, 149, 179, 0.15);
            border-radius: 999px;
            padding: 0.66rem 1rem;
        }
        button[data-baseweb="tab"] > div {
            color: var(--muted-text);
            font-weight: 600;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: var(--accent-soft);
            border-color: rgba(47, 95, 132, 0.16);
        }
        button[data-baseweb="tab"][aria-selected="true"] > div {
            color: var(--heading);
        }
        button[data-baseweb="tab"]::after {
            background: transparent !important;
        }
        button[data-baseweb="tab"][aria-selected="true"]::after {
            background: var(--accent-deep) !important;
            height: 0.18rem !important;
            border-radius: 999px !important;
        }
        div[data-testid="stTextInputRootElement"] input,
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] > div,
        div[data-testid="stTextArea"] textarea {
            background: var(--blue-1);
            color: var(--body-text);
            border: 1px solid rgba(111, 149, 179, 0.22);
            border-radius: 14px;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] {
            padding-top: 0.3rem;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
            background: var(--blue-4) !important;
        }
        div[data-testid="stSlider"] [role="slider"] {
            background: var(--accent-deep) !important;
            border: 2px solid #ffffff !important;
            box-shadow: 0 0 0 3px rgba(111, 149, 179, 0.18) !important;
        }
        div[data-testid="stSliderTickBarMin"] {
            background: linear-gradient(90deg, var(--blue-6), var(--accent-deep)) !important;
        }
        div[data-testid="stSliderTickBarMax"] {
            background: var(--blue-3) !important;
        }
        button[kind="secondary"] {
            background: var(--blue-1) !important;
            color: var(--heading) !important;
            border: 1px solid rgba(127, 178, 214, 0.24) !important;
        }
        div[data-baseweb="radio"] > div label[data-baseweb="radio"] > div:first-child {
            border-color: var(--accent-deep) !important;
        }
        div[data-baseweb="radio"] input:checked + div {
            background: var(--accent-deep) !important;
            border-color: var(--accent-deep) !important;
        }
        div[data-baseweb="checkbox"] label > div:first-child {
            border-color: rgba(47, 95, 132, 0.28) !important;
            background: var(--blue-1) !important;
        }
        div[data-baseweb="checkbox"] input:checked + div {
            background: var(--accent-deep) !important;
            border-color: var(--accent-deep) !important;
        }
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li,
        div[data-testid="stCaptionContainer"] {
            color: var(--body-text);
        }
        div[data-testid="stMarkdownContainer"] ul {
            padding-left: 1.1rem;
        }
        div[data-testid="stTable"] {
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            overflow: hidden;
            background: var(--blue-1);
        }
        div[data-testid="stTable"] table {
            width: 100%;
            color: var(--body-text);
            background: var(--blue-1);
        }
        div[data-testid="stTable"] thead tr th {
            background: var(--blue-3);
            color: var(--heading);
        }
        div[data-testid="stTable"] tbody tr:nth-child(odd) {
            background: var(--blue-1);
        }
        div[data-testid="stTable"] tbody tr:nth-child(even) {
            background: var(--blue-2);
        }
        div[data-testid="stAlertContainer"] {
            border-radius: 16px;
        }
        div[data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] p {
            color: var(--heading);
        }
        div[data-testid="stAlertContainer"]:has([data-testid="stNotificationContentSuccess"]) {
            background: #eef5fa !important;
            border: 1px solid #d3e1ec !important;
        }
        div[data-testid="stAlertContainer"]:has([data-testid="stNotificationContentWarning"]) {
            background: #f3f8fc !important;
            border: 1px solid #d8e6f1 !important;
        }
        div[data-testid="stAlertContainer"]:has([data-testid="stNotificationContentError"]) {
            background: #f4f8fb !important;
            border: 1px solid #d9e3eb !important;
        }
        div[data-testid="stAlertContainer"] svg {
            color: var(--accent-deep) !important;
            fill: var(--accent-deep) !important;
        }
        div[data-testid="stPlotlyChart"],
        div[data-testid="stImage"] {
            border-radius: 22px;
            overflow: hidden;
        }
        @media (max-width: 900px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }
            .hero-title {
                font-size: 2.45rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()
data = load_data()

st.markdown(
    f"""
    <div class="hero-card">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">Portfolio Project</div>
          <h1 class="hero-title">Baby Name<br/>Prediction Studio</h1>
          <p class="hero-copy">
            Explore how names rise and fall over time, compare historical momentum,
            and test a model that estimates whether a name is likely to break into
            the Top 100. The goal is to present the capstone like a thoughtful data
            product instead of a notebook dump.
          </p>
          <div class="pill-row">
            <div class="pill">Historical trend explorer</div>
            <div class="pill">Top-100 prediction workflow</div>
            <div class="pill">Portfolio-ready storytelling</div>
          </div>
        </div>
        <div class="hero-stat-block">
          <div class="hero-stat-kicker">Why It Stands Out</div>
          <div class="hero-stat-value">{data['Year'].min()}-{data['Year'].max()}</div>
          <p class="hero-stat-note">
            A century-spanning naming dataset becomes something interactive:
            part research summary, part model demo, part polished case study.
          </p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, trends_tab, prediction_tab, insights_tab = st.tabs(
    ["Overview", "Trend Explorer", "Prediction Studio", "Project Insights"]
)


with overview_tab:
    st.markdown(
        """
        <div class="section-intro">
          Start here for the project framing: what data is covered, how large the
          dataset is, and what kind of story this app is trying to tell.
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows in dataset", f"{len(data):,}")
    col2.metric("Unique names", f"{data['Name'].nunique():,}")
    col3.metric("Years covered", f"{data['Year'].min()}-{data['Year'].max()}")
    col4.metric("Latest year", f"{data['Year'].max()}")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown(
            """
            <div class="soft-card">
              <div class="eyebrow">Problem Focus</div>
              <h3>Can historical naming patterns help predict future popularity?</h3>
              <p class="note">
                The project combines historical U.S. baby name records with feature
                engineering and classification modeling. Instead of leaving the work
                in notebooks only, this app turns the strongest outputs into an
                interactive experience: trend exploration and popularity prediction.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        top_2023 = data[data["Year"] == data["Year"].max()].sort_values("Count", ascending=False).head(8)
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown("#### Sample of Recent Top Names")
        render_dataframe(top_2023[["Name", "Gender", "Count"]].reset_index(drop=True))
        st.markdown("</div>", unsafe_allow_html=True)


with trends_tab:
    st.subheader("Trend Explorer")
    st.markdown(
        """
        <div class="section-intro">
          Compare one or more names across time using raw counts or ratio-based
          popularity signals. This is the quickest way to see whether a name is
          stable, surging, fading, or split strongly by gender.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="subtle-label">Explore Inputs</div>', unsafe_allow_html=True)

    controls_left, controls_right = st.columns([1.2, 1])
    with controls_left:
        names_input = st.text_input("Names to compare", "Olivia, Liam")
    with controls_right:
        metric = st.selectbox("Metric", ["Count", "Name_Ratio", "Gender_Name_Ratio"])

    years_left, years_right = st.columns(2)
    with years_left:
        start_year = st.slider("Start year", min_value=int(data["Year"].min()), max_value=int(data["Year"].max() - 1), value=2000)
    with years_right:
        end_year = st.slider("End year", min_value=int(data["Year"].min() + 1), max_value=int(data["Year"].max()), value=int(data["Year"].max()))

    st.markdown("</div>", unsafe_allow_html=True)

    selected_names = [normalize_name(name) for name in names_input.split(",") if name.strip()]
    trend_data = data[(data["Year"] >= start_year) & (data["Year"] <= end_year) & (data["Name"].isin(selected_names))]

    if trend_data.empty:
        st.warning("No matching records were found for that name and year range.")
    else:
        stat1, stat2, stat3 = st.columns(3)
        stat1.metric("Names compared", len(selected_names))
        stat2.metric("Filtered rows", f"{len(trend_data):,}")
        stat3.metric("Year window", f"{start_year}-{end_year}")

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor(BLUE_PALETTE[0])
        ax.set_facecolor(BLUE_PALETTE[0])
        palette = {"F": BLUE_PALETTE[5], "M": BLUE_PALETTE[6]}

        for name in selected_names:
            name_slice = trend_data[trend_data["Name"] == name]
            for gender in ("F", "M"):
                gender_slice = name_slice[name_slice["Gender"] == gender]
                if gender_slice.empty:
                    continue
                label = f"{name} ({gender})"
                ax.plot(gender_slice["Year"], gender_slice[metric], label=label, linewidth=2.4, color=palette[gender], alpha=0.9)

        ax.set_title(f"{metric.replace('_', ' ')} from {start_year} to {end_year}", fontsize=16, color=BLUE_PALETTE[6])
        ax.set_xlabel("Year")
        ax.set_ylabel(metric.replace("_", " "))
        ax.tick_params(colors="#35556f")
        for spine in ax.spines.values():
            spine.set_color(BLUE_PALETTE[2])
        ax.grid(alpha=0.24, color=BLUE_PALETTE[3])
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        summary = (
            trend_data.groupby(["Name", "Gender"], as_index=False)
            .agg(
                latest_year=("Year", "max"),
                peak_count=("Count", "max"),
                average_ratio=("Name_Ratio", "mean"),
            )
            .sort_values(["peak_count"], ascending=False)
        )
        render_dataframe(summary)


with prediction_tab:
    st.subheader("Prediction Studio")
    st.markdown(
        """
        <div class="section-intro">
          Build a model-ready feature row from a name, gender lens, and target year.
          The app then combines historical context with the saved classifier to
          estimate Top-100 potential.
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_col, result_col = st.columns([1, 1.05])

    with input_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="subtle-label">Prediction Inputs</div>', unsafe_allow_html=True)
        name = normalize_name(st.text_input("Baby name", "Olivia"))
        gender = st.radio("Gender lens", ["F", "M"], horizontal=True, format_func=lambda value: "Female" if value == "F" else "Male")
        target_year = st.slider("Prediction year", min_value=1880, max_value=2035, value=2028)
        is_famous = st.toggle(
            "This name has strong famous-person or pop-culture association",
            value=False,
            help="This maps to the Is_Famous feature used in the original model pipeline.",
        )

        features, summary = build_prediction_features(data, name, gender, target_year, int(is_famous))

        st.markdown("#### Auto-generated feature summary")
        feature_preview = pd.DataFrame(
            [
                {
                    "Name": name,
                    "Gender": gender,
                    "Target Year": target_year,
                    "Vowel Count": int(features.loc[0, "Vowel_Count"]),
                    "Ends with a/e/i": int(features.loc[0, "Ends_With_Specified_Letters"]),
                    "Rolling gender ratio": round(float(features.loc[0, "Rolling_Average_Gender_Ratio_5_Years"]), 3),
                }
            ]
        )
        render_dataframe(feature_preview)
        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        st.markdown(
            """
            <div class="soft-card">
              <div class="eyebrow">Model Output</div>
            """,
            unsafe_allow_html=True,
        )

        history = name_history(data, name, gender)
        if history.empty:
            st.info("This exact name/gender pair does not appear in the historical dataset yet, so the app is extrapolating from the selected gender baseline.")
        else:
            latest = history.iloc[-1]
            st.metric("Latest observed count", f"{int(latest['Count']):,}", delta=f"{int(latest['Year'])}")
            st.metric("Latest name ratio", f"{float(latest['Name_Ratio']):.2f} per 1,000")

        try:
            model, preprocessor = load_model_assets()
            transformed = preprocessor.transform(features)
            prediction = int(model.predict(transformed)[0])
            probability = float(model.predict_proba(transformed)[0][1])

            if prediction == 1:
                st.success(f"Estimated result: **{name}** has a strong chance of landing in the Top 100.")
            else:
                st.warning(f"Estimated result: **{name}** is less likely to land in the Top 100 with the current feature profile.")

            st.metric("Top 100 probability", f"{probability:.1%}")
        except Exception as exc:
            st.error("The model dependencies are missing or incompatible in the current environment.")
            if "sklearn" in str(exc).lower():
                st.info(
                    "Install `scikit-learn` and `joblib`, then redeploy. "
                    "For Streamlit Community Cloud, keep these packages in `requirements.txt` "
                    "and select Python 3.9 or 3.10 in Advanced settings."
                )
            st.caption(f"Debug detail: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)

        if not history.empty:
            render_metric = st.selectbox(
                "Historical metric for this name",
                ["Count", "Name_Ratio", "Gender_Name_Ratio"],
                key="single_name_metric",
            )
            render_trend_plot(history, render_metric, name)

        st.markdown("#### Why this prediction is reasonable")
        st.markdown(
            f"""
            - Historical records found for this name/gender pair: **{summary['history_rows']:,}**
            - Latest observed year in training data: **{summary['latest_year'] or 'No direct history'}**
            - Five-year gender-ratio baseline used by the model: **{summary['rolling_ratio']:.3f}**
            - Famous-name signal: **{'On' if is_famous else 'Off'}**
            """
        )


with insights_tab:
    st.subheader("What makes this project portfolio-ready")
    st.markdown(
        """
        <div class="section-intro">
          This section frames the app the way a recruiter, teammate, or portfolio
          reviewer would read it: not just as analysis, but as a productized ML
          case study with a clear audience and purpose.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(
            """
            <div class="soft-card">
              <div class="eyebrow">Strengths</div>
              <ul>
                <li>The topic is easy for non-technical audiences to understand.</li>
                <li>The project combines exploratory analysis, modeling, and a user-facing app.</li>
                <li>Trend analysis and prediction live together in one experience.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_right:
        st.markdown(
            """
            <div class="soft-card">
              <div class="eyebrow">Next Improvements</div>
              <ul>
                <li>Add curated screenshots and model-performance notes to the README.</li>
                <li>Replace manual famous-name labeling with a transparent lookup source.</li>
                <li>Deploy the app and link it from a future portfolio homepage.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Suggested portfolio framing")
    st.write(
        "This project works best when presented as a machine learning product story: "
        "historical naming data, feature design, interpretable prediction, and a simple "
        "interface for exploration."
    )
