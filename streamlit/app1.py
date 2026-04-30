from pathlib import Path
import platform
import base64

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "notebooks" / "data.csv"
DEPLOY_DATA_PATH = PROJECT_ROOT / "data_source" / "app_data.pkl.gz"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"
HERO_ILLUSTRATION_PATH = PROJECT_ROOT / "streamlit" / "assets" / "baby-hero-modern.svg"
PAGE_BG_IMAGE_CANDIDATES = [
    PROJECT_ROOT / "streamlit" / "assets" / "baby-hero-bg.png",
    PROJECT_ROOT / "streamlit" / "assets" / "baby-hero-bg.jpg",
    PROJECT_ROOT / "streamlit" / "assets" / "baby-hero-bg.webp",
    PROJECT_ROOT / "streamlit" / "assets" / "baby-hero-modern.svg",
]
END_LETTERS = ("a", "e", "i")
BLUE_PALETTE = ["#f8fbfe", "#f3f8fc", "#e8f1f8", "#d8e6f1", "#bdd2e2", "#6f95b3", "#173954"]
COMPARISON_PALETTE = [
    "#d84f68",  # rose red
    "#2f7a63",  # jade green
    "#7a5bc6",  # violet
    "#d9822b",  # amber orange
    "#2f6fb0",  # cobalt blue
    "#b34f9d",  # magenta plum
]
GENDER_STYLES = {
    "F": {"linestyle": "-", "marker": "o"},
    "M": {"linestyle": "--", "marker": "s"},
}


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


def darken_color(color: str, factor: float = 0.68) -> str:
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex((r * factor, g * factor, b * factor))


def runtime_package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    packages = ("python", "streamlit", "pandas", "numpy", "scikit-learn", "joblib")
    versions: dict[str, str] = {"python": platform.python_version()}
    for package_name in packages[1:]:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = "not installed"
    return versions


def to_data_uri(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    suffix = image_path.suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/svg+xml"
    return f"data:{mime_type};base64,{encoded}"


def resolve_page_background_data_uri() -> str | None:
    for candidate in PAGE_BG_IMAGE_CANDIDATES:
        if candidate.exists():
            return to_data_uri(candidate)
    return None


def metric_display_label(metric: str) -> str:
    labels = {
        "Count": "Count",
        "Name_Ratio": "Name Ratio (per 1,000)",
        "Gender_Name_Ratio": "Gender Name Ratio (per 1,000)",
    }
    return labels.get(metric, metric.replace("_", " "))


def enable_tab_anchor_navigation() -> None:
    components.html(
        """
        <script>
        (function () {
          function clickTrendTabFromHash() {
            try {
              const hash = window.parent.location.hash || "";
              if (hash !== "#trend-explorer") return;
              const tabButtons = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
              for (const btn of tabButtons) {
                const text = (btn.innerText || "").trim().toLowerCase();
                if (text === "trend explorer") {
                  btn.click();
                  break;
                }
              }
            } catch (e) {
              // no-op: DOM access can fail in some hosted contexts
            }
          }

          clickTrendTabFromHash();
          window.parent.addEventListener("hashchange", clickTrendTabFromHash);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


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
    axis_label = metric_display_label(metric)
    ax.set_title(f"{title_name} {axis_label} Over Time", fontsize=15, color="#173954")
    ax.set_xlabel("Year")
    ax.set_ylabel(axis_label)
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


def inject_styles(page_background_data_uri: str | None = None) -> None:
    page_background_css = ""
    if page_background_data_uri:
        page_background_css = f"""
        .stApp {{
            background-image:
              linear-gradient(180deg, rgba(246, 251, 255, 0.78) 0%, rgba(237, 244, 252, 0.82) 100%),
              url("{page_background_data_uri}") !important;
            background-size: cover !important;
            background-repeat: no-repeat !important;
            background-position: center center !important;
            background-attachment: fixed !important;
        }}
        """
    st.markdown(
        """
        <style>
        :root {
            --blue-1: #f9fbfd;
            --blue-2: #f1f6fc;
            --blue-3: #e6eff9;
            --blue-4: #d5e4f4;
            --blue-5: #b6cdee;
            --blue-6: #4b7fbe;
            --blue-7: #0f2740;
            --page-bg: #edf4fc;
            --panel-bg: rgba(255, 255, 255, 0.92);
            --panel-border: rgba(28, 70, 120, 0.14);
            --heading: #0f2740;
            --body-text: #26435f;
            --muted-text: #4f6f91;
            --accent: #3178d3;
            --accent-soft: #e0edff;
            --accent-deep: #1d5fb6;
            --mint-soft: #eef6ff;
            --input-bg: #f7fbff;
            --input-bg-focus: #ffffff;
            --input-border: rgba(45, 108, 186, 0.26);
            --input-border-focus: rgba(32, 94, 170, 0.55);
        }
        .stApp {
            background:
              radial-gradient(circle at top left, rgba(81, 145, 220, 0.22), transparent 34%),
              radial-gradient(circle at top right, rgba(68, 124, 205, 0.12), transparent 22%),
              radial-gradient(circle at 50% -5%, rgba(68, 124, 205, 0.10), transparent 42%),
              linear-gradient(180deg, #f7fbff 0%, var(--page-bg) 100%);
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
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }
        .hero-card, .soft-card, .section-card {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 20px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 14px 34px rgba(29, 85, 158, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.78);
            backdrop-filter: blur(16px) saturate(130%);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }
        .hero-card:hover, .soft-card:hover, .section-card:hover {
            transform: translateY(-2px);
            border-color: rgba(40, 109, 188, 0.40);
            box-shadow: 0 20px 38px rgba(29, 85, 158, 0.17), inset 0 1px 0 rgba(255, 255, 255, 0.84);
        }
        .hero-card {
            padding: 1.8rem;
            position: relative;
        }
        .hero-card::after {
            content: "";
            position: absolute;
            right: -80px;
            top: -90px;
            width: 320px;
            height: 320px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(84, 133, 187, 0.20) 0%, rgba(84, 133, 187, 0) 70%);
            pointer-events: none;
            z-index: 0;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.9fr);
            gap: 1.25rem;
            align-items: stretch;
        }
        .hero-title {
            font-size: clamp(2.5rem, 6.6vw, 4.8rem);
            line-height: 0.98;
            margin: 0.2rem 0 0.7rem;
            font-weight: 680;
            letter-spacing: -0.03em;
        }
        .hero-copy {
            max-width: 38rem;
            font-size: 1.05rem;
            line-height: 1.6;
            color: var(--muted-text);
        }
        .hero-actions {
            margin-top: 1.05rem;
            display: flex;
            gap: 0.7rem;
            align-items: center;
            flex-wrap: wrap;
        }
        .hero-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            padding: 0.62rem 1.15rem;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.92rem;
            letter-spacing: 0.01em;
            transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
        }
        .hero-btn-primary {
            color: #ffffff !important;
            background: linear-gradient(135deg, #2e79d8 0%, #1d5fb6 100%);
            border: 1px solid rgba(29, 95, 182, 0.95);
            box-shadow: 0 12px 28px rgba(29, 95, 182, 0.34);
        }
        .hero-btn-primary:hover {
            opacity: 0.98;
            transform: translateY(-2px);
            box-shadow: 0 16px 30px rgba(29, 95, 182, 0.42);
        }
        .hero-stat-block {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.9) 0%, rgba(224, 236, 249, 0.78) 100%);
            border: 1px solid rgba(45, 108, 186, 0.22);
            border-radius: 18px;
            padding: 1rem 1.1rem;
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
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
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
            border-radius: 12px;
            background: rgba(247, 250, 253, 0.92);
            border: 1px solid rgba(95, 127, 157, 0.18);
            color: var(--heading);
            font-weight: 560;
            font-size: 0.88rem;
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
            position: relative;
            padding-left: 0.85rem;
        }
        .section-intro::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0.25rem;
            bottom: 0.25rem;
            width: 2px;
            border-radius: 999px;
            background: linear-gradient(180deg, rgba(95, 127, 157, 0.78), rgba(95, 127, 157, 0.20));
        }
        .subtle-label {
            color: var(--accent-deep);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.74rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        div[data-testid="stWidgetLabel"] p {
            color: var(--heading) !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
        }
        div[data-testid="stTextInputRootElement"],
        div[data-testid="stNumberInput"],
        div[data-testid="stSelectbox"],
        div[data-testid="stTextArea"] {
            border-radius: 16px;
        }
        div[data-testid="stMetric"] {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            padding: 1rem;
            border-radius: 18px;
            box-shadow: 0 6px 16px rgba(29, 85, 158, 0.10), inset 0 1px 0 rgba(255, 255, 255, 0.88);
            transition: border-color 0.2s ease, transform 0.2s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-color: rgba(49, 120, 211, 0.45);
            transform: translateY(-2px);
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
            border: 1px solid rgba(49, 120, 211, 0.24);
            border-radius: 999px;
            padding: 0.66rem 1rem;
            transition: border-color 0.2s ease, background 0.2s ease, transform 0.2s ease;
        }
        button[data-baseweb="tab"]:hover {
            border-color: rgba(49, 120, 211, 0.55);
            background: rgba(224, 239, 255, 0.98);
            transform: translateY(-2px);
        }
        button[data-baseweb="tab"] > div {
            color: var(--muted-text);
            font-weight: 600;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, #e3f0ff 0%, #d4e7ff 100%);
            border-color: rgba(49, 120, 211, 0.56);
        }
        button[data-baseweb="tab"][aria-selected="true"] > div {
            color: var(--heading);
        }
        button[data-baseweb="tab"]::after {
            background: transparent !important;
        }
        button[data-baseweb="tab"][aria-selected="true"]::after {
            background: var(--accent-deep) !important;
            height: 0.20rem !important;
            border-radius: 999px !important;
        }
        div[data-testid="stTextInputRootElement"] input,
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] > div,
        div[data-testid="stTextArea"] textarea {
            background: var(--input-bg);
            color: var(--body-text);
            border: 1px solid var(--input-border);
            border-radius: 14px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.96), 0 3px 9px rgba(49, 120, 211, 0.08);
            min-height: 46px;
            transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
        }
        div[data-testid="stTextInputRootElement"] input:focus,
        div[data-testid="stNumberInput"] input:focus,
        div[data-baseweb="select"] > div:focus-within,
        div[data-testid="stTextArea"] textarea:focus {
            background: var(--input-bg-focus);
            border-color: var(--input-border-focus) !important;
            box-shadow: 0 0 0 4px rgba(49, 120, 211, 0.26), inset 0 1px 0 rgba(255, 255, 255, 0.98);
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
            box-shadow: 0 0 0 4px rgba(49, 120, 211, 0.22) !important;
        }
        div[data-testid="stSliderTickBarMin"] {
            background: linear-gradient(90deg, var(--blue-6), var(--accent-deep)) !important;
        }
        div[data-testid="stSliderTickBarMax"] {
            background: var(--blue-3) !important;
        }
        button[kind="secondary"] {
            background: rgba(236, 245, 255, 0.96) !important;
            color: #1d5fb6 !important;
            border: 1px solid rgba(49, 120, 211, 0.34) !important;
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
        div[data-testid="stImage"],
        div[data-testid="stPyplot"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(49, 120, 211, 0.24);
            background: linear-gradient(180deg, rgba(253, 255, 255, 0.96) 0%, rgba(233, 244, 255, 0.92) 100%);
            box-shadow: 0 12px 24px rgba(49, 120, 211, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.78);
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
    if page_background_css:
        st.markdown(f"<style>{page_background_css}</style>", unsafe_allow_html=True)


page_background_data_uri = resolve_page_background_data_uri()
inject_styles(page_background_data_uri=page_background_data_uri)
enable_tab_anchor_navigation()
data = load_data()

st.markdown(
    f"""
    <div class="hero-card">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">Portfolio Project</div>
          <h1 class="hero-title">Baby Name<br/>Studio</h1>
          <p class="hero-copy">
            Explore trends fast. Predict Top-100 potential in seconds.
          </p>
          <div class="hero-actions">
            <a class="hero-btn hero-btn-primary" href="#trend-explorer">Start Exploring</a>
          </div>
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
        st.markdown("#### Sample of Recent Top Names")
        render_dataframe(top_2023[["Name", "Gender", "Count"]].reset_index(drop=True))


with trends_tab:
    st.markdown('<div id="trend-explorer"></div>', unsafe_allow_html=True)
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

    st.markdown('<div class="subtle-label">Explore Inputs</div>', unsafe_allow_html=True)

    controls_left, controls_right = st.columns([1.2, 1])
    with controls_left:
        names_input = st.text_input("Names to compare", "Olivia, Liam")
    with controls_right:
        metric_options = {
            "Count": "Count",
            "Name ratio (per 1,000)": "Name_Ratio",
            "Gender ratio (per 1,000)": "Gender_Name_Ratio",
        }
        metric_label = st.selectbox("Metric", list(metric_options.keys()))
        metric = metric_options[metric_label]

    years_left, years_right = st.columns(2)
    with years_left:
        start_year = st.slider("Start year", min_value=int(data["Year"].min()), max_value=int(data["Year"].max() - 1), value=2000)
    with years_right:
        end_year = st.slider("End year", min_value=int(data["Year"].min() + 1), max_value=int(data["Year"].max()), value=int(data["Year"].max()))

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
        for index, name in enumerate(selected_names):
            name_color = COMPARISON_PALETTE[index % len(COMPARISON_PALETTE)]
            name_slice = trend_data[trend_data["Name"] == name]
            for gender in ("F", "M"):
                gender_slice = name_slice[name_slice["Gender"] == gender]
                if gender_slice.empty:
                    continue
                label = f"{name} ({gender})"
                style = GENDER_STYLES[gender]
                line_color = name_color if gender == "F" else darken_color(name_color)
                ax.plot(
                    gender_slice["Year"],
                    gender_slice[metric],
                    label=label,
                    linewidth=2.7,
                    color=line_color,
                    alpha=0.95,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=4.2,
                    markerfacecolor=BLUE_PALETTE[0],
                    markeredgewidth=1.1,
                )

        axis_label = metric_display_label(metric)
        ax.set_title(f"{axis_label} from {start_year} to {end_year}", fontsize=16, color=BLUE_PALETTE[6])
        ax.set_xlabel("Year")
        ax.set_ylabel(axis_label)
        ax.tick_params(colors="#35556f")
        for spine in ax.spines.values():
            spine.set_color(BLUE_PALETTE[2])
        ax.grid(alpha=0.24, color=BLUE_PALETTE[3])
        ax.legend(
            title="Name and gender",
            frameon=True,
            facecolor="#f8fbfe",
            edgecolor=BLUE_PALETTE[4],
            fontsize=10.5,
            title_fontsize=11,
        )
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
        summary = summary.rename(columns={"average_ratio": "average_ratio_per_1000"})
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

    with result_col:
        st.markdown('<div class="eyebrow">Model Output</div>', unsafe_allow_html=True)

        history = name_history(data, name, gender)
        if history.empty:
            st.info("This exact name/gender pair does not appear in the historical dataset yet, so the app is extrapolating from the selected gender baseline.")
        else:
            latest = history.iloc[-1]
            st.metric("Latest observed count", f"{int(latest['Count']):,}", delta=f"{int(latest['Year'])}")
            st.metric("Latest name ratio (per 1,000)", f"{float(latest['Name_Ratio']):.2f}")

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
            versions = runtime_package_versions()
            st.info(
                "Recommended runtime: Python 3.10 with compatible scientific stack. "
                "If this app is deployed, pin the runtime to Python 3.10 in platform settings."
            )
            st.caption(
                "Detected runtime -> "
                f"Python {versions['python']} | "
                f"streamlit {versions['streamlit']} | "
                f"pandas {versions['pandas']} | "
                f"numpy {versions['numpy']} | "
                f"scikit-learn {versions['scikit-learn']} | "
                f"joblib {versions['joblib']}"
            )
            st.markdown(
                """
                **Quick fix checklist**
                1. Create/use a Python 3.10 virtual environment.
                2. Run `pip install -r requirements.txt`.
                3. If model artifacts were produced under another stack, regenerate them with `python scripts/retrain_logistic_model.py`.
                """
            )
            st.caption(f"Debug detail: {exc}")

        if not history.empty:
            render_metric = st.selectbox(
                "Historical metric for this name",
                ["Count", "Name_Ratio (per 1,000)", "Gender_Name_Ratio (per 1,000)"],
                key="single_name_metric",
            )
            render_metric = (
                render_metric.replace("Name_Ratio (per 1,000)", "Name_Ratio")
                .replace("Gender_Name_Ratio (per 1,000)", "Gender_Name_Ratio")
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
    if HERO_ILLUSTRATION_PATH.exists():
        with st.expander("Brand illustration asset"):
            st.image(str(HERO_ILLUSTRATION_PATH), width=380)
