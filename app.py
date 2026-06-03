from __future__ import annotations

from io import BytesIO
from pathlib import Path
from datetime import datetime
import re

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# KONFIGURASI APLIKASI
# ============================================================

st.set_page_config(
    page_title="Prediksi Biogas Sapi Perah Indonesia",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = APP_DIR / "data" / "agstar-livestock-ad-database.xlsx"

RANDOM_STATE = 42
TARGET_COLUMN = "Biogas_Generation_Estimate_cu_ft_day"

ANIMAL_COLUMNS = [
    "Cattle",
    "Dairy",
    "Poultry",
    "Swine"
]

MODEL_FEATURES = [
    "Dairy_Population",
    "Total_Animals",
    "Project_Type",
    "Digester_Type",
    "Has_Co_Digestion",
    "Operational_Years",
    "Uses_Electricity",
    "Uses_Cogeneration",
    "Uses_Pipeline_Gas",
    "Uses_CNG",
    "Uses_Boiler",
    "Uses_Flaring"
]

FEATURE_LABELS_ID = {
    "Dairy_Population": "Populasi sapi perah",
    "Total_Animals": "Total ternak",
    "Project_Type": "Skala/jenis proyek",
    "Digester_Type": "Tipe biodigester",
    "Has_Co_Digestion": "Co-digestion",
    "Operational_Years": "Lama operasional",
    "Uses_Electricity": "Pemanfaatan listrik",
    "Uses_Cogeneration": "Pemanfaatan kogenerasi",
    "Uses_Pipeline_Gas": "Pemanfaatan gas pipa",
    "Uses_CNG": "Pemanfaatan CNG",
    "Uses_Boiler": "Pemanfaatan boiler",
    "Uses_Flaring": "Flaring"
}

INDONESIA_PROVINCES = [
    "Aceh",
    "Sumatera Utara",
    "Sumatera Barat",
    "Riau",
    "Kepulauan Riau",
    "Jambi",
    "Sumatera Selatan",
    "Bangka Belitung",
    "Bengkulu",
    "Lampung",
    "Banten",
    "DKI Jakarta",
    "Jawa Barat",
    "Jawa Tengah",
    "DI Yogyakarta",
    "Jawa Timur",
    "Bali",
    "Nusa Tenggara Barat",
    "Nusa Tenggara Timur",
    "Kalimantan Barat",
    "Kalimantan Tengah",
    "Kalimantan Selatan",
    "Kalimantan Timur",
    "Kalimantan Utara",
    "Sulawesi Utara",
    "Gorontalo",
    "Sulawesi Tengah",
    "Sulawesi Barat",
    "Sulawesi Selatan",
    "Sulawesi Tenggara",
    "Maluku",
    "Maluku Utara",
    "Papua",
    "Papua Barat",
    "Papua Selatan",
    "Papua Tengah",
    "Papua Pegunungan",
    "Papua Barat Daya"
]

LOCAL_DIGESTER_MAP = {
    "Kubah tetap / fixed dome": "Complete Mix",
    "Plastik tubular / balon": "Plug Flow",
    "Lagoon tertutup": "Covered Lagoon",
    "Tangki campur lengkap / CSTR": "Complete Mix",
    "Plug flow skala peternakan": "Plug Flow",
    "Biodigester beton komunal": "Complete Mix"
}

PROJECT_TYPE_MAP = {
    "Peternakan rakyat": "Farm Scale",
    "Kelompok ternak / koperasi": "Farm Scale",
    "Peternakan komersial": "Farm Scale",
    "Unit komunal desa": "Centralized",
    "Integrasi koperasi susu": "Farm Scale"
}

CLIMATE_FACTOR_MAP = {
    "Dataran rendah hangat": 1.05,
    "Dataran sedang": 1.00,
    "Dataran tinggi/sejuk": 0.92
}


# ============================================================
# CSS: BLOK EMBLEM / BRANDING STREAMLIT + FOOTER CUSTOM
# ============================================================

def inject_custom_css():
    st.markdown(
        """
        <style>
            /*
                Theme-ready design.
                Streamlit exposes CSS variables such as:
                --background-color, --secondary-background-color,
                --text-color, --primary-color.
                These variables automatically adapt to light and dark mode.
            */

            #MainMenu {
                visibility: hidden;
            }

            header {
                visibility: hidden;
            }

            footer {
                visibility: hidden;
            }

            div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }

            div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }

            div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }

            :root {
                --app-card-radius: 1rem;
                --app-border-soft: rgba(128, 128, 128, 0.28);
                --app-shadow-soft: 0 8px 26px rgba(15, 23, 42, 0.08);
            }

            html,
            body,
            [class*="css"] {
                color: var(--text-color);
            }

            .stApp {
                background: var(--background-color);
                color: var(--text-color);
            }

            .block-container {
                padding-top: 1.25rem;
                padding-bottom: 6rem;
                max-width: 1280px;
            }

            section[data-testid="stSidebar"] {
                background: var(--secondary-background-color);
                border-right: 1px solid var(--app-border-soft);
            }

            section[data-testid="stSidebar"] * {
                color: var(--text-color);
            }

            h1, h2, h3, h4, h5, h6,
            p, span, label, div {
                color: inherit;
            }

            a {
                color: var(--primary-color);
                font-weight: 650;
                text-decoration: none;
            }

            a:hover {
                text-decoration: underline;
            }

            .hero-box {
                padding: 1.35rem 1.5rem;
                border-radius: var(--app-card-radius);
                background:
                    linear-gradient(
                        135deg,
                        color-mix(in srgb, var(--primary-color) 12%, var(--secondary-background-color)),
                        var(--secondary-background-color)
                    );
                border: 1px solid var(--app-border-soft);
                color: var(--text-color);
                margin-bottom: 1rem;
                box-shadow: var(--app-shadow-soft);
            }

            .hero-box h1,
            .hero-box p {
                color: var(--text-color) !important;
            }

            .hero-box h1 {
                line-height: 1.2;
                font-size: clamp(1.75rem, 2.4vw, 2.6rem);
                letter-spacing: -0.02em;
            }

            .hero-box p {
                line-height: 1.65;
                font-size: 1.02rem;
                opacity: 0.92;
            }

            .info-card {
                padding: 1rem;
                border-radius: 0.9rem;
                border: 1px solid var(--app-border-soft);
                background: var(--secondary-background-color);
                color: var(--text-color);
                margin-bottom: 0.75rem;
                box-shadow: var(--app-shadow-soft);
            }

            .small-muted {
                color: color-mix(in srgb, var(--text-color) 72%, transparent);
                font-size: 0.9rem;
            }

            div[data-testid="stMetric"] {
                background: var(--secondary-background-color);
                border: 1px solid var(--app-border-soft);
                border-radius: 0.85rem;
                padding: 0.85rem 1rem;
                box-shadow: var(--app-shadow-soft);
            }

            div[data-testid="stMetric"] label,
            div[data-testid="stMetric"] [data-testid="stMetricLabel"],
            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                color: var(--text-color) !important;
            }

            .stDataFrame,
            div[data-testid="stDataFrame"] {
                border: 1px solid var(--app-border-soft);
                border-radius: 0.85rem;
                overflow: hidden;
                background: var(--secondary-background-color);
            }

            div[data-testid="stMarkdownContainer"] code {
                color: var(--text-color);
                background: color-mix(in srgb, var(--secondary-background-color) 86%, var(--primary-color));
                border: 1px solid var(--app-border-soft);
                border-radius: 0.35rem;
                padding: 0.1rem 0.25rem;
            }

            pre,
            code,
            .stCodeBlock {
                color: var(--text-color) !important;
            }

            .stAlert {
                border-radius: 0.85rem;
                border: 1px solid var(--app-border-soft);
            }

            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div,
            div[data-baseweb="textarea"] > div {
                background: var(--secondary-background-color);
                color: var(--text-color);
                border-color: var(--app-border-soft);
            }

            input,
            textarea {
                color: var(--text-color) !important;
            }

            .custom-footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background:
                    color-mix(in srgb, var(--secondary-background-color) 96%, var(--primary-color));
                color: var(--text-color);
                text-align: center;
                padding: 0.72rem 1rem;
                font-size: 0.86rem;
                line-height: 1.35;
                z-index: 999999;
                border-top: 1px solid var(--app-border-soft);
                box-shadow: 0 -8px 24px rgba(15, 23, 42, 0.12);
            }

            .custom-footer strong {
                color: var(--text-color);
                font-weight: 750;
            }

            .custom-footer a {
                color: var(--primary-color);
                text-decoration: none;
                font-weight: 700;
            }

            .custom-footer a:hover {
                text-decoration: underline;
            }

            @media (max-width: 768px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                    padding-bottom: 7rem;
                }

                .hero-box {
                    padding: 1rem;
                }

                .custom-footer {
                    font-size: 0.76rem;
                    padding: 0.62rem 0.75rem;
                }
            }

            @supports not (color: color-mix(in srgb, white, black)) {
                .hero-box,
                .info-card,
                div[data-testid="stMetric"],
                .custom-footer {
                    background: var(--secondary-background-color);
                    color: var(--text-color);
                }

                .small-muted {
                    opacity: 0.76;
                    color: var(--text-color);
                }
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_footer():
    st.markdown(
        """
        <div class="custom-footer">
            Created by <strong>Galuh Adi Insani</strong> with Kaggle data
            (<a href="https://www.kaggle.com/datasets/mehmetisik/livestock-anaerobic-digester-database/data" target="_blank">
            Livestock Anaerobic Digester Database</a>)
        </div>
        """,
        unsafe_allow_html=True
    )


inject_custom_css()


# ============================================================
# FUNGSI PEMBERSIHAN DATA
# ============================================================

def normalize_column_name(name: str) -> str:
    name = str(name).strip()
    name = name.replace("³", "3")
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def clean_duplicate_columns(columns) -> list[str]:
    result = []
    used = {}

    for col in columns:
        cleaned = normalize_column_name(col)

        if cleaned == "":
            cleaned = "Column"

        if cleaned in used:
            used[cleaned] += 1
            cleaned = f"{cleaned}_{used[cleaned]}"
        else:
            used[cleaned] = 1

        result.append(cleaned)

    return result


def clean_text_value(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text in ["", "-", "nan", "None", "N/A", "NA", "null"]:
        return np.nan

    return text


def to_numeric_clean(series: pd.Series) -> pd.Series:
    cleaned = (
        series
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(cleaned, errors="coerce")


def contains_text(series: pd.Series, pattern: str) -> pd.Series:
    return series.astype(str).str.contains(pattern, case=False, na=False)


def safe_mode(series: pd.Series, fallback="Tidak diketahui"):
    value_counts = series.dropna().astype(str).value_counts()

    if value_counts.empty:
        return fallback

    return value_counts.index[0]


def make_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def sanitize_features_for_sklearn(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            X[col] = X[col].astype(object)
            X[col] = X[col].where(pd.notna(X[col]), np.nan)

    return X


def operational_years(row, reference_year: int):
    year_operational = row.get("Year_Operational", np.nan)

    if pd.isna(year_operational):
        return np.nan

    year_shutdown = row.get("Year_Shutdown", np.nan)
    status = str(row.get("Status", "")).strip().lower()

    if status == "construction":
        return 0

    if status == "shut down" and not pd.isna(year_shutdown):
        return max(year_shutdown - year_operational, 0)

    return max(reference_year - year_operational, 0)


def choose_available_option(preferred_value: str, available_options: list[str]) -> str:
    if preferred_value in available_options:
        return preferred_value

    if available_options:
        return available_options[0]

    return preferred_value


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def read_default_excel(path: str) -> dict[str, pd.DataFrame]:
    excel = pd.ExcelFile(path)
    return {
        sheet_name: pd.read_excel(path, sheet_name=sheet_name)
        for sheet_name in excel.sheet_names
    }


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes: bytes, file_name: str) -> dict[str, pd.DataFrame]:
    lower_name = file_name.lower()

    if lower_name.endswith(".csv"):
        return {
            "Uploaded CSV": pd.read_csv(BytesIO(file_bytes))
        }

    excel = pd.ExcelFile(BytesIO(file_bytes))

    return {
        sheet_name: pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)
        for sheet_name in excel.sheet_names
    }


def prepare_dataframe(
    sheet_frames: dict[str, pd.DataFrame],
    reference_year: int
) -> pd.DataFrame:
    frames = []

    for sheet_name, df in sheet_frames.items():
        temp = df.copy()
        temp.columns = clean_duplicate_columns(temp.columns)
        temp = temp.dropna(how="all")
        temp["Source_Sheet"] = sheet_name
        frames.append(temp)

    df = pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(clean_text_value)

    numeric_columns = [
        "Year_Operational",
        "Year_Shutdown",
        "Cattle",
        "Dairy",
        "Poultry",
        "Swine",
        TARGET_COLUMN
    ]

    for col in numeric_columns:
        if col not in df.columns:
            df[col] = np.nan

        df[col] = to_numeric_clean(df[col])

    for animal in ANIMAL_COLUMNS:
        if animal not in df.columns:
            df[animal] = np.nan

    required_text_columns = [
        "Animal_Farm_Type_s",
        "Biogas_End_Use_s",
        "Project_Type",
        "Digester_Type",
        "Co_Digestion",
        "Status",
        "Project_Name",
        "State"
    ]

    for col in required_text_columns:
        if col not in df.columns:
            df[col] = np.nan

    df["Total_Animals"] = df[ANIMAL_COLUMNS].fillna(0).sum(axis=1)
    df["Dairy_Population"] = df["Dairy"]

    dairy_type_mask = df["Animal_Farm_Type_s"].astype(str).str.contains(
        "dairy",
        case=False,
        na=False
    )

    df.loc[
        df["Dairy_Population"].isna() & dairy_type_mask,
        "Dairy_Population"
    ] = df.loc[
        df["Dairy_Population"].isna() & dairy_type_mask,
        "Total_Animals"
    ]

    df["Has_Co_Digestion"] = np.where(
        df["Co_Digestion"].notna(),
        "Yes",
        "No"
    )

    df["Operational_Years"] = df.apply(
        lambda row: operational_years(row, reference_year),
        axis=1
    )

    df["Uses_Electricity"] = contains_text(
        df["Biogas_End_Use_s"],
        "electricity"
    ).astype(int)

    df["Uses_Cogeneration"] = contains_text(
        df["Biogas_End_Use_s"],
        "cogeneration"
    ).astype(int)

    df["Uses_Pipeline_Gas"] = contains_text(
        df["Biogas_End_Use_s"],
        "pipeline"
    ).astype(int)

    df["Uses_CNG"] = contains_text(
        df["Biogas_End_Use_s"],
        "cng"
    ).astype(int)

    df["Uses_Boiler"] = contains_text(
        df["Biogas_End_Use_s"],
        "boiler|furnace"
    ).astype(int)

    df["Uses_Flaring"] = contains_text(
        df["Biogas_End_Use_s"],
        "flared|flare"
    ).astype(int)

    df["Biogas_m3_day"] = df[TARGET_COLUMN] * 0.0283168

    dairy_mask = (
        df["Dairy_Population"].fillna(0).gt(0)
        |
        df["Animal_Farm_Type_s"].astype(str).str.contains(
            "dairy",
            case=False,
            na=False
        )
    )

    dairy_df = df[dairy_mask].copy()

    dairy_df = dairy_df[
        dairy_df[TARGET_COLUMN].notna()
        &
        dairy_df[TARGET_COLUMN].gt(0)
        &
        dairy_df["Dairy_Population"].notna()
        &
        dairy_df["Dairy_Population"].gt(0)
    ].copy()

    return dairy_df


# ============================================================
# MODEL
# ============================================================

def build_training_data(dairy_df: pd.DataFrame):
    work_df = dairy_df.copy()

    for col in MODEL_FEATURES:
        if col not in work_df.columns:
            work_df[col] = np.nan

    X = work_df[MODEL_FEATURES].copy()
    y = work_df[TARGET_COLUMN].copy()

    X = sanitize_features_for_sklearn(X)

    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, test_size: float):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "Jumlah_Data_Train": len(y_train),
        "Jumlah_Data_Test": len(y_test)
    }

    prediction_df = pd.DataFrame(
        {
            "Aktual_cu_ft_day": y_test.values,
            "Prediksi_cu_ft_day": y_pred,
            "Aktual_m3_day": y_test.values * 0.0283168,
            "Prediksi_m3_day": y_pred * 0.0283168,
            "Absolute_Error_m3_day": np.abs(y_test.values - y_pred) * 0.0283168
        }
    ).reset_index(drop=True)

    return pipeline, metrics, prediction_df, numeric_features, categorical_features


def translate_feature_name(feature_name: str) -> str:
    for raw, label in FEATURE_LABELS_ID.items():
        if feature_name == raw or feature_name.startswith(f"{raw}_"):
            suffix = feature_name.replace(f"{raw}_", "")

            if suffix == feature_name:
                return label

            if suffix == "Yes":
                return f"{label}: Ya"

            if suffix == "No":
                return f"{label}: Tidak"

            return f"{label}: {suffix}"

    return feature_name


def get_feature_importance(
    pipeline,
    numeric_features: list[str],
    categorical_features: list[str]
) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = []
    feature_names.extend(numeric_features)

    if categorical_features:
        onehot = (
            preprocessor
            .named_transformers_["cat"]
            .named_steps["onehot"]
        )

        try:
            cat_names = onehot.get_feature_names_out(categorical_features)
        except Exception:
            cat_names = onehot.get_feature_names(categorical_features)

        feature_names.extend(list(cat_names))

    importance = model.feature_importances_

    if len(feature_names) != len(importance):
        feature_names = [f"Feature_{i}" for i in range(len(importance))]

    result = (
        pd.DataFrame(
            {
                "Fitur_Model": feature_names,
                "Importance": importance
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    result["Faktor"] = result["Fitur_Model"].apply(translate_feature_name)

    return result


def create_default_input_row(X: pd.DataFrame) -> dict:
    row = {}

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            value = X[col].median()
            row[col] = 0.0 if pd.isna(value) else float(value)
        else:
            row[col] = safe_mode(X[col])

    return row


def predict_baseline_biogas(
    pipeline,
    input_row: dict,
    X_columns: list[str]
):
    input_df = pd.DataFrame([input_row], columns=X_columns)
    input_df = sanitize_features_for_sklearn(input_df)

    raw_cuft_day = float(pipeline.predict(input_df)[0])
    raw_m3_day = raw_cuft_day * 0.0283168

    return {
        "raw_cuft_day": raw_cuft_day,
        "raw_m3_day": raw_m3_day
    }


def local_indonesia_estimate(
    dairy_population: float,
    manure_kg_per_head_day: float,
    collection_rate: float,
    biogas_yield_m3_per_kg_manure: float,
    digester_efficiency: float,
    climate_factor: float,
    operation_factor: float
):
    collected_manure_kg_day = (
        dairy_population
        * manure_kg_per_head_day
        * collection_rate
    )

    local_m3_day = (
        collected_manure_kg_day
        * biogas_yield_m3_per_kg_manure
        * digester_efficiency
        * climate_factor
        * operation_factor
    )

    return {
        "collected_manure_kg_day": collected_manure_kg_day,
        "local_m3_day": local_m3_day,
        "local_m3_year": local_m3_day * 365
    }


def combine_prediction(
    baseline_m3_day: float,
    local_m3_day: float,
    baseline_weight: float
):
    baseline_weight = float(baseline_weight)
    local_weight = 1.0 - baseline_weight

    final_m3_day = (
        baseline_m3_day
        * baseline_weight
        +
        local_m3_day
        * local_weight
    )

    return {
        "final_m3_day": final_m3_day,
        "final_m3_year": final_m3_day * 365,
        "baseline_weight": baseline_weight,
        "local_weight": local_weight
    }


def format_number(value, decimals=2):
    if pd.isna(value):
        return "-"

    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return str(value)


def get_option_list(df: pd.DataFrame, col: str, fallback: list[str]):
    if col not in df.columns:
        return fallback

    options = (
        df[col]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    if not options:
        return fallback

    return options


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🐄 Biogas Sapi Perah")

data_source = st.sidebar.radio(
    "Sumber data training",
    [
        "Data bawaan Kaggle/AGSTAR",
        "Upload file sendiri"
    ],
    index=0
)

uploaded_file = None

if data_source == "Upload file sendiri":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel/CSV",
        type=["xlsx", "xls", "csv"]
    )

reference_year = st.sidebar.number_input(
    "Tahun referensi model",
    min_value=1980,
    max_value=2100,
    value=datetime.now().year,
    step=1
)

st.sidebar.divider()

test_size = st.sidebar.slider(
    "Proporsi data uji",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-box">
        <h1>🐄 Prediksi Produksi Biogas Sapi Perah Indonesia</h1>
        <p>
            Aplikasi ini difokuskan untuk estimasi produksi biogas dari populasi sapi perah di Indonesia.
            Model machine learning dilatih dari data Kaggle/AGSTAR khusus dairy sebagai baseline, lalu
            dikombinasikan dengan parameter lokal Indonesia seperti produksi kotoran, tingkat pengumpulan,
            tipe biodigester, iklim, dan efisiensi operasional.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

if data_source == "Data bawaan Kaggle/AGSTAR":
    if not DEFAULT_DATA_PATH.exists():
        st.error("File data bawaan tidak ditemukan di folder data/.")
        render_footer()
        st.stop()

    sheet_frames = read_default_excel(str(DEFAULT_DATA_PATH))
else:
    if uploaded_file is None:
        st.info("Upload file terlebih dahulu atau gunakan data bawaan.")
        render_footer()
        st.stop()

    sheet_frames = read_uploaded_file(
        uploaded_file.getvalue(),
        uploaded_file.name
    )

dairy_df = prepare_dataframe(
    sheet_frames=sheet_frames,
    reference_year=int(reference_year)
)

if dairy_df.empty or len(dairy_df) < 30:
    st.error(
        "Data dairy dengan target biogas valid terlalu sedikit. "
        "Pastikan file memiliki kolom Dairy dan Biogas Generation Estimate."
    )
    render_footer()
    st.stop()

X, y = build_training_data(dairy_df)

pipeline, metrics, prediction_df, numeric_features, categorical_features = train_model(
    X=X,
    y=y,
    test_size=float(test_size)
)

importance_df = get_feature_importance(
    pipeline=pipeline,
    numeric_features=numeric_features,
    categorical_features=categorical_features
)

page = st.sidebar.radio(
    "Halaman",
    [
        "Prediksi Indonesia",
        "Faktor Berpengaruh",
        "Evaluasi Model",
        "Data Dairy",
        "Metodologi"
    ]
)


# ============================================================
# HALAMAN: PREDIKSI INDONESIA
# ============================================================

if page == "Prediksi Indonesia":
    st.header("🔮 Prediksi Biogas Berdasarkan Kondisi Indonesia")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Data dairy valid", f"{len(dairy_df):,}")

    with c2:
        st.metric(
            "Median populasi dairy",
            format_number(dairy_df["Dairy_Population"].median(), 0)
        )

    with c3:
        st.metric(
            "Median biogas data",
            f"{format_number(dairy_df['Biogas_m3_day'].median())} m³/hari"
        )

    with c4:
        st.metric(
            "R² model",
            format_number(metrics["R2"], 3)
        )

    st.subheader("Input Lokasi dan Populasi")

    default_row = create_default_input_row(X)

    col_left, col_right = st.columns(2)

    with col_left:
        province = st.selectbox(
            "Provinsi",
            INDONESIA_PROVINCES,
            index=INDONESIA_PROVINCES.index("Jawa Timur")
            if "Jawa Timur" in INDONESIA_PROVINCES
            else 0
        )

        dairy_population = st.number_input(
            "Populasi sapi perah yang dikelola",
            min_value=1,
            value=int(max(default_row.get("Dairy_Population", 100), 1)),
            step=10,
            help="Jumlah sapi perah yang kotorannya berpotensi dikumpulkan untuk biodigester."
        )

        farm_scale = st.selectbox(
            "Skala usaha peternakan",
            list(PROJECT_TYPE_MAP.keys()),
            index=1
        )

    with col_right:
        local_digester_type = st.selectbox(
            "Tipe biodigester yang digunakan",
            list(LOCAL_DIGESTER_MAP.keys()),
            index=0
        )

        has_co_digestion = st.selectbox(
            "Apakah ada co-digestion?",
            ["Tidak", "Ya"],
            index=0,
            help="Co-digestion berarti kotoran sapi dicampur dengan bahan organik lain seperti limbah pertanian atau limbah pangan."
        )

        climate_zone = st.selectbox(
            "Kondisi iklim/lokasi kandang",
            list(CLIMATE_FACTOR_MAP.keys()),
            index=1
        )

    st.subheader("Parameter Lokal Indonesia")

    p1, p2, p3 = st.columns(3)

    with p1:
        manure_kg_per_head_day = st.slider(
            "Kotoran segar per ekor per hari, kg",
            min_value=10.0,
            max_value=40.0,
            value=25.0,
            step=1.0,
            help="Parameter ini dapat disesuaikan berdasarkan data lapangan peternakan sapi perah lokal."
        )

        collection_rate = st.slider(
            "Tingkat kotoran yang terkumpul",
            min_value=0.10,
            max_value=1.00,
            value=0.70,
            step=0.05,
            help="Contoh 0,70 berarti 70% kotoran sapi benar-benar masuk ke biodigester."
        )

    with p2:
        biogas_yield_m3_per_kg_manure = st.slider(
            "Potensi biogas per kg kotoran, m³/kg",
            min_value=0.010,
            max_value=0.080,
            value=0.035,
            step=0.005,
            help="Nilai ini adalah asumsi teknis yang dapat dikalibrasi dengan data lapangan Indonesia."
        )

        digester_efficiency = st.slider(
            "Efisiensi biodigester",
            min_value=0.30,
            max_value=1.00,
            value=0.75,
            step=0.05
        )

    with p3:
        operation_factor = st.slider(
            "Faktor manajemen operasi",
            min_value=0.50,
            max_value=1.30,
            value=1.00,
            step=0.05,
            help="Mewakili kualitas perawatan, stabilitas feeding, pemeliharaan, dan disiplin operasi."
        )

        baseline_weight = st.slider(
            "Bobot model Kaggle/AGSTAR",
            min_value=0.00,
            max_value=1.00,
            value=0.40,
            step=0.05,
            help="0 berarti hanya rumus lokal; 1 berarti hanya baseline ML dari data Kaggle/AGSTAR."
        )

    available_project_types = get_option_list(
        dairy_df,
        "Project_Type",
        ["Farm Scale"]
    )

    available_digester_types = get_option_list(
        dairy_df,
        "Digester_Type",
        ["Complete Mix", "Plug Flow", "Covered Lagoon"]
    )

    mapped_project_type = choose_available_option(
        PROJECT_TYPE_MAP[farm_scale],
        available_project_types
    )

    mapped_digester_type = choose_available_option(
        LOCAL_DIGESTER_MAP[local_digester_type],
        available_digester_types
    )

    end_use_electricity = 1
    end_use_cogeneration = 0
    end_use_pipeline = 0
    end_use_cng = 0
    end_use_boiler = 1
    end_use_flaring = 0

    input_row = default_row.copy()

    input_row.update(
        {
            "Dairy_Population": float(dairy_population),
            "Total_Animals": float(dairy_population),
            "Project_Type": mapped_project_type,
            "Digester_Type": mapped_digester_type,
            "Has_Co_Digestion": "Yes" if has_co_digestion == "Ya" else "No",
            "Operational_Years": 1.0,
            "Uses_Electricity": end_use_electricity,
            "Uses_Cogeneration": end_use_cogeneration,
            "Uses_Pipeline_Gas": end_use_pipeline,
            "Uses_CNG": end_use_cng,
            "Uses_Boiler": end_use_boiler,
            "Uses_Flaring": end_use_flaring
        }
    )

    baseline_result = predict_baseline_biogas(
        pipeline=pipeline,
        input_row=input_row,
        X_columns=X.columns.tolist()
    )

    local_result = local_indonesia_estimate(
        dairy_population=float(dairy_population),
        manure_kg_per_head_day=float(manure_kg_per_head_day),
        collection_rate=float(collection_rate),
        biogas_yield_m3_per_kg_manure=float(biogas_yield_m3_per_kg_manure),
        digester_efficiency=float(digester_efficiency),
        climate_factor=float(CLIMATE_FACTOR_MAP[climate_zone]),
        operation_factor=float(operation_factor)
    )

    final_result = combine_prediction(
        baseline_m3_day=baseline_result["raw_m3_day"],
        local_m3_day=local_result["local_m3_day"],
        baseline_weight=float(baseline_weight)
    )

    st.subheader("Hasil Prediksi")

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "Prediksi final Indonesia",
            f"{format_number(final_result['final_m3_day'])} m³/hari"
        )

    with r2:
        st.metric(
            "Prediksi tahunan",
            f"{format_number(final_result['final_m3_year'])} m³/tahun"
        )

    with r3:
        st.metric(
            "Estimasi kotoran terkumpul",
            f"{format_number(local_result['collected_manure_kg_day'])} kg/hari"
        )

    with r4:
        st.metric(
            "Provinsi",
            province
        )

    with st.expander("Lihat komponen perhitungan"):
        component_df = pd.DataFrame(
            [
                {
                    "Komponen": "Baseline model Kaggle/AGSTAR",
                    "Nilai_m3_hari": baseline_result["raw_m3_day"],
                    "Bobot": final_result["baseline_weight"]
                },
                {
                    "Komponen": "Estimasi lokal Indonesia",
                    "Nilai_m3_hari": local_result["local_m3_day"],
                    "Bobot": final_result["local_weight"]
                },
                {
                    "Komponen": "Prediksi final gabungan",
                    "Nilai_m3_hari": final_result["final_m3_day"],
                    "Bobot": 1.00
                }
            ]
        )

        st.dataframe(
            component_df,
            use_container_width=True
        )

        st.code(
            f"""
Biogas lokal =
    Populasi sapi perah
    × kotoran kg/ekor/hari
    × tingkat kotoran terkumpul
    × potensi biogas m³/kg
    × efisiensi biodigester
    × faktor iklim
    × faktor manajemen operasi

Biogas lokal =
    {dairy_population}
    × {manure_kg_per_head_day}
    × {collection_rate}
    × {biogas_yield_m3_per_kg_manure}
    × {digester_efficiency}
    × {CLIMATE_FACTOR_MAP[climate_zone]}
    × {operation_factor}
    = {format_number(local_result['local_m3_day'])} m³/hari

Prediksi final =
    baseline Kaggle/AGSTAR × {baseline_weight}
    + estimasi lokal Indonesia × {1 - baseline_weight}
    = {format_number(final_result['final_m3_day'])} m³/hari
            """.strip()
        )

    st.subheader("Sensitivitas Populasi Sapi Perah")

    sensitivity_rows = []

    for multiplier in [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00]:
        simulated_population = max(int(dairy_population * multiplier), 1)

        simulated_input = input_row.copy()
        simulated_input["Dairy_Population"] = simulated_population
        simulated_input["Total_Animals"] = simulated_population

        simulated_baseline = predict_baseline_biogas(
            pipeline=pipeline,
            input_row=simulated_input,
            X_columns=X.columns.tolist()
        )

        simulated_local = local_indonesia_estimate(
            dairy_population=float(simulated_population),
            manure_kg_per_head_day=float(manure_kg_per_head_day),
            collection_rate=float(collection_rate),
            biogas_yield_m3_per_kg_manure=float(biogas_yield_m3_per_kg_manure),
            digester_efficiency=float(digester_efficiency),
            climate_factor=float(CLIMATE_FACTOR_MAP[climate_zone]),
            operation_factor=float(operation_factor)
        )

        simulated_final = combine_prediction(
            baseline_m3_day=simulated_baseline["raw_m3_day"],
            local_m3_day=simulated_local["local_m3_day"],
            baseline_weight=float(baseline_weight)
        )

        sensitivity_rows.append(
            {
                "Populasi_Sapi_Perah": simulated_population,
                "Prediksi_Biogas_m3_hari": simulated_final["final_m3_day"]
            }
        )

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    st.line_chart(
        sensitivity_df,
        x="Populasi_Sapi_Perah",
        y="Prediksi_Biogas_m3_hari",
        use_container_width=True
    )

    st.dataframe(
        sensitivity_df,
        use_container_width=True
    )


# ============================================================
# HALAMAN: FAKTOR BERPENGARUH
# ============================================================

elif page == "Faktor Berpengaruh":
    st.header("📌 Faktor yang Mempengaruhi Produksi Biogas")

    st.write(
        "Faktor di bawah berasal dari dua sumber: feature importance model Random Forest "
        "dan parameter lokal Indonesia yang digunakan dalam rumus penyesuaian."
    )

    st.subheader("Feature Importance dari Model")

    top_importance = importance_df.head(20).copy()

    st.dataframe(
        top_importance[["Faktor", "Fitur_Model", "Importance"]],
        use_container_width=True
    )

    st.bar_chart(
        top_importance,
        x="Faktor",
        y="Importance",
        use_container_width=True
    )

    st.subheader("Faktor Lokal Indonesia yang Perlu Diperhatikan")

    factors_local = pd.DataFrame(
        [
            {
                "Faktor": "Populasi sapi perah",
                "Dampak": "Semakin banyak sapi, potensi bahan baku kotoran semakin besar."
            },
            {
                "Faktor": "Kotoran segar kg/ekor/hari",
                "Dampak": "Menentukan jumlah substrat yang tersedia untuk biodigester."
            },
            {
                "Faktor": "Tingkat kotoran yang terkumpul",
                "Dampak": "Sangat penting pada peternakan rakyat karena tidak semua kotoran masuk ke sistem."
            },
            {
                "Faktor": "Tipe biodigester",
                "Dampak": "Mempengaruhi stabilitas proses, retensi, dan efisiensi produksi gas."
            },
            {
                "Faktor": "Iklim/lokasi kandang",
                "Dampak": "Dataran tinggi yang lebih sejuk dapat menurunkan performa proses biologis."
            },
            {
                "Faktor": "Manajemen operasi",
                "Dampak": "Feeding tidak stabil, kebocoran, dan perawatan buruk dapat menurunkan produksi."
            },
            {
                "Faktor": "Co-digestion",
                "Dampak": "Campuran limbah organik lain dapat menaikkan atau menurunkan produksi tergantung kualitas bahan."
            }
        ]
    )

    st.dataframe(
        factors_local,
        use_container_width=True
    )

    st.subheader("Hubungan Populasi Sapi Perah vs Biogas pada Data Training")

    relation_df = dairy_df[
        [
            "Dairy_Population",
            "Biogas_m3_day",
            "Digester_Type",
            "Project_Type",
            "Has_Co_Digestion"
        ]
    ].dropna(
        subset=[
            "Dairy_Population",
            "Biogas_m3_day"
        ]
    )

    st.scatter_chart(
        relation_df,
        x="Dairy_Population",
        y="Biogas_m3_day",
        use_container_width=True
    )


# ============================================================
# HALAMAN: EVALUASI MODEL
# ============================================================

elif page == "Evaluasi Model":
    st.header("📈 Evaluasi Model Baseline Kaggle/AGSTAR")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("MAE", f"{format_number(metrics['MAE'] * 0.0283168)} m³/hari")

    with m2:
        st.metric("RMSE", f"{format_number(metrics['RMSE'] * 0.0283168)} m³/hari")

    with m3:
        st.metric("R² Score", format_number(metrics["R2"], 3))

    with m4:
        st.metric("Data uji", f"{metrics['Jumlah_Data_Test']:,}")

    st.subheader("Aktual vs Prediksi")

    display_prediction_df = prediction_df[
        [
            "Aktual_m3_day",
            "Prediksi_m3_day",
            "Absolute_Error_m3_day"
        ]
    ].copy()

    display_prediction_df.columns = [
        "Aktual m³/hari",
        "Prediksi m³/hari",
        "Absolute Error m³/hari"
    ]

    st.dataframe(
        display_prediction_df,
        use_container_width=True
    )

    st.line_chart(
        display_prediction_df[
            [
                "Aktual m³/hari",
                "Prediksi m³/hari"
            ]
        ],
        use_container_width=True
    )

    st.info(
        "Evaluasi ini hanya untuk baseline model dari data Kaggle/AGSTAR. "
        "Prediksi final Indonesia tetap memakai penyesuaian parameter lokal."
    )


# ============================================================
# HALAMAN: DATA DAIRY
# ============================================================

elif page == "Data Dairy":
    st.header("🧾 Data Dairy yang Digunakan")

    selected_columns = [
        "Project_Name",
        "Source_Sheet",
        "Project_Type",
        "State",
        "Digester_Type",
        "Status",
        "Year_Operational",
        "Dairy_Population",
        "Total_Animals",
        "Has_Co_Digestion",
        "Biogas_Generation_Estimate_cu_ft_day",
        "Biogas_m3_day",
        "Biogas_End_Use_s"
    ]

    selected_columns = [
        col for col in selected_columns
        if col in dairy_df.columns
    ]

    st.dataframe(
        dairy_df[selected_columns],
        use_container_width=True
    )

    st.download_button(
        "Download data dairy bersih CSV",
        data=dairy_df.to_csv(index=False).encode("utf-8"),
        file_name="dairy_biogas_cleaned.csv",
        mime="text/csv"
    )


# ============================================================
# HALAMAN: METODOLOGI
# ============================================================

elif page == "Metodologi":
    st.header("📚 Metodologi Penyesuaian Indonesia")

    st.markdown(
        """
        ### Fokus aplikasi

        Aplikasi ini dibuat hanya untuk:

        - sapi perah/dairy;
        - prediksi produksi biogas;
        - konteks peternakan sapi perah di Indonesia;
        - identifikasi faktor yang mempengaruhi produksi biogas.

        ### Sumber data training

        Model baseline dilatih dari data Kaggle:

        `Livestock Anaerobic Digester Database`

        Data difilter hanya untuk proyek yang memiliki:

        - populasi sapi perah;
        - target `Biogas Generation Estimate`;
        - nilai biogas lebih besar dari nol.

        ### Penyesuaian Indonesia

        Prediksi final tidak hanya memakai model Kaggle/AGSTAR. Aplikasi ini menambahkan
        estimasi lokal berbasis parameter Indonesia:

        ```text
        Biogas lokal =
            Populasi sapi perah
            × kotoran kg/ekor/hari
            × tingkat kotoran terkumpul
            × potensi biogas m³/kg
            × efisiensi biodigester
            × faktor iklim
            × faktor manajemen operasi
        ```

        Prediksi final dihitung sebagai gabungan:

        ```text
        Prediksi final =
            baseline Kaggle/AGSTAR × bobot baseline
            + estimasi lokal Indonesia × bobot lokal
        ```

        ### Kenapa perlu bobot?

        Karena data Kaggle/AGSTAR bukan data Indonesia murni. Bobot memungkinkan pengguna
        menentukan seberapa besar aplikasi mengikuti pola data internasional dan seberapa besar
        mengikuti asumsi lokal Indonesia.


        ### Desain light/dark mode

        Tampilan aplikasi menggunakan warna adaptif dari tema Streamlit, sehingga elemen seperti
        card, metric, sidebar, tabel, kode, link, dan footer tetap terbaca pada tema terang maupun gelap.

        ### Batasan

        Aplikasi ini tetap membutuhkan validasi lapangan. Untuk hasil yang lebih akurat,
        pengguna sebaiknya mengisi parameter berdasarkan data riil peternakan, seperti:

        - jumlah sapi perah aktif;
        - produksi kotoran aktual;
        - persentase kotoran terkumpul;
        - jenis biodigester;
        - performa operasi;
        - data produksi biogas harian.
        """
    )


render_footer()
