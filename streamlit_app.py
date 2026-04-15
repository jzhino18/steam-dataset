import pandas as pd
import streamlit as st

st.set_page_config(page_title="Steam Price Dataset Explorer", page_icon="📊", layout="wide")
st.markdown(
    """
<style>
.hero-line {
    font-size: 1.5rem;
    font-weight: 600;
    line-height: 1.25;
    margin: 0 0 0.5rem 0;
    border: 2px solid #1f1f1f;
    border-radius: 10px;
    padding: 0.6rem 0.85rem;
    background: #fafafa;
}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown('<p class="hero-line">DS 4420 Final Project</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-line">Predicting Steam Game Prices Using Machine Learning</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-line">Use the top tabs to switch between: the full dashboard and dataset-based example games.</p>',
    unsafe_allow_html=True,
)

FEATURE_GROUPS = {
    "Identity/Basic Info": ["appID", "name", "release_date"],
    "Target Variable": ["price"],
    "Engagement/Popularity Metrics": [
        "peak_ccu",
        "estimated_owners",
        "average_playtime_forever",
        "average_playtime_2weeks",
        "median_playtime_forever",
        "median_playtime_2weeks",
        "recommendations",
    ],
    "Review/Rating Info": [
        "positive",
        "negative",
        "user_score",
        "metacritic_score",
        "metacritic_url",
        "score_rank",
        "reviews",
    ],
    "Game Content Features": ["dlc_count", "achievements", "required_age"],
    "Platform Support": ["windows", "mac", "linux"],
    "Categorical/List Features": [
        "genres",
        "categories",
        "tags",
        "developers",
        "publishers",
        "supported_languages",
        "full_audio_languages",
    ],
    "Text/Media": [
        "detailed_description",
        "short_description",
        "header_image",
        "screenshots",
        "movies",
        "website",
        "support_url",
        "support_email",
        "notes",
        "packages",
    ],
}

SELECTED_MODEL_FEATURES = [
    "genres",
    "tags",
    "metacritic_score",
    "positive",
    "negative",
    "dlc_count",
    "achievements",
    "average_playtime_forever",
    "required_age",
    "categories",
    "peak_ccu",
]


def normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def format_feature_value(value):
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def build_game_profile_table(row, preferred_columns):
    available_columns = [col for col in preferred_columns if col in row.index]
    return pd.DataFrame(
        {
            "Feature": available_columns,
            "Value": [format_feature_value(row[col]) for col in available_columns],
        }
    )


def build_feature_status(df_columns, feature_groups):
    normalized_columns = {normalize_name(col): col for col in df_columns}
    rows = []
    for group_name, features in feature_groups.items():
        for feature in features:
            mapped_column = normalized_columns.get(normalize_name(feature))
            rows.append(
                {
                    "Group": group_name,
                    "Feature in README": feature,
                    "Status in steam_clean.csv": (
                        f"Available as '{mapped_column}'"
                        if mapped_column
                        else "Not present in cleaned file"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_selected_feature_status(df_columns, selected_features):
    normalized_columns = {normalize_name(col): col for col in df_columns}
    rows = []
    for feature in selected_features:
        mapped_column = normalized_columns.get(normalize_name(feature))
        rows.append(
            {
                "Selected modeling feature": feature,
                "Status in steam_clean.csv": (
                    f"Available as '{mapped_column}'"
                    if mapped_column
                    else "Not present in cleaned file"
                ),
            }
        )
    return pd.DataFrame(rows)


def make_game_label(row_index, row):
    possible_name_columns = ["name", "Name", "title", "Title", "game", "Game"]
    for col in possible_name_columns:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    if "appID" in row.index and pd.notna(row["appID"]):
        try:
            return f"App {int(float(row['appID']))}"
        except (TypeError, ValueError):
            return f"App {row['appID']}"
    return f"Dataset Game #{row_index + 1}"


def build_dataset_game_description(row):
    platforms = []
    for col_name, label in [("Windows", "Windows"), ("Mac", "Mac"), ("Linux", "Linux")]:
        if col_name in row.index:
            try:
                if float(row[col_name]) > 0:
                    platforms.append(label)
            except (TypeError, ValueError):
                pass
    platform_text = ", ".join(platforms) if platforms else "unknown platforms"
    return (
        "This entry comes from the cleaned dataset slice used for price-extreme examples. "
        f"Positive={format_feature_value(row.get('Positive', 'N/A'))}, "
        f"Negative={format_feature_value(row.get('Negative', 'N/A'))}, "
        f"Peak CCU={format_feature_value(row.get('Peak CCU', 'N/A'))}, "
        f"Average playtime forever={format_feature_value(row.get('Average playtime forever', 'N/A'))}. "
        f"Platform support: {platform_text}. "
        "These are transformed modeling values from `steam_clean.csv`."
    )


def select_price_extreme_games(df, total_examples=5):
    candidate_df = df.copy()
    candidate_df["price_num"] = pd.to_numeric(candidate_df.get("Price"), errors="coerce")
    candidate_df["positive_num"] = pd.to_numeric(candidate_df.get("Positive"), errors="coerce")
    candidate_df["negative_num"] = pd.to_numeric(candidate_df.get("Negative"), errors="coerce")

    candidate_df = candidate_df[
        (candidate_df["price_num"].notna())
        & (candidate_df["positive_num"] > 0)
        & (candidate_df["negative_num"] > 0)
    ].copy()

    if candidate_df.empty:
        return candidate_df

    low_count = (total_examples + 1) // 2
    high_count = total_examples - low_count

    lowest = candidate_df.nsmallest(low_count, "price_num").copy()
    highest = candidate_df.nlargest(high_count, "price_num").copy()

    lowest["Price Segment"] = "Lowest-price extreme"
    highest["Price Segment"] = "Highest-price extreme"

    selected = pd.concat([lowest, highest]).drop_duplicates()
    if len(selected) < total_examples:
        remaining = candidate_df.drop(selected.index, errors="ignore")
        needed = total_examples - len(selected)
        filler = remaining.nsmallest(needed, "price_num")
        filler = filler.copy()
        filler["Price Segment"] = "Lowest-price extreme"
        selected = pd.concat([selected, filler]).drop_duplicates()

    selected["Dataset Row"] = selected.index + 1
    selected["Game"] = [make_game_label(idx, selected.loc[idx]) for idx in selected.index]
    selected = selected.sort_values(by=["Price Segment", "price_num"], ascending=[True, True])
    return selected.head(total_examples)


@st.cache_data
def load_data():
    return pd.read_csv("data/steam_clean.csv")


@st.cache_data
def load_mlp_predictions():
    return pd.read_csv("data/manual_mlp_regressor_pred_vs_actual.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error("Could not find data/steam_clean.csv. Add the dataset file and rerun.")
    st.stop()

try:
    mlp_pred_df = load_mlp_predictions()
except FileNotFoundError:
    mlp_pred_df = None

# Custom styling for a cleaner black square grid in the cube explorer.
st.markdown(
    """
<style>
div.stButton > button {
    width: 100%;
    aspect-ratio: 1 / 1;
    min-height: 10px;
    padding: 0;
    margin: 0;
    border-radius: 2px;
    font-size: 0;
}
div[data-testid="stHorizontalBlock"] {
    gap: 0.02rem !important;
}
div[data-testid="column"] {
    padding-left: 0.01rem !important;
    padding-right: 0.01rem !important;
}
div.stButton > button[kind="secondary"] {
    background: #000000;
    border: 1px solid #1f1f1f;
}
div.stButton > button[kind="secondary"]:hover {
    background: #111111;
    border: 1px solid #3a3a3a;
}
div.stButton > button[kind="primary"] {
    background: #000000;
    border: 2px solid #ffffff;
}
</style>
""",
    unsafe_allow_html=True,
)

main_tab, examples_tab = st.tabs(["Main Dashboard", "Dataset Game Examples"])

with main_tab:
    st.subheader("1) 200-Cube Dataset Explorer")
    st.write(
        "Click a cube to open one game profile from the dataset. "
        "Cubes map to the first 200 rows in `steam_clean.csv`."
    )

    max_cubes = min(200, len(df))
    if "selected_cube_game_index" not in st.session_state:
        st.session_state.selected_cube_game_index = 0

    cube_columns = 20
    for row_start in range(0, max_cubes, cube_columns):
        cols = st.columns(cube_columns, gap="small")
        for offset, col in enumerate(cols):
            idx = row_start + offset
            if idx >= max_cubes:
                continue
            is_selected = st.session_state.selected_cube_game_index == idx
            with col:
                if st.button(
                    " ",
                    key=f"cube_btn_{idx}",
                    help=f"Dataset row #{idx + 1}",
                    type="primary" if is_selected else "secondary",
                ):
                    st.session_state.selected_cube_game_index = idx

    selected_idx = st.session_state.selected_cube_game_index
    selected_game = df.iloc[selected_idx]
    selected_label = make_game_label(selected_idx, selected_game)
    st.markdown(f"**Selected game:** {selected_label}  |  **Dataset row:** `{selected_idx + 1}`")

    general_feature_columns = [
        "appID",
        "Price",
        "Positive",
        "Negative",
        "Metacritic score",
        "User score",
        "Peak CCU",
        "Recommendations",
        "Average playtime forever",
        "Achievements",
        "Estimated owners",
        "Required age",
        "Windows",
        "Mac",
        "Linux",
        "language_count",
    ]
    profile_df = build_game_profile_table(selected_game, general_feature_columns)
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    st.subheader("2) Dataset Display")
    rows_to_show = st.slider(
        "Rows to display", min_value=10, max_value=200, value=30, step=10, key="rows_display"
    )
    st.dataframe(df.head(rows_to_show), use_container_width=True, hide_index=True)
    st.caption("Loaded from `data/steam_clean.csv`")

    st.subheader("3) Dataset Properties")
    prop_col_1, prop_col_2, prop_col_3 = st.columns(3)
    prop_col_1.metric("Rows", f"{len(df):,}")
    prop_col_2.metric("Columns", f"{len(df.columns):,}")
    prop_col_3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    if "Price" in df.columns:
        price_col_1, price_col_2, price_col_3 = st.columns(3)
        price_col_1.metric("Price Min (transformed)", f"{df['Price'].min():.4f}")
        price_col_2.metric("Price Median (transformed)", f"{df['Price'].median():.4f}")
        price_col_3.metric("Price Max (transformed)", f"{df['Price'].max():.4f}")

    st.markdown(
        """
**Project Properties (from README)**
- Goal: Predict Steam game price from content, quality, and engagement metrics.
- Original source: Steam platform data scraped from the Steam API.
- Modeling dataset in this repo: `steam_clean.csv` (40k+ rows).
- Implemented methods: Manual MLP (NumPy) and Bayesian nonlinear regression (R + brms).
"""
    )

    st.subheader("4) Feature Groups From README")
    feature_status_df = build_feature_status(df.columns, FEATURE_GROUPS)
    st.dataframe(feature_status_df, use_container_width=True, hide_index=True)

    st.subheader("5) Selected Modeling Features")
    selected_status_df = build_selected_feature_status(df.columns, SELECTED_MODEL_FEATURES)
    st.dataframe(selected_status_df, use_container_width=True, hide_index=True)

    with st.expander("Show all columns present in steam_clean.csv"):
        column_df = pd.DataFrame(
            {"Column Name": df.columns, "Data Type": [str(dtype) for dtype in df.dtypes]}
        )
        st.dataframe(column_df, use_container_width=True, hide_index=True)

    st.subheader("6) Methods Used In This Project")
    mlp_tab, bayes_tab = st.tabs(
        ["Method 1: Manual MLP (Python/NumPy)", "Method 2: Bayesian Nonlinear Model (R/brms)"]
    )

    with mlp_tab:
        st.markdown(
            """
**What it is**
- A feedforward neural network implemented manually in NumPy.
- One hidden layer (`H = 4`) with ReLU activation.
- Parameters are `W1` (input to hidden) and `w2` (hidden to output).

**How it was trained**
- Data split: 80/20 train-test after shuffling.
- Features: all columns except `Price`; target: `Price`.
- Feature scaling: min-max scaling using training statistics.
- Loss: mean squared error (MSE).
- Optimization: manual gradient descent (`eta = 0.01`, `500` epochs).
- Training loop computes forward pass and hand-derived backprop gradients for `W1` and `w2`.

**Prediction and evaluation**
- Forward pass: `x -> ReLU(W1^T x) -> w2^T h`.
- Evaluated with test RMSE.
"""
        )

    with bayes_tab:
        st.markdown(
            """
**What it is**
- A Bayesian nonlinear regression model implemented in R with `brms`.
- Target modeled as `log_price = log(Price + 1)` under Gaussian likelihood.

**How it works**
- Uses smooth terms `s(...)` for key engagement signals.
- Includes linear terms for additional features.
- Uses weakly informative priors and MCMC posterior inference.

**Prediction and evaluation**
- Back-transforms posterior predictions with `exp(pred_log) - 1`.
- Reports RMSE and uncertainty intervals.
"""
        )

    st.subheader("7) Manual MLP Predictions vs Actual (CSV Preview)")
    if mlp_pred_df is None:
        st.warning(
            "Could not find `data/manual_mlp_regressor_pred_vs_actual.csv` for the MLP preview."
        )
    else:
        if {"actual_price", "predicted_price"}.issubset(set(mlp_pred_df.columns)):
            rmse_preview = (
                (mlp_pred_df["predicted_price"] - mlp_pred_df["actual_price"]) ** 2
            ).mean() ** 0.5
            mae_preview = (
                mlp_pred_df["predicted_price"] - mlp_pred_df["actual_price"]
            ).abs().mean()

            preview_col_1, preview_col_2, preview_col_3 = st.columns(3)
            preview_col_1.metric("Rows in prediction CSV", f"{len(mlp_pred_df):,}")
            preview_col_2.metric("RMSE (from CSV)", f"{rmse_preview:.4f}")
            preview_col_3.metric("MAE (from CSV)", f"{mae_preview:.4f}")

        preview_rows = st.slider(
            "Rows to preview from prediction CSV",
            min_value=10,
            max_value=200,
            value=30,
            step=10,
            key="pred_rows_display",
        )
        st.dataframe(mlp_pred_df.head(preview_rows), use_container_width=True, hide_index=True)
        st.caption("Loaded from `data/manual_mlp_regressor_pred_vs_actual.csv`")

        if {"actual_price", "predicted_price"}.issubset(set(mlp_pred_df.columns)):
            st.subheader("8) Our Model Accuracy For Games Above $3.50")
            threshold = 3.5
            above_threshold_df = mlp_pred_df[mlp_pred_df["actual_price"] > threshold].copy()

            if len(above_threshold_df) == 0:
                st.info("No games in the prediction CSV have actual price above $3.50.")
            else:
                correct_above = int((above_threshold_df["predicted_price"] > threshold).sum())
                missed_above = int(len(above_threshold_df) - correct_above)
                accuracy_above = correct_above / len(above_threshold_df)

                acc_col_1, acc_col_2, acc_col_3 = st.columns(3)
                acc_col_1.metric("Games with actual price > $3.50", f"{len(above_threshold_df):,}")
                acc_col_2.metric("Correctly predicted as > $3.50", f"{correct_above:,}")
                acc_col_3.metric("Accuracy for this group", f"{accuracy_above:.2%}")

                pie_df = pd.DataFrame(
                    {
                        "Outcome": [
                            "Correctly predicted above $3.50",
                            "Missed above-$3.50 games",
                        ],
                        "Count": [correct_above, missed_above],
                    }
                )

                pie_spec = {
                    "mark": {"type": "arc", "innerRadius": 45},
                    "encoding": {
                        "theta": {"field": "Count", "type": "quantitative"},
                        "color": {"field": "Outcome", "type": "nominal"},
                        "tooltip": [
                            {"field": "Outcome", "type": "nominal"},
                            {"field": "Count", "type": "quantitative"},
                        ],
                    },
                }
                st.vega_lite_chart(pie_df, pie_spec, use_container_width=True)
                st.caption(
                    "Definition used: among games with actual price > $3.50, "
                    "a prediction is counted as correct when predicted price is also > $3.50."
                )

with examples_tab:
    st.subheader("Dataset-Based Example Games")
    st.write(
        "This tab uses only rows from `steam_clean.csv`. "
        "Examples are selected from price extremes only, among games where `Positive > 0` and `Negative > 0`."
    )
    st.image(
        "https://media.giphy.com/media/YnYgi93MEB9LkT0fIj/giphy.gif",
        caption="Stardew Valley GIF",
        use_container_width=False,
        width=420,
    )

    if not {"Positive", "Negative"}.issubset(set(df.columns)):
        st.error("`steam_clean.csv` must contain `Positive` and `Negative` to build this view.")
        st.stop()

    top_games = select_price_extreme_games(df, total_examples=5)
    if top_games.empty:
        st.warning(
            "No rows match the required filter (`Positive > 0` and `Negative > 0` with valid `Price`)."
        )
        st.stop()

    display_columns = ["Dataset Row", "Game", "Price Segment", "Price", "Positive", "Negative"]
    for col in ["Metacritic score", "Peak CCU", "Recommendations"]:
        if col in top_games.columns:
            display_columns.append(col)

    st.dataframe(top_games[display_columns], use_container_width=True, hide_index=True)

    st.subheader("Descriptions (Generated from Dataset Features)")
    detail_columns = [
        "Price",
        "Positive",
        "Negative",
        "Metacritic score",
        "User score",
        "Peak CCU",
        "Recommendations",
        "Average playtime forever",
        "Achievements",
        "Estimated owners",
        "Windows",
        "Mac",
        "Linux",
    ]

    for idx, row in top_games.iterrows():
        game_label = row["Game"]
        with st.expander(game_label):
            st.write(
                f"This is a **{row['Price Segment']}** example with non-zero positive and negative values. "
                f"Price={format_feature_value(row.get('Price', 'N/A'))}, "
                f"Positive={format_feature_value(row.get('Positive', 'N/A'))}, "
                f"Negative={format_feature_value(row.get('Negative', 'N/A'))}. "
                + build_dataset_game_description(row)
            )
            feature_rows = []
            for col in detail_columns:
                if col in top_games.columns:
                    feature_rows.append({"Feature": col, "Value": format_feature_value(row[col])})
            st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True)
