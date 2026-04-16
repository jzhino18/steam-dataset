import pandas as pd
import streamlit as st

st.set_page_config(page_title="Steam Price Modeling Dashboard", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
.hero-line {
    font-size: 1.35rem;
    font-weight: 600;
    line-height: 1.25;
    margin: 0 0 0.5rem 0;
    border: 2px solid #1f1f1f;
    border-radius: 10px;
    padding: 0.55rem 0.8rem;
    background: #fafafa;
}
div.stButton > button {
    width: 100%;
    aspect-ratio: 1 / 1;
    min-height: 0 !important;
    height: auto;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 1px;
    box-sizing: border-box;
    line-height: 0 !important;
    font-size: 0;
}
div[data-testid="stHorizontalBlock"] {
    gap: 0.05rem !important;
}
div[data-testid="column"] {
    padding-left: 0.02rem !important;
    padding-right: 0.02rem !important;
}
div.stButton > button[kind="secondary"] {
    background: #000000;
    border: 1px solid #232323;
}
div.stButton > button[kind="secondary"]:hover {
    background: #111111;
    border: 1px solid #4a4a4a;
}
div.stButton > button[kind="primary"] {
    background: #000000;
    border: 1px solid #232323;
    box-shadow: inset 0 0 0 1px #ffffff;
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
    '<p class="hero-line">Updated pipeline: steam_clean_v2.csv + Model1_Vfinal_2.ipynb + Model2_Vfinal.Rmd</p>',
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return pd.read_csv("data/steam_clean_v2.csv")


def format_feature_value(value):
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def make_game_label(row_index, row):
    for col in ["name", "title", "game", "appid", "steam_appid"]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return f"Dataset Game #{row_index + 1}"


def build_game_profile_table(row, feature_columns):
    cols = [col for col in feature_columns if col in row.index]
    return pd.DataFrame({"Feature": cols, "Value": [format_feature_value(row[col]) for col in cols]})


def select_price_extreme_examples(df, total_examples=5):
    required_cols = {"price", "positive", "negative"}
    if not required_cols.issubset(set(df.columns)):
        return pd.DataFrame()

    candidate_df = df.copy()
    candidate_df["price_num"] = pd.to_numeric(candidate_df["price"], errors="coerce")
    candidate_df["positive_num"] = pd.to_numeric(candidate_df["positive"], errors="coerce")
    candidate_df["negative_num"] = pd.to_numeric(candidate_df["negative"], errors="coerce")

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

    lowest["price_segment"] = "Lowest-price extreme"
    highest["price_segment"] = "Highest-price extreme"

    selected = pd.concat([lowest, highest]).drop_duplicates()
    if len(selected) < total_examples:
        remaining = candidate_df.drop(selected.index, errors="ignore")
        filler = remaining.nsmallest(total_examples - len(selected), "price_num").copy()
        filler["price_segment"] = "Lowest-price extreme"
        selected = pd.concat([selected, filler]).drop_duplicates()

    selected["dataset_row"] = selected.index + 1
    selected["game"] = [make_game_label(idx, selected.loc[idx]) for idx in selected.index]

    return selected.head(total_examples)


try:
    df = load_data()
except FileNotFoundError:
    st.error("Could not find data/steam_clean_v2.csv. Please add the file and rerun.")
    st.stop()

main_tab, examples_tab = st.tabs(["Main Dashboard", "Dataset Game Examples"])

with main_tab:
    st.subheader("1) Cube Explorer (20 Games from steam_clean_v2.csv)")
    st.write(
        "This first section is a compact interactive grid from the new `steam_clean_v2.csv` dataset. "
        "Tap a cube to inspect one game row. The layout is fixed to **5 columns x 4 rows** for mobile friendliness."
    )

    max_cubes = min(20, len(df))
    if "selected_cube_game_index" not in st.session_state or st.session_state.selected_cube_game_index >= max_cubes:
        st.session_state.selected_cube_game_index = 0

    cube_columns = 5
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
                    help=f"steam_clean_v2 row #{idx + 1}",
                    type="primary" if is_selected else "secondary",
                ):
                    st.session_state.selected_cube_game_index = idx

    selected_idx = st.session_state.selected_cube_game_index
    selected_row = df.iloc[selected_idx]
    selected_label = make_game_label(selected_idx, selected_row)
    st.markdown(
        f"**Selected row:** `{selected_idx + 1}` | **Game label:** {selected_label}"
    )

    profile_features = [
        "price",
        "required_age",
        "estimated_owners",
        "peak_ccu",
        "positive",
        "negative",
        "recommendations",
        "metacritic_score",
        "user_score",
        "achievements",
        "average_playtime_forever",
        "median_playtime_forever",
        "language_count",
        "windows",
        "mac",
        "linux",
    ]
    profile_df = build_game_profile_table(selected_row, profile_features)
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    st.subheader("2) Dataset Snapshot")
    st.write(
        "Quick preview of the cleaned v2 modeling table. This is the direct source used by the updated Model 1 and Model 2 pipelines."
    )
    rows_to_show = st.slider("Rows to preview", min_value=10, max_value=120, value=25, step=5)
    st.dataframe(df.head(rows_to_show), use_container_width=True, hide_index=True)
    st.caption("Source file: `data/steam_clean_v2.csv`")

    st.subheader("3) Dataset Properties")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")

    if "price" in df.columns:
        p1, p2, p3 = st.columns(3)
        p1.metric("Price Min (log-scale)", f"{df['price'].min():.4f}")
        p2.metric("Price Median (log-scale)", f"{df['price'].median():.4f}")
        p3.metric("Price Max (log-scale)", f"{df['price'].max():.4f}")

    st.subheader("4) Source Code Used in This Version")
    st.write(
        "The website methodology text and workflow below are aligned to your latest source files and dataset version."
    )
    source_df = pd.DataFrame(
        [
            {"Component": "Main dataset", "File": "steam_clean_v2.csv", "Role": "Primary training table (v2)"},
            {"Component": "Model 1 (Python)", "File": "Model1_Vfinal_2.ipynb", "Role": "Segmented manual MLP training"},
            {"Component": "Model 2 (R)", "File": "Model2_Vfinal.Rmd", "Role": "Bayesian multinomial logistic workflow"},
        ]
    )
    st.dataframe(source_df, use_container_width=True, hide_index=True)

    st.subheader("5) Methodology Overview")
    method_tab_1, method_tab_2 = st.tabs(
        ["Model 1: Segmented MLP (Python)", "Model 2: Bayesian + Multinomial (R)"]
    )

    with method_tab_1:
        st.write(
            "Model 1 uses a manual NumPy MLP implementation, then improves fit with a segmented strategy "
            "that trains separate networks for lower-price and higher-price games."
        )
        st.markdown(
            """
- Start from `steam_clean_v2.csv` (~97k rows, 54 columns in your run).
- Reduce genre dummy width by keeping top frequent genres.
- Engineer three extra features: `review_ratio`, `engagement`, `platform_count`.
- Shuffle and split data 80/20.
- Min-max scale features using train-only statistics.
- Add bias column manually.
- Train manual MLP (`ReLU`, gradient clipping, hand-coded backprop).
- Compare one single MLP vs segmented MLP using median price split.
            """
        )
        model1_results = pd.DataFrame(
            [
                {"Model": "Single MLP", "MSE": 0.6753, "RMSE": 0.8218, "R2": -0.0849},
                {"Model": "Segmented MLP", "MSE": 0.2714, "RMSE": 0.5209, "R2": 0.5640},
                {"Model": "Low-price MLP", "MSE": 0.1337, "RMSE": 0.3657, "R2": -0.1752},
                {"Model": "High-price MLP", "MSE": 0.4081, "RMSE": 0.6389, "R2": -0.3175},
            ]
        )
        st.dataframe(model1_results, use_container_width=True, hide_index=True)
        st.caption("Reported metrics from your provided Model1_Vfinal_2 run summary.")

    with method_tab_2:
        st.write(
            "Model 2 builds a Bayesian multinomial logistic setup around price tiers, then evaluates with a multinomial classifier and confusion-matrix diagnostics."
        )
        st.markdown(
            """
- Source file: `Model2_Vfinal.Rmd`.
- Normalize R column names (`tolower`, replace `.` with `_`).
- Build `price_tier` categories on log-price scale:
  `Free`, `Low`, `Mid`, `High`, `Premium`.
- Keep complete cases for key predictors.
- Train/test split: 80/20 with fixed seed.
- Bayesian model: `MCMCmnl(...)` with 5000 MCMC draws, 1000 burn-in, thin=5.
- Evaluation workflow includes multinomial predictions, confusion matrix, class-wise accuracy, and confidence histograms.
            """
        )

with examples_tab:
    st.subheader("Dataset-Based Game Examples (v2)")
    st.write(
        "These examples are taken directly from `steam_clean_v2.csv`, focusing on **price extremes** while requiring both `positive > 0` and `negative > 0`."
    )

    top_games = select_price_extreme_examples(df, total_examples=5)
    if top_games.empty:
        st.warning(
            "No rows matched the filter criteria (`price` valid, `positive > 0`, `negative > 0`)."
        )
        st.stop()

    display_columns = ["dataset_row", "game", "price_segment", "price", "positive", "negative"]
    for col in [
        "metacritic_score",
        "user_score",
        "peak_ccu",
        "recommendations",
        "average_playtime_forever",
    ]:
        if col in top_games.columns:
            display_columns.append(col)

    st.dataframe(top_games[display_columns], use_container_width=True, hide_index=True)

    st.subheader("Natural Descriptions")
    detail_cols = [
        "price",
        "required_age",
        "positive",
        "negative",
        "metacritic_score",
        "user_score",
        "peak_ccu",
        "recommendations",
        "average_playtime_forever",
        "achievements",
        "language_count",
        "windows",
        "mac",
        "linux",
    ]

    for idx, row in top_games.iterrows():
        with st.expander(f"{row['game']} ({row['price_segment']})"):
            st.write(
                f"This row is from the **{row['price_segment']}** slice. "
                f"It has log-price `{format_feature_value(row.get('price'))}`, "
                f"positive reviews `{format_feature_value(row.get('positive'))}`, and "
                f"negative reviews `{format_feature_value(row.get('negative'))}`."
            )
            row_df = build_game_profile_table(row, detail_cols)
            st.dataframe(row_df, use_container_width=True, hide_index=True)
