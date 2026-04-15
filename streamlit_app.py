import pandas as pd
import streamlit as st

st.set_page_config(page_title="Steam Price Dataset Explorer", page_icon="📊", layout="wide")
st.title("DS 4420 Final Project")
st.subheader("Predicting Steam Game Prices Using Machine Learning")
st.write(
    """
    This app displays the cleaned Steam dataset used in your project and summarizes the
    properties and feature groups defined in the README.
    """
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


@st.cache_data
def load_data():
    return pd.read_csv("data/steam_clean.csv")


@st.cache_data
def load_mlp_predictions():
    return pd.read_csv("data/manual_mlp_regressor_pred_vs_actual.csv")


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


try:
    df = load_data()
except FileNotFoundError:
    st.error("Could not find data/steam_clean.csv. Add the dataset file and rerun.")
    st.stop()

st.subheader("1) Dataset Display")
rows_to_show = st.slider("Rows to display", min_value=10, max_value=200, value=30, step=10)
st.dataframe(df.head(rows_to_show), use_container_width=True, hide_index=True)
st.caption("Loaded from `data/steam_clean.csv`")

st.subheader("2) Dataset Properties")
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

st.subheader("3) Feature Groups From README")
feature_status_df = build_feature_status(df.columns, FEATURE_GROUPS)
st.dataframe(feature_status_df, use_container_width=True, hide_index=True)

st.subheader("4) Selected Modeling Features")
selected_status_df = build_selected_feature_status(df.columns, SELECTED_MODEL_FEATURES)
st.dataframe(selected_status_df, use_container_width=True, hide_index=True)

with st.expander("Show all columns present in steam_clean.csv"):
    column_df = pd.DataFrame(
        {"Column Name": df.columns, "Data Type": [str(dtype) for dtype in df.dtypes]}
    )
    st.dataframe(column_df, use_container_width=True, hide_index=True)

st.subheader("5) Methods Used In This Project")

mlp_tab, bayes_tab = st.tabs(
    ["Method 1: Manual MLP (Python/NumPy)", "Method 2: Bayesian Nonlinear Model (R/brms)"]
)

with mlp_tab:
    st.markdown(
        """
**What it is**
- A feedforward neural network implemented manually in NumPy (not using high-level deep learning frameworks).
- Architecture used: one hidden layer with `H = 4` hidden units and ReLU activation.
- Parameters are explicitly represented as `W1` (input-to-hidden weights) and `w2` (hidden-to-output weights).

**How it was trained**
- Data split: 80/20 train-test after shuffling.
- Features: all columns except `Price`; target: `Price`.
- Feature scaling: min-max scaling based on training set statistics.
- Loss: mean squared error (MSE) on the transformed target.
- Optimization: manual gradient descent over 500 epochs with learning rate `eta = 0.01`.
- Training loop is sample-by-sample: forward pass (`x -> ReLU(W1^T x) -> w2^T h`), then hand-derived backprop gradients for both `W1` and `w2`.

**Prediction and evaluation**
- Forward pass: input -> hidden ReLU layer -> output price prediction.
- Test metric: RMSE on the test set (`Test RMSE (log scale)` in notebook output).

**Why this method helps**
- Captures nonlinear relationships between gameplay/popularity features and price.
- Provides a flexible baseline for comparison against probabilistic models.
"""
    )

with bayes_tab:
    st.markdown(
        """
**What it is**
- A Bayesian nonlinear regression model implemented in R using `brms`.
- Target is modeled as `log_price = log(Price + 1)` with Gaussian likelihood.

**How nonlinearity is modeled**
- Uses smooth terms `s(...)` for key engagement features, including:
  `Peak.CCU`, `Recommendations`, `Positive`, `Negative`,
  `Average.playtime.forever`, and `Median.playtime.forever`.
- Includes linear terms for additional predictors such as:
  `Required.age`, `DiscountDLC.count`, `Windows`, `Mac`, `Linux`,
  `Achievements`, and `language_count`.

**Bayesian setup**
- Weakly informative priors on intercept, coefficients, residual scale, and smooth-term scales.
- Posterior estimated with MCMC through `brm(...)`.
- Produces posterior predictive means and 95% credible intervals for each game price.

**Prediction and evaluation**
- Posterior expected predictions are back-transformed with `exp(pred_log) - 1`.
- Evaluation uses RMSE plus uncertainty intervals and residual diagnostics.

**Why this method helps**
- Captures nonlinear effects while also quantifying uncertainty in predictions.
- Makes model interpretation stronger by showing confidence/credible bounds.
"""
    )

st.subheader("6) Manual MLP Predictions vs Actual (CSV Preview)")

try:
    mlp_pred_df = load_mlp_predictions()
except FileNotFoundError:
    st.warning(
        "Could not find `data/manual_mlp_regressor_pred_vs_actual.csv` for the MLP prediction preview."
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
        "Rows to preview from prediction CSV", min_value=10, max_value=200, value=30, step=10
    )
    st.dataframe(mlp_pred_df.head(preview_rows), use_container_width=True, hide_index=True)
    st.caption("Loaded from `data/manual_mlp_regressor_pred_vs_actual.csv`")

    if {"actual_price", "predicted_price"}.issubset(set(mlp_pred_df.columns)):
        st.subheader("7) Our Model Accuracy For Games Above $3.50")
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
                "Definition used: among games with actual price > $3.50, a prediction is counted as correct when predicted price is also > $3.50."
            )
