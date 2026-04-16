from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DS 4420 - Steam Price Modeling", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
.hero-line {
    font-size: 1.35rem;
    font-weight: 650;
    line-height: 1.2;
    margin: 0 0 0.45rem 0;
    border: 2px solid #1f1f1f;
    border-radius: 10px;
    padding: 0.55rem 0.8rem;
    background: #fafafa;
}
.section-head {
    margin-top: 0.8rem;
}
.note-box {
    border-left: 5px solid #1f1f1f;
    padding: 0.6rem 0.8rem;
    background: #f7f7f7;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="hero-line">DS 4420 Final Project</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-line">Predicting Steam Game Prices with Manual Neural Networks and Bayesian Modeling</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-line">How game quality, engagement, and content signals shape pricing on the Steam marketplace.</p>',
    unsafe_allow_html=True,
)

# Per request: keep Stardew GIF on the first page/main view area.
st.image(
    "https://media.giphy.com/media/YnYgi93MEB9LkT0fIj/giphy.gif",
    caption="Stardew Valley",
    width=380,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("data/steam_clean_v2.csv")


@st.cache_data
def build_model2_plots_data():
    # Values taken from your poster / Model2_Vfinal.Rmd discussion
    tier_share = pd.DataFrame(
        {
            "Price Tier": ["Low", "Mid", "High", "Premium"],
            "Share (%)": [67.6, 26.8, 4.4, 1.1],
        }
    )
    class_recall = pd.DataFrame(
        {
            "Price Tier": ["Low", "Mid", "High", "Premium"],
            "Correctly Classified (%)": [98.0, 92.0, 0.2, 0.0],
        }
    )
    posterior_effects = pd.DataFrame(
        {
            "Feature": [
                "Positive reviews",
                "Peak CCU",
                "Language count",
            ],
            "Posterior takeaway": [
                "Higher positive reviews align with cheaper tiers in posterior summaries (Premium beta around -0.38).",
                "Higher peak CCU increases premium-tier tendency.",
                "Small but consistent positive effect across tiers.",
            ],
        }
    )
    return tier_share, class_recall, posterior_effects


def safe_metric(df: pd.DataFrame, col: str, fn: str):
    if col not in df.columns:
        return "N/A"
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return "N/A"
    if fn == "min":
        return f"{series.min():.4f}"
    if fn == "median":
        return f"{series.median():.4f}"
    if fn == "max":
        return f"{series.max():.4f}"
    return "N/A"


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


def select_price_extreme_examples(df: pd.DataFrame, total_examples: int = 5) -> pd.DataFrame:
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


def show_model1_figures():
    fig_dir = Path("data/model1_figures")
    fig_paths = {
        "Model 1 Diagnostics: Predicted vs Actual + Residuals (Single vs Segmented)": fig_dir
        / "mlp_pred_vs_actual_scatter.png",
        "Single MLP: Training Loss Over 700 Epochs": fig_dir / "mlp_residuals_vs_predicted.png",
        "Segmented MLP: Training Loss Over 700 Epochs (Low vs High Price Models)": fig_dir
        / "mlp_actual_vs_pred_distribution.png",
    }

    available = {k: v for k, v in fig_paths.items() if v.exists()}
    if not available:
        st.info(
            "Model 1 plot images were not found in `data/model1_figures/`. "
            "Add the PNG files there to display them in the dashboard."
        )
        return

    st.write("These visuals are included to mirror the key diagnostics discussed in Model 1.")
    cols = st.columns(min(3, len(available)))
    for i, (title, path) in enumerate(available.items()):
        with cols[i % len(cols)]:
            st.image(str(path), caption=title, use_container_width=True)


try:
    df = load_data()
except FileNotFoundError:
    st.error("Could not find `data/steam_clean_v2.csv`. Please add the file and rerun.")
    st.stop()

main_tab, examples_tab, future_tab = st.tabs(
    ["Main Dashboard", "Dataset Game Examples", "Future Enhancements + Code Notes"]
)

with main_tab:
    st.markdown("<h3 class='section-head'>Project Purpose</h3>", unsafe_allow_html=True)
    st.write(
        "Our central question is: how much of a Steam game's price can be explained by the signals we observe in the data "
        "(reviews, engagement, content, and quality indicators)? "
        "This matters because pricing on Steam is not random. Lower-priced indie titles and higher-priced premium titles often behave like different markets, "
        "so the modeling approach needs to account for that."
    )

    st.markdown("<h3 class='section-head'>Dataset and Preprocessing Context</h3>", unsafe_allow_html=True)
    summary_c1, summary_c2, summary_c3 = st.columns(3)
    summary_c1.metric("Rows", f"{len(df):,}")
    summary_c2.metric("Columns", f"{len(df.columns):,}")
    summary_c3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    price_c1, price_c2, price_c3 = st.columns(3)
    price_c1.metric("Price Min (log scale)", safe_metric(df, "price", "min"))
    price_c2.metric("Price Median (log scale)", safe_metric(df, "price", "median"))
    price_c3.metric("Price Max (log scale)", safe_metric(df, "price", "max"))

    st.write(
        "The app now uses `steam_clean_v2.csv` as the main dataset. "
        "This table is already prepared for modeling, including transformed skewed variables and encoded content features. "
        "That gives us a cleaner input space for both the neural-network model and the Bayesian model, while keeping the most useful pricing signals."
    )

    rows_to_show = st.slider("Rows to preview", min_value=10, max_value=100, value=20, step=5, key="rows_preview")
    st.dataframe(df.head(rows_to_show), use_container_width=True, hide_index=True)

    st.markdown("<h3 class='section-head'>Source Files Driving This Version</h3>", unsafe_allow_html=True)
    source_df = pd.DataFrame(
        [
            {"Component": "Dataset", "File": "steam_clean_v2.csv", "Purpose": "Main modeling input table"},
            {"Component": "Model 1 (Python)", "File": "Model1_Vfinal_2.ipynb", "Purpose": "Manual + segmented MLP training"},
            {"Component": "Model 2 (R)", "File": "Model2_Vfinal.Rmd", "Purpose": "Bayesian multinomial workflow"},
        ]
    )
    st.dataframe(source_df, use_container_width=True, hide_index=True)

    st.markdown("<h3 class='section-head'>Model 1: Manual MLP and Segmented MLP (Python)</h3>", unsafe_allow_html=True)
    st.write(
        "We built Model 1 manually in NumPy so we could fully control the forward pass, gradients, and updates. "
        "The base architecture is one hidden layer (64 units), ReLU activation, and a linear output for log-price prediction. "
        "We trained with full-batch gradient descent at learning rate 0.001 for 700 epochs, and clipped gradients at +/-5 to keep training stable."
    )

    st.markdown(
        """
<div class="note-box">
<strong>Why segmentation improved results:</strong> after fitting one global MLP, residual plots showed that error patterns were very different for cheaper vs. expensive games. 
Splitting at the median log-price let us train two specialized networks, each focused on a narrower price regime.
</div>
""",
        unsafe_allow_html=True,
    )

    model1_metrics = pd.DataFrame(
        [
            {"Model": "Single MLP", "MSE": 0.6753, "RMSE": 0.8218, "R²": -0.0849},
            {"Model": "Low-Price MLP", "MSE": 0.1337, "RMSE": 0.3657, "R²": -0.1752},
            {"Model": "High-Price MLP", "MSE": 0.4081, "RMSE": 0.6389, "R²": -0.3175},
            {"Model": "Segmented MLP (Combined)", "MSE": 0.2714, "RMSE": 0.5209, "R²": 0.5640},
        ]
    )
    st.dataframe(model1_metrics, use_container_width=True, hide_index=True)

    st.write(
        "The key takeaway is not only better metrics. It is that Steam pricing is a multi-regime problem, "
        "and model design choices should reflect that structure."
    )

    st.markdown("<h4>Model 1 Diagnostic Graphs</h4>", unsafe_allow_html=True)
    show_model1_figures()

    st.markdown("<h3 class='section-head'>Model 2: Bayesian Multinomial Logistic Regression (R)</h3>", unsafe_allow_html=True)
    st.write(
        "Model 2 reframes the task from regression to price-tier classification (Low, Mid, High, Premium). "
        "That framing fits Steam's price behavior well, because titles tend to cluster around common storefront price points."
    )

    st.markdown(
        """
- `MCMCpack::MCMCmnl` is used for Bayesian multinomial regression so we can estimate posterior uncertainty, not just point estimates.
- `nnet::multinom` is used for clean out-of-sample class predictions and confusion-matrix reporting.
- `ggplot2` (and reshaping helpers in R) supports clear, reproducible visual diagnostics for the report.
"""
    )

    tier_share_df, class_recall_df, posterior_effects_df = build_model2_plots_data()

    m2a, m2b = st.columns(2)
    with m2a:
        st.metric("Overall Test Accuracy", "68.3%")
    with m2b:
        st.metric("Naive Majority-Class Baseline", "67.6%")

    st.write(
        "Performance is strongest on Low and Mid tiers, while High and Premium are harder because those classes are much smaller. "
        "Even so, the Bayesian setup is valuable because it tells us which features have consistent directional influence on tier membership."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Class distribution (imbalance context)**")
        tier_chart = (
            alt.Chart(tier_share_df)
            .mark_bar(color="#355C7D")
            .encode(x=alt.X("Price Tier:N", sort=["Low", "Mid", "High", "Premium"]), y="Share (%):Q")
        )
        st.altair_chart(tier_chart, use_container_width=True)

    with right:
        st.markdown("**Class-wise accuracy (from confusion results)**")
        recall_chart = (
            alt.Chart(class_recall_df)
            .mark_bar(color="#6C5B7B")
            .encode(
                x=alt.X("Price Tier:N", sort=["Low", "Mid", "High", "Premium"]),
                y=alt.Y("Correctly Classified (%):Q", scale=alt.Scale(domain=[0, 100])),
            )
        )
        st.altair_chart(recall_chart, use_container_width=True)

    st.markdown("**Posterior interpretation highlights**")
    st.dataframe(posterior_effects_df, use_container_width=True, hide_index=True)

    st.markdown("<h3 class='section-head'>Results Interpretation</h3>", unsafe_allow_html=True)
    st.write(
        "Putting both models together gives a clearer story. "
        "The segmented manual MLP gives stronger prediction performance once price regimes are separated. "
        "The Bayesian tier model gives better interpretability and uncertainty context, but is limited by imbalance in expensive classes."
    )

    st.markdown("<h3 class='section-head'>Conclusion and Next Steps</h3>", unsafe_allow_html=True)
    st.markdown(
        """
- Treat price-tier classification as a first-class task, not only continuous regression.
- Introduce imbalance-aware objectives (class weights / focal loss / re-sampling) for expensive tiers.
- Move manual full-batch MLP to mini-batch PyTorch while preserving segmented regime logic.
- Add publisher/developer embeddings to capture brand premium effects.
"""
    )

    st.markdown(
        "<h3 class='section-head'>Bottom Appendix Tabs</h3>",
        unsafe_allow_html=True,
    )
    appendix_tab1, appendix_tab2 = st.tabs(
        ["Implementation Notes", "MLP NN vs Bayesian Nonlinear Use Cases"]
    )

    with appendix_tab1:
        st.write(
            "These model implementations were built from lecture-inspired methods and then adapted to this Steam dataset, "
            "rather than copied directly from outside scripts."
        )
        st.markdown(
            """
- **Context note:** `nn_mlp.ipynb` and the other reference files were used as lecture-style inspiration only. The final implementation logic shown here reflects your own project design decisions.
"""
        )

        st.markdown("**How we designed the manual MLP training**")
        st.markdown(
            """
- We trained the first neural model manually in NumPy to keep every step auditable: initialization, forward pass, backpropagation, and update rules.
- We selected a single hidden layer with **64 ReLU units** as a deliberate middle ground: expressive enough to model nonlinearity, but still easy to debug and explain.
- We used a **linear output** because price prediction is a continuous regression problem.
- We used **full-batch gradient descent** at learning rate `0.001` to produce stable, interpretable learning curves over time.
- We trained for **700 epochs** because the loss continued to improve well past early epochs, and we wanted convergence behavior to be visible in diagnostics.
- We applied **gradient clipping at +/-5** to prevent rare unstable updates and keep training numerically controlled.
"""
        )

        st.markdown("**Why segmentation became necessary**")
        st.markdown(
            """
- After training one global MLP, residual plots showed a clear pattern: the model struggled differently on cheaper vs. expensive games.
- That signal suggested two pricing regimes, so we split data at the **median log-price threshold** and trained separate low-price and high-price MLPs.
- This was a strategic choice, not only a metric trick: each network can specialize in a narrower distribution instead of compromising across the entire market.
- The combined segmented model then improved overall predictive fit and produced more realistic price-range behavior.
"""
        )

        st.markdown("**Feature engineering logic we used**")
        st.markdown(
            """
- We reduced dimensional noise by keeping only the most common genre indicators instead of a very sparse full genre matrix.
- We built `review_ratio = positive / (positive + negative)` to capture review sentiment in one robust feature, with fallback handling when review counts are zero.
- We created `engagement` from average and median playtime to summarize long-term usage intensity.
- We created `platform_count` to represent release breadth across Windows, Mac, and Linux.
- We kept predictors that have pricing intuition: player interest (`peak_ccu`, `recommendations`), perceived quality (`metacritic_score`, review signals), and game depth (`achievements`, playtime, language coverage).
"""
        )

        st.markdown("**Training time and implementation summary**")
        st.markdown(
            """
- Manual MLP: trained for 700 epochs on an 80/20 split with scaled features and explicit bias handling.
- Segmented MLP: two additional 700-epoch models (low-price and high-price regimes) using the same core training routine.
- Bayesian model: MCMC setup selected to balance posterior stability and runtime.
- Overall workflow: lecture-inspired foundations, then adapted to the Steam pricing problem through regime-aware architecture and feature decisions.
"""
        )

    with appendix_tab2:
        st.write(
            "This is the quick practical decision guide for when to prefer an MLP neural network versus a Bayesian nonlinear approach. "
            "The simplest way to think about it is: MLPs are usually better when your main goal is predictive accuracy at scale, "
            "while Bayesian nonlinear methods are stronger when uncertainty quantification and interpretability are core requirements."
        )

        use_case_df = pd.DataFrame(
            [
                {
                    "Typical use case": "Large, noisy datasets with complex interactions",
                    "MLP NN": "Usually strong fit because it learns nonlinear feature combinations efficiently.",
                    "Bayesian nonlinear": "Can work, but often heavier computationally.",
                },
                {
                    "Typical use case": "Need calibrated uncertainty for decision risk",
                    "MLP NN": "Requires extra calibration steps or ensembling.",
                    "Bayesian nonlinear": "Natural choice because posterior distributions are built in.",
                },
                {
                    "Typical use case": "Explainability for feature-direction confidence",
                    "MLP NN": "Possible with SHAP/ablation, but less direct.",
                    "Bayesian nonlinear": "Posterior effects/credible intervals are directly interpretable.",
                },
                {
                    "Typical use case": "Fast deployment and iterative tuning",
                    "MLP NN": "Very practical with mini-batch training and modern tooling.",
                    "Bayesian nonlinear": "Often slower to train and tune.",
                },
                {
                    "Typical use case": "Small to medium datasets with class imbalance concerns",
                    "MLP NN": "Can overfit without strong regularization.",
                    "Bayesian nonlinear": "Often more robust with priors and uncertainty-aware diagnostics.",
                },
            ]
        )
        st.dataframe(use_case_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
**How this maps to your Steam project**
- The segmented MLP is a strong performance-oriented model for nonlinear pricing patterns.
- The Bayesian tier model gives the uncertainty and interpretation layer needed for research storytelling and poster discussion.
- Using both is a practical hybrid strategy: MLP for prediction strength, Bayesian modeling for uncertainty-aware explanation.
"""
        )

with examples_tab:
    st.subheader("Five Dataset Samples from steam_clean_v2.csv")
    st.write(
        "This tab brings back the focused sample view. "
        "We select five rows from price extremes, requiring both `positive > 0` and `negative > 0`, "
        "so each example includes meaningful mixed review signal."
    )

    examples_df = select_price_extreme_examples(df, total_examples=5)
    if examples_df.empty:
        st.warning("No rows matched the sample filter (`price` valid, `positive > 0`, `negative > 0`).")
        st.stop()

    sample_cols = ["dataset_row", "game", "price_segment", "price", "positive", "negative"]
    for col in ["metacritic_score", "peak_ccu", "recommendations", "average_playtime_forever"]:
        if col in examples_df.columns:
            sample_cols.append(col)
    st.dataframe(examples_df[sample_cols], use_container_width=True, hide_index=True)

    st.subheader("Sample Profiles")
    detail_cols = [
        "price",
        "required_age",
        "estimated_owners",
        "peak_ccu",
        "positive",
        "negative",
        "metacritic_score",
        "user_score",
        "recommendations",
        "achievements",
        "average_playtime_forever",
        "language_count",
        "windows",
        "mac",
        "linux",
    ]

    for idx, row in examples_df.iterrows():
        label = f"{row['game']} ({row['price_segment']})"
        with st.expander(label):
            st.write(
                f"Row `{int(row['dataset_row'])}` in `steam_clean_v2.csv` with log-price `{format_feature_value(row.get('price'))}`. "
                f"This sample has positive reviews `{format_feature_value(row.get('positive'))}` and negative reviews `{format_feature_value(row.get('negative'))}`."
            )
            st.dataframe(build_game_profile_table(row, detail_cols), use_container_width=True, hide_index=True)

with future_tab:
    st.subheader("Future Enhancements from Lecture Ideas + Code Walkthrough")
    st.write(
        "This section is designed as a direct continuation of `Dataset Game Examples`: "
        "we first grounded the analysis in real rows from the table, then moved into what those rows imply for the next model iteration."
    )
    st.write(
        "At a high level, the current pipeline is a strong proof of concept. "
        "The workflow already includes both neural-network and Bayesian modeling, and it already captures one key market insight: "
        "Steam pricing behaves like a multi-regime system, where low-price, high-price, and premium behavior can diverge."
    )
    st.write(
        "From a lecture-aligned perspective, the next improvements should focus on three concrete priorities: "
        "regularization control, optimization strategy, and uncertainty communication."
    )
    st.write(
        "A good first improvement is to tune regularization more systematically. "
        "L2 is already integrated in the manual MLP training loop, and that is exactly the mechanism that helps prevent unstable growth in weights and overfitting in noisy feature spaces."
    )

    st.markdown("**Python snippet: L2 regularization in the manual MLP (Model 1 workflow)**")
    st.code(
        """def train_manual_mlp_regressor(
    x_train, y_train, x_val, y_val,
    hidden1=128, hidden2=64, lr=1e-3, l2=1e-4,
    epochs=800, batch_size=256, patience=40, seed=42
):
    ...
    gW3 = a2.T @ dy + l2 * params["W3"]
    gW2 = a1.T @ dz2 + l2 * params["W2"]
    gW1 = xb.T @ dz1 + l2 * params["W1"]
    ...""",
        language="python",
    )
    st.write(
        "This is the exact lecture pattern we usually want: regularize the weight matrices directly in each gradient step. "
        "A practical next step is a compact `l2` sweep (for example `1e-5`, `1e-4`, `1e-3`) and side-by-side validation RMSE reporting by segment."
    )
    st.write(
        "A second enhancement area is optimization style. "
        "The full-batch setup has strong educational value and clear reproducibility; "
        "the natural extension is mini-batch training with the same architecture and segmentation logic. "
        "That shift usually improves iteration speed and can improve generalization while preserving explainability."
    )
    st.write(
        "A third enhancement area is diagnostic depth. "
        "Instead of reporting one aggregate metric only, we can report segmented error behavior, calibration behavior, and confidence behavior together. "
        "That creates a tighter narrative between data examples, model design, and final interpretation."
    )

    st.markdown("**R snippet: Bayesian + multinomial structure in Model 2**")
    st.code(
        """library(MCMCpack)
library(nnet)

bayes_model <- MCMCmnl(
  price_tier ~ peak_ccu + recommendations + positive + negative +
    metacritic_score + achievements + average_playtime_forever + language_count,
  data = train_df,
  mcmc = 5000,
  burnin = 1000,
  thin = 5,
  verbose = 0
)

multinom_model <- multinom(
  price_tier ~ peak_ccu + recommendations + positive + negative +
    metacritic_score + achievements + average_playtime_forever + language_count,
  data = train_df,
  trace = FALSE
)""",
        language="r",
    )
    st.write(
        "This creates a useful two-layer R strategy: posterior insight from `MCMCmnl` and clean held-out prediction behavior from `multinom`. "
        "From a lecture standpoint, this combination is ideal when both interpretability and practical test evaluation are needed."
    )
    st.write(
        "This is especially valuable in High and Premium tiers, where class counts are smaller and behavior is harder to model. "
        "Point estimates alone can hide uncertainty in those classes, while posterior summaries make confidence gaps explicit and interpretable."
    )

    st.markdown("**How this connects back to Dataset Game Examples**")
    st.markdown(
        """
- In the sample rows, some lower-priced games still show strong engagement or recommendation counts.
- In premium rows, quality and playtime patterns can look noisier and less frequent.
- That mismatch is exactly why one global model can underperform and why segmented MLP + Bayesian tier interpretation is a strong hybrid strategy.
"""
    )
    st.write(
        "That connection matters for storytelling quality. "
        "When example-level evidence and model-level diagnostics align, conclusions are easier to defend and easier to present."
    )
    st.write(
        "A clear way to structure the enhancement roadmap is to separate near-term changes from medium-term research changes."
    )
    st.markdown(
        """
**Near-term implementation upgrades**
- Add mini-batch training while preserving current architecture and segmented split.
- Add per-segment early-stopping plots and validation tracking.
- Add compact hyperparameter sweeps for learning rate, hidden width, and L2 weight.
- Add an explicit model-card summary with assumptions and known failure zones.
"""
    )
    st.markdown(
        """
**Medium-term research upgrades**
- Explore richer feature blocks (developer/publisher effects, release-window context, and genre interactions).
- Test imbalance-aware objectives for expensive tiers (class-weighted loss, focal variants, and targeted resampling).
- Add uncertainty comparison panels across neural and Bayesian pipelines.
- Add sensitivity checks for threshold choices in segmented modeling.
"""
    )
    st.markdown(
        """
**Longer-term integration upgrades**
- Create one reproducible evaluation dashboard that includes:
  - aggregate regression metrics,
  - segmented regression metrics,
  - tier confusion patterns,
  - calibration and confidence overlays.
- Add experiment logging so each run records data version, feature set, and parameter configuration.
"""
    )

    st.markdown(
        """
**Concrete next enhancements**
- Move full-batch training to mini-batch updates (same architecture, faster convergence checks).
- Add early-stopping dashboards by segment (low-price and high-price models shown separately).
- Expand Bayesian reporting with class-wise posterior uncertainty summaries for High and Premium tiers.
- Add calibration/uncertainty plots to compare neural confidence versus Bayesian confidence in one panel.
"""
    )
    st.write(
        "A final presentation enhancement is a side-by-side visualization block: "
        "left panel for neural metrics by segment, right panel for Bayesian confidence by class. "
        "That layout makes the improvement path visually explicit and ties examples, modeling choices, and future roadmap into one coherent narrative."
    )
    st.write(
        "Overall, the enhancement strategy is not to replace the existing models, but to tighten them. "
        "The manual segmented MLP remains the core performance engine, and the Bayesian tier model remains the uncertainty-and-interpretation layer. "
        "The next phase is to make both components more robust, better documented, and easier to compare in a single evaluation view."
    )
