import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dataset Game Examples", page_icon="📊", layout="wide")
st.title("Dataset-Based Example Games")
st.write(
    """
This page uses only rows from `steam_clean.csv`.
The five examples below are selected from your dataset by strongest review signal.
"""
)


@st.cache_data
def load_data():
    return pd.read_csv("data/steam_clean.csv")


def make_game_label(row_index, row):
    possible_name_columns = ["name", "Name", "title", "Title", "game", "Game"]
    for col in possible_name_columns:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    if "appID" in row.index and pd.notna(row["appID"]):
        return f"App {int(row['appID'])}"
    return f"Dataset Game #{row_index + 1}"


def format_value(value):
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def build_description(row):
    platforms = []
    if "Windows" in row.index and float(row["Windows"]) > 0:
        platforms.append("Windows")
    if "Mac" in row.index and float(row["Mac"]) > 0:
        platforms.append("Mac")
    if "Linux" in row.index and float(row["Linux"]) > 0:
        platforms.append("Linux")
    platform_text = ", ".join(platforms) if platforms else "unknown platforms"

    return (
        "This game has strong review/engagement signal in the cleaned dataset, "
        f"with Positive={format_value(row.get('Positive', 'N/A'))}, "
        f"Negative={format_value(row.get('Negative', 'N/A'))}, "
        f"Peak CCU={format_value(row.get('Peak CCU', 'N/A'))}, and "
        f"Average playtime forever={format_value(row.get('Average playtime forever', 'N/A'))}. "
        f"It is tagged as supporting: {platform_text}. "
        "Values shown are from the transformed modeling table (`steam_clean.csv`)."
    )


df = load_data()

if not {"Positive", "Negative"}.issubset(set(df.columns)):
    st.error("`steam_clean.csv` must contain `Positive` and `Negative` to rank well-received games.")
    st.stop()

ranked_df = df.copy()
ranked_df["review_signal"] = ranked_df["Positive"] - ranked_df["Negative"]

sort_cols = ["review_signal"]
if "Metacritic score" in ranked_df.columns:
    sort_cols.append("Metacritic score")
if "User score" in ranked_df.columns:
    sort_cols.append("User score")

top_games = ranked_df.sort_values(by=sort_cols, ascending=False).head(5).copy()
top_games["Dataset Row"] = top_games.index + 1
top_games["Game"] = [
    make_game_label(idx, top_games.loc[idx]) for idx in top_games.index
]

display_columns = ["Dataset Row", "Game", "review_signal"]
for col in ["Price", "Positive", "Negative", "Metacritic score", "Peak CCU"]:
    if col in top_games.columns:
        display_columns.append(col)

st.subheader("Top 5 Well-Received Games In Your Dataset")
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
        st.write(build_description(row))
        feature_rows = []
        for col in detail_columns:
            if col in top_games.columns:
                feature_rows.append({"Feature": col, "Value": format_value(row[col])})
        st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True)
