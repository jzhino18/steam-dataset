import pandas as pd
import streamlit as st

st.set_page_config(page_title="Example Games", page_icon="🎮", layout="wide")
st.title("Example Games and Descriptions")
st.write(
    """
This page highlights five well-received Steam games with short descriptions.
These are representative examples from the broader Steam catalog used in your project context.
"""
)

st.info(
    "Note: `steam_clean.csv` is a cleaned modeling table and does not include game title columns. "
    "So this page shows researched example titles and links from Steam store pages."
)

games = [
    {
        "Game": "Hades",
        "Reception": "Overwhelmingly Positive (98% of 138,694 reviews)",
        "Description": "A fast-paced roguelike dungeon crawler where Zagreus battles out of the Underworld using different weapons and Olympian boons.",
        "Steam URL": "https://store.steampowered.com/app/1145360/Hades/",
    },
    {
        "Game": "Portal 2",
        "Reception": "Overwhelmingly Positive (98% of 168,294 reviews)",
        "Description": "A first-person puzzle game centered on portal mechanics, combining physics puzzles, story-driven single-player, and a separate co-op campaign.",
        "Steam URL": "https://store.steampowered.com/app/620/Portal_2/",
    },
    {
        "Game": "Stardew Valley",
        "Reception": "Overwhelmingly Positive (98% of 381,171 reviews)",
        "Description": "A farming and life-sim RPG where you restore an inherited farm, build relationships in town, and explore caves, crafting, and seasonal events.",
        "Steam URL": "https://store.steampowered.com/app/413150/Stardew_Valley/",
    },
    {
        "Game": "Terraria",
        "Reception": "Overwhelmingly Positive (97% of 586,304 reviews)",
        "Description": "A 2D sandbox adventure with exploration, crafting, building, and boss combat across a world shaped by player progression.",
        "Steam URL": "https://store.steampowered.com/app/105600/Terraria/",
    },
    {
        "Game": "Hollow Knight",
        "Reception": "Overwhelmingly Positive (97% of 173,360 reviews)",
        "Description": "A challenging atmospheric metroidvania in the ruined kingdom of Hallownest, focused on exploration, precision combat, and discovery.",
        "Steam URL": "https://store.steampowered.com/app/367520/Hollow_Knight/",
    },
]

games_df = pd.DataFrame(games)
st.subheader("Five Well-Received Steam Games")
st.dataframe(games_df, use_container_width=True, hide_index=True)

st.subheader("Game Details")
for game in games:
    with st.expander(game["Game"]):
        st.write(f"**Steam reception:** {game['Reception']}")
        st.write(f"**Description:** {game['Description']}")
        st.markdown(f"**Source:** [Steam Store Page]({game['Steam URL']})")
