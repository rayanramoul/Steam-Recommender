import streamlit as st
from src.db import SteamDB

st.title("Hello World")
steam_db = SteamDB("data/games.csv")

query = st.text_input("Search", "action")
if st.button("Search"):
    results = steam_db.search_similar(query)
    st.write(results)
