import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Steam Game Recommender", page_icon="🎮", layout="wide")

API_URL = os.environ.get("API_URL", "http://localhost:8000")


@st.cache_data(show_spinner=False, ttl=60)
def search_games(query: str, num_results: int = 5) -> pd.DataFrame:
    params = {"query": query, "num_results": num_results}
    resp = requests.get(f"{API_URL}/search", params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Normalize genres display (CSV may contain a semicolon/comma-separated string)
    if "genres" in df.columns:
        def fmt_genres(x):
            try:
                if isinstance(x, str):
                    # Try to split on comma or semicolon
                    parts = [p.strip() for p in x.replace(";", ",").split(",") if p.strip()]
                    return ", ".join(parts)
                if isinstance(x, list):
                    return ", ".join(map(str, x))
                return str(x)
            except Exception:
                return str(x)
        df["genres"] = df["genres"].apply(fmt_genres)
    return df


def pill(text: str, color: str = "#1f6feb") -> str:
    return f"<span style='background:{color};padding:4px 10px;border-radius:999px;color:white;font-size:12px;margin-right:6px;display:inline-block'>{text}</span>"


st.markdown("""
<div style='display:flex;align-items:center;gap:12px;'>
  <h1 style='margin:0'>🎮 Steam Game Recommender</h1>
</div>
<p style='color:#6b7280;margin-top:4px'>Search similar games from embeddings built over your <code>data/games.csv</code>.</p>
""", unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns([4, 2, 1])
    with col1:
        query = st.text_input("Search", placeholder="e.g., roguelike deckbuilder with progression")
    with col2:
        num_results = st.slider("Results", min_value=5, max_value=50, value=10, step=5)
    with col3:
        st.write("")
        st.write("")
        do_search = st.button("🔎 Search", use_container_width=True)

st.caption(f"Backend: {API_URL}")

if do_search and query.strip():
    with st.spinner("Finding similar games..."):
        try:
            df = search_games(query.strip(), num_results)
        except Exception as e:
            st.error(f"Request failed: {e}")
            df = pd.DataFrame()

    if df.empty:
        st.info("No results found.")
    else:
        # Nice summary cards for the top 3
        top = df.head(3)
        for _, row in top.iterrows():
            with st.container():
                st.markdown("---")
                title = str(row.get("game", "Unknown"))
                about = str(row.get("about", ""))[:400]
                score = row.get("score", "-")
                genres = str(row.get("genres", "")).split(", ") if isinstance(row.get("genres"), str) else []
                image = row.get("image")
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.subheader(title)
                    st.write(about)
                with c2:
                    st.metric("User score", score)
                    if genres:
                        st.markdown(" ".join(pill(g) for g in genres[:8]), unsafe_allow_html=True)
                    if isinstance(image, str) and image:
                        st.image(image, use_column_width=True)

        st.markdown("---")
        st.subheader("All results")
        show_cols = [c for c in ["game", "genres", "score", "about"] if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)


