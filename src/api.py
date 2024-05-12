from fastapi import FastAPI
from src.db import SteamDB

app = FastAPI()

steam_db = SteamDB("data/games.csv")

@app.get("/search")
def search(query: str, num_results: int = 5):
    results = steam_db.search_similar(query, num_results)
    return results.to_dict(orient="records")
