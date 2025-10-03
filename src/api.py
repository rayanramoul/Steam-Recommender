from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from src.db import SteamDB

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_csv = os.environ.get("DATA_CSV", "data/games.csv")
steam_db = SteamDB(data_csv)

@app.get("/search")
def search(query: str, num_results: int = 5):
    results = steam_db.search_similar(query, num_results)
    return results.to_dict(orient="records")
