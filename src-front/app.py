import flet as ft
import requests
from loguru import logger
import pandas as pd
import os

def headers(df : pd.DataFrame) -> list:
    return [ft.DataColumn(ft.Text(header)) for header in df.columns]


def rows(df : pd.DataFrame) -> list:
    rows = []
    for index, row in df.iterrows():
        rows.append(ft.DataRow(cells = [ft.DataCell(ft.Text(row[header])) for header in df.columns]))
    return rows

def main(page):
    page.title = "Steam Game Recommender"

    intro_text = ft.Text("Welcome to the Steam Game Recommender! Enter a query to search for games.")
    query = ft.TextField(label="Search", hint_text="Enter a query")
    num_results = ft.Slider(min=5, max=1000, divisions=995, value=5, label="Number of results")
    results = ft.DataTable(columns=[], rows=[])

    def search_games(e):
        global results
        api_url = os.environ.get("API_URL", "http://localhost:8000")
        url = f"{api_url}/search"
        params = {"query": query.value, "num_results": num_results.value}
        logger.debug(f"Query: {params}")
        response = requests.get(url, params=params)
        data = response.json()
        data = pd.DataFrame(data)
        logger.debug(f"Df from response: {data}")
        
        results = ft.DataTable(columns=headers(data), rows=rows(data))
        page.add(results)

    search_button = ft.ElevatedButton("Search", on_click=search_games)

    page.add(query, num_results, search_button, results)

ft.app(target=main)
