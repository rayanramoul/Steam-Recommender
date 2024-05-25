import requests
import pandas as pd
import os
from tqdm import tqdm
from loguru import logger
import schedule
import time


def fetch_steam_games(output_path="steam_games.csv"):
    """Fetch all games and their details from Steam API and save to CSV."""
    logger.info("Fetching game list from Steam API...")
    app_list_url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
    response = requests.get(app_list_url)

    app_list = response.json()["applist"]["apps"]
    apps = [{"appid": app["appid"], "name": app["name"]} for app in app_list]
    # shuffle the list of apps
    apps = sorted(apps, key=lambda x: x["appid"])

    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        existing_games_app_ids = existing_df["App ID"].values
    else:
        existing_df = pd.DataFrame()
        existing_games_app_ids = []

    new_games_data = []
    skipped = 0
    scrapped = 0
    logger.info(f"Number of games in Steam: {len(app_list)}")
    for app in tqdm(apps, desc="Fetching game details", unit="games"):
        app_id = app["appid"]
        logger.info(
            f"Scrapped number of games: {scrapped} - Skipped: {skipped} - Existing games: {len(existing_games_app_ids)}"
        )
        # if app already exists in the CSV, skip fetching details
        if app_id in existing_games_app_ids:
            skipped += 1
            continue
        app_details_url = (
            f"https://store.steampowered.com/api/appdetails?appids={app_id}"
        )
        details_response = None
        while not details_response:
            try:
                details_response = requests.get(app_details_url)
            except ConnectionError:
                logger.info("Connection error. Retrying...")
                time.sleep(5)

        if details_response.status_code == 200:
            details = details_response.json().get(str(app_id), {}).get("data", {})
            logger.info(f"Details: {details.keys()}, for game app_id: {app_id}")
            if details:
                logger.info(f"Fetching details for game: {details.get('name')}")
                logger.info(f"Details: {details.keys()}")
                game_info = {
                    "Name": details.get("name"),
                    "About the game": details.get("short_description", ""),
                    "Detailed Descriptions": details.get("detailed_description", ""),
                    "App ID": app_id,
                    "Header Image": details.get("header_image", ""),
                    "User score": details.get("metacritic", {}).get("score", ""),
                    "Genres": details.get("genres", []),
                    "Supported languages": details.get("supported_languages", ""),
                    "Capsule Imagev5": details.get("capsule_imagev5", ""),
                    "Capsule Image": details.get("capsule_image", ""),
                    "Platforms": details.get("platforms", ""),
                    "Fullgame": details.get("fullgame", ""),
                    "Categories": details.get("categories", []),
                    "Price Overview": details.get("price_overview", ""),
                    "Ratings": details.get("recommendations", {}).get("total", ""),
                    "Background Image": details.get("background", ""),
                    "Website": details.get("website", ""),
                    "Screenshots": details.get("screenshots", []),
                    "Developers": details.get("developers", []),
                    "Publishers": details.get("publishers", []),
                    "Release Date": details.get("release_date", {}),
                    "Recommendations": details.get("recommendations", {}),
                    "PC Requirements": details.get("pc_requirements", ""),
                    "Mac Requirements": details.get("mac_requirements", ""),
                    "Linux Requirements": details.get("linux_requirements", ""),
                }
                scrapped += 1
                new_games_data.append(game_info)

                if new_games_data:
                    new_df = pd.DataFrame(new_games_data)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df.to_csv(output_path, index=False)
                    logger.info(
                        f"Appended details of {len(new_df)} new games to {output_path}"
                    )
                else:
                    logger.info("No new games found to append.")
                # time.sleep(3)
        else:
            logger.error(
                f"Failed to fetch details for game: {app.get('name')}, app_id: {app_id}"
            )
            new_games_data.append({"Name": app.get("name"), "App ID": app_id})
            if new_games_data:
                new_df = pd.DataFrame(new_games_data)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_csv(output_path, index=False)
                logger.info(
                    f"Appended details of {len(new_df)} new games to {output_path}"
                )


def job():
    fetch_steam_games("data/steam_games.csv")


# Schedule the job every Sunday
# schedule.every().sunday.at("00:00").do(job)

# logger.info("Scheduler started. The job will run every Sunday at 00:00.")
# while True:
#    schedule.run_pending()
#    time.sleep(1)

if __name__ == "__main__":
    job()
