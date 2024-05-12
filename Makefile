# Variables
APP_FILE = src/api.py
FRONT_FILE = src-front/app.py
VENV_NAME = .venv

# Install dependencies

install:
	poetry install

# Run the Streamlit app
run-back:
	poetry run fastapi run $(APP_FILE)

run-front:
	poetry run flet run $(FRONT_FILE) --web -p 9000
