# Variables
APP_FILE = app.py
VENV_NAME = .venv

# Install dependencies

install:
	poetry install

# Run the Streamlit app
run:
	poetry run streamlit run $(APP_FILE)
