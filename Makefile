# Variables
APP_FILE = src/api.py
FRONT_FILE = src-front/streamlit_app.py

# Install dependencies with uv
install:
	uv sync

# Run the FastAPI backend
run-back:
	uv run uvicorn src.api:app --host 0.0.0.0 --port 8000

# Run the Streamlit frontend
run-front:
	API_URL=http://localhost:8000 uv run streamlit run $(FRONT_FILE) --server.port 9000 --server.headless true

# Run backend and frontend together (local)
run:
	uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 & \
	API_URL=http://localhost:8000 uv run streamlit run $(FRONT_FILE) --server.port 9000 --server.headless true

# Docker
docker-build:
	docker build -t steam-recommender:latest .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/data:/app/data steam-recommender:latest

compose-up:
	docker compose up --build

compose-down:
	docker compose down

# Docker Compose (GPU override)
compose-up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

compose-down-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
