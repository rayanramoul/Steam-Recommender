## SteamDB-RAG

FastAPI service that recommends Steam games using a vector database over `data/games.csv`. Optional Flet web UI is included.

### Prerequisites
- Python 3.10+
- `uv` package manager (`pip install uv` or see `https://docs.astral.sh/uv/`)

### Setup and Run (uv)
```bash
# Install dependencies
uv sync

# Run backend API
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000

# Optional: run Streamlit UI (connects to API at http://localhost:8000)
API_URL=http://localhost:8000 uv run streamlit run src-front/streamlit_app.py --server.port 9000 --server.headless true
```

Open API docs at `http://localhost:8000/docs`.

Example request:
```bash
curl "http://localhost:8000/search?query=roguelike&num_results=5"
```

### Docker
Build and run the backend (mounts local `data/` so the API reads `data/games.csv`):
```bash
docker build -t steam-recommender:latest .
docker run --rm -p 8000:8000 -v $(pwd)/data:/app/data steam-recommender:latest
```

Open API docs at `http://localhost:8000/docs`.

### Docker Compose (API + Streamlit UI)
```bash
docker compose up --build
```
Then open the UI at `http://localhost:9000`.

#### Use NVIDIA GPU (optional)
- Install NVIDIA Container Toolkit and ensure `docker run --gpus all nvidia/cuda:12.2.0-base nvidia-smi` works.
- Start with GPU enabled (uses `Dockerfile.gpu` and `gpus: all`):
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### Makefile shortcuts
```bash
make install       # uv sync
make run-back      # start API (uvicorn)
make run-front     # start Flet web UI
make docker-build  # build image
make docker-run    # run container
make compose-up    # docker compose up --build
make compose-down  # docker compose down
```

### Notes
- The database is built at startup from `data/games.csv` and persisted in `data/chroma`.
- CPU-only Torch is used by default; GPU is auto-detected if available.
 - If you see `attempt to write a readonly database`, fix permissions or change the location with `CHROMA_DIR`:
```bash
mkdir -p data/chroma && chmod -R u+rwX data/chroma
CHROMA_DIR=~/.cache/steam-recommender/chroma uv run uvicorn src.api:app --host 0.0.0.0 --port 8000
```
