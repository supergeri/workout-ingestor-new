# Workout Ingestor API

A FastAPI service that converts workout text, images, and videos into structured data (JSON, FIT, or TCX) compatible with Garmin and other fitness platforms.

## 🚀 Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload