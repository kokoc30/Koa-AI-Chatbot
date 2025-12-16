# Koa-Chatbot
Designed and implemented Koa, a machine-learning chat assistant with a FastAPI backend, custom JSON system prompts, and a responsive light/dark-mode web UI featuring profile controls and voice input, built for fast, streaming conversational responses and web deployment.



Koa is a lightweight machine-learning chat assistant with a FastAPI backend and a responsive web UI (light/dark mode) featuring profile controls and voice input. It is built for fast, streaming conversational responses and straightforward web deployment.

## Features
- FastAPI inference server with a simple chat API
- Streaming-friendly chat flow (designed for responsive UX)
- Responsive frontend with light/dark mode
- Voice input support in the web UI
- JSONL-based sample data for quick testing

## Repo Structure
- `frontend/` — Web UI (HTML/CSS/JS)
- `inference/` — FastAPI server + chat logic
- `data/` — Small sample data (JSONL)
- `requirements.txt` — Python dependencies

## Demo
Add your demo video here:

> **Demo video:** (upload a short MP4 to the repo or link it)
- Option A (recommended): Upload a GIF preview to `assets/demo.gif` and embed it:
  - `![Koa demo](assets/demo.gif)`
- Option B: Upload an MP4 and link it:
  - `[Watch the demo video](assets/koa-demo.mp4)`
- Option C: Link YouTube / Google Drive:
  - `[Watch the demo video](PASTE_LINK_HERE)`

## Quick Start (Local)
### 1) Backend (FastAPI)
```bash
pip install -r requirements.txt
python inference/api_server.py
