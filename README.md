# Koa-Chatbot
Designed and implemented Koa, a machine-learning chat assistant featuring a FastAPI inference backend, configurable JSON-based system prompts, and a responsive light/dark-mode web UI with profile controls and voice input, delivering fast, streaming-style conversational responses and a clean path to web deployment.

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

- [Watch the demo video](https://youtu.be/QX3a01UH3ik)

## UI Screenshot

<img src="assets/Koa_Demo.png" alt="Koa UI" width="600" />


## Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/kokoc30/Koa-Chatbot/blob/main/notebooks/koa_colab_demo.ipynb
)

## Prerequisites
- Python 3.10+ (recommended: 3.11)
- (Optional) NVIDIA GPU + CUDA for faster inference


## Quick Start (Local)
### 1) Backend (FastAPI)
```bash
pip install -r requirements.txt
python -m uvicorn inference.api_server:app --host 127.0.0.1 --port 9010 --reload
