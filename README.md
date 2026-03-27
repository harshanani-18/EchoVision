# 🎓 EchoVision — AI-Powered Teaching Visualizer

**EchoVision** is a real-time AI-powered lecture analysis platform that listens to live lectures, transcribes speech, classifies content intelligently, and generates visual aids — all in real time.

It captures audio from the instructor's microphone, transcribes it using **Deepgram**, classifies each segment using **Google Gemini 2.5 Flash**, and generates educational images and videos from core concepts using **Gemini Image Generation** and **Veo 3**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙️ **Real-Time Transcription** | Live speech-to-text using Deepgram Nova-2 via WebSocket |
| 🧠 **AI Content Classification** | Classifies speech into **Filler**, **Administration**, and **Visual Concepts** using Gemini 2.5 Flash |
| 🖼️ **Auto Image Generation** | Generates educational diagrams from accumulated visual concepts via Gemini Image Generation |
| 🎬 **Video Generation** | Creates short educational animations from concepts using Google Veo 3 |
| 📊 **Lecture Summary** | AI-generated structured summary with teaching score, key concepts, and improvement suggestions |
| 📈 **Live Statistics** | Real-time dashboard showing filler %, admin %, and concept distribution |
| 🔄 **Buffered Batch Processing** | Accumulates segments for context-aware batch classification (reduces API calls) |
| 💓 **WebSocket Stability** | Keepalive pings, heartbeat, and graceful disconnect handling |

---

## 🏗️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Real-Time Audio:** Deepgram SDK (Nova-2 model)
- **AI Classification:** Google Gemini 2.5 Flash (`google-generativeai`)
- **Image Generation:** Gemini Image Generation (`google-genai`)
- **Video Generation:** Google Veo 3 (`google-genai`)
- **Frontend:** HTML, CSS, JavaScript (Jinja2 templates)
- **WebSocket:** FastAPI WebSocket for bidirectional audio/data streaming

---

## 📁 Project Structure

```
EchoVision/
├── main.py                    # FastAPI app — routes, WebSocket, API endpoints
├── transcription_analyzer.py  # AI classification, image/video generation, summarization
├── requirements.txt           # Python dependencies
├── .env                       # API keys (Deepgram + Google)
├── .gitignore                 # Git ignore rules
├── templates/
│   └── index.html             # Frontend UI (single-page app)
├── static/                    # Static assets
└── tests/                     # Test files
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **Deepgram API Key** — [Get one here](https://console.deepgram.com/)
- **Google API Key** (Gemini) — [Get one here](https://aistudio.google.com/apikey)

### 1. Clone the Repository

```bash
git clone https://github.com/harshanani-18/EchoVision.git
cd EchoVision
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
DEEPGRAM_API_KEY=your_deepgram_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Run the Application

```bash
uvicorn main:app --reload
```

The app will be available at **http://127.0.0.1:8000**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main UI page |
| `GET` | `/api/analysis` | Full transcription analysis with classifications |
| `GET` | `/api/stats` | Quick statistics (filler %, admin %, concepts %) |
| `GET` | `/api/buffer_status` | Current buffer and visual concepts status |
| `GET` | `/api/summary` | AI-generated lecture summary |
| `POST` | `/api/generate_video` | Generate educational video from visual concepts |
| `WS` | `/listen` | WebSocket endpoint for real-time audio streaming |

---

## 🔧 How It Works

1. **Audio Capture** — The browser captures microphone audio and streams it via WebSocket to the FastAPI backend.
2. **Transcription** — Audio chunks are forwarded to Deepgram's Nova-2 model for real-time speech-to-text.
3. **Buffered Classification** — Transcripts are accumulated in a buffer (default: 5 segments) for context-aware batch classification via Gemini 2.5 Flash.
4. **Visual Concept Accumulation** — Segments classified as `VISUAL_CONCEPT` are collected. Once enough accumulate (default: 3), an educational image is auto-generated.
5. **Live Dashboard** — All results stream back to the frontend in real time via WebSocket, updating the live transcript, classification tags, statistics, and generated images.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👥 Authors

- **harshanani-18** — [GitHub Profile](https://github.com/harshanani-18)
