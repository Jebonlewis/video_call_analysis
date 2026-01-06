# Multimodal Video Transcription System

An end-to-end production-ready system that transcribes video call conversations and enriches transcripts with metadata including emotions, delivery nuances, and reactions.

## 🎯 Overview

This system processes recorded video calls to generate enriched transcripts that include:
- **Speaker labels** (who said what)
- **Emotions** (detected from audio prosody and facial expressions)
- **Tone & delivery** (enthusiasm, pace, etc.)
- **Reactions** (responses to questions, engagement level)

## 🏗️ Architecture

The system uses a **hybrid architecture** combining signal-processing models with LLM reasoning:

```
Video Input (.mp4)
    ↓
┌─────────────────────────────────────┐
│ 1. Video Ingestion                  │
│    - Extract audio (FFmpeg)         │
│    - Extract frames (OpenCV, 2 FPS) │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. Audio Processing                 │
│    - Transcription (Sarvam API)     │
│    - Speaker Diarization (clustering)│
│    - Prosody Analysis (librosa)     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. Visual Processing                 │
│    - Facial Emotion (DeepFace)      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. Multimodal Fusion                 │
│    - Align by timestamps             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. LLM Reasoning (Groq/LLaMA-3)      │
│    - Emotion normalization           │
│    - Reaction classification         │
│    - Metadata synthesis              │
└─────────────────────────────────────┘
    ↓
Enriched Transcript (JSON/Text)
```

## 📦 Installation

### Prerequisites

1. **Python 3.9+**
2. **FFmpeg** (must be installed separately)
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

3. **Sarvam API Key** (for speech-to-text)
   - Sign up at [Sarvam AI Dashboard](https://dashboard.sarvam.ai/signin)
   - Get your API key from the dashboard
   - Set as `SARVAM_API_KEY` environment variable

**Note:** Speaker diarization now uses a free clustering-based approach (no external API keys required)

### Setup

1. **Clone or navigate to the project directory**

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
   
   **Option 1: Use the template file (Recommended)**
   ```bash
   # Copy the template file
   cp env.template .env  # On Windows: copy env.template .env
   
   # Then edit .env and fill in your actual API keys
   ```
   
   **Option 2: Create manually**
   Create a `.env` file in the project root with:
   ```env
   SARVAM_API_KEY=your_sarvam_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   HF_TOKEN=your_huggingface_token_here
   ```
   
   **Required Variables:**
   - `SARVAM_API_KEY` - Get from https://dashboard.sarvam.ai/signin
   - `GROQ_API_KEY` - Get from https://console.groq.com/keys
   
   **Note:** `HF_TOKEN` is no longer required - speaker diarization uses free clustering
   
   **Optional Variables:**
   - `SARVAM_MODEL` - Default: `saarika:v2.5`
   - `PORT` - Default: `8000`

5. **No additional model downloads required** - Speaker diarization uses built-in clustering algorithms

## 🚀 Usage

> **📖 For detailed step-by-step instructions, see [QUICKSTART.md](QUICKSTART.md)**

### Start the API Server

```bash
# Make sure you're in the project root directory
python -m app.main
```

The server will start at `http://localhost:8000`

**Alternative method:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Transcribe Video
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "video=@path/to/your/video.mp4"
```

#### 3. API Documentation
Visit `http://localhost:8000/docs` for interactive Swagger UI

### Example Output

```json
{
  "transcript": [
    {
      "timestamp": "00:01:23",
      "speaker": "Speaker 1",
      "text": "I'm really excited about this project.",
      "emotion": "Joy",
      "tone": "Enthusiastic",
      "reaction": "Positive engagement",
      "audio_confidence": 0.92,
      "visual_confidence": 0.88
    }
  ],
  "summary": {
    "total_speakers": 2,
    "duration": "00:05:30",
    "dominant_emotions": ["Joy", "Neutral", "Surprise"]
  }
}
```

## 📁 Project Structure

```
vllm_video_transcription/
│
├── app/
│   ├── ingestion/
│   │   ├── extract_audio.py      # Audio extraction from video
│   │   └── extract_frames.py     # Frame extraction for visual analysis
│   │
│   ├── asr/
│   │   ├── whisper_asr.py        # Whisper transcription
│   │   └── diarization.py       # Speaker diarization
│   │
│   ├── audio_emotion/
│   │   └── prosody.py            # Audio prosody analysis
│   │
│   ├── vision/
│   │   └── facial_emotion.py     # Facial emotion detection
│   │
│   ├── fusion/
│   │   └── align_and_merge.py    # Multimodal data alignment
│   │
│   ├── llm/
│   │   └── groq_reasoning.py     # LLM-based metadata synthesis
│   │
│   ├── pipeline.py               # Main processing pipeline
│   └── main.py                   # FastAPI application
│
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

### Model Selection

- **Sarvam Model**: Default is `saarika:v2.5`. Change via `SARVAM_MODEL` environment variable
  - Options: `saarika:v2.5`, `saarika:v2.0`, or other available models
  - Check [Sarvam AI Documentation](https://docs.sarvam.ai) for latest models

- **Groq Model**: Default is `llama3-70b-8192`. Change in `app/llm/groq_reasoning.py`

### Processing Parameters

- **Frame Rate**: 2 FPS (configurable in `app/ingestion/extract_frames.py`)
- **Audio Sample Rate**: 16kHz (standard for ASR)

## 🔧 Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Ensure FFmpeg is installed and in PATH
   - Test: `ffmpeg -version`

2. **Speaker diarization issues**
   - Diarization uses clustering and works offline
   - If accuracy is low, try specifying number of speakers manually
   - Works best with 2-4 speakers

3. **CUDA/GPU issues**
   - System works on CPU, but GPU speeds up processing
   - Install CUDA-enabled PyTorch if using GPU

4. **DeepFace model download**
   - First run downloads models (~500MB)
   - Ensure stable internet connection

5. **Sarvam API errors**
   - Ensure SARVAM_API_KEY is set correctly
   - Check API rate limits and quota
   - Verify audio format is supported (WAV, MP3)

6. **Memory issues with long videos**
   - Process in chunks (modify pipeline.py)
   - Consider splitting long videos

## ⚠️ Limitations

1. **Processing Time**: 
   - Real-time factor: ~2-5x (5 min video = 10-25 min processing)
   - Depends on hardware and video length

2. **Accuracy**:
   - Speaker diarization works best with 2-4 speakers
   - Emotion detection accuracy: ~70-80%
   - Visual emotion requires clear facial visibility

3. **Supported Formats**:
   - Video: MP4, AVI, MOV (any FFmpeg-supported format)
   - Audio: Extracted automatically

4. **Resource Requirements**:
   - RAM: Minimum 8GB, recommended 16GB+
   - Storage: ~500MB for models (first download)
   - GPU: Optional but recommended

5. **API Rate Limits**:
   - Groq free tier has rate limits
   - Sarvam API has rate limits based on your plan
   - Consider caching for repeated requests

6. **Language Support**:
   - Sarvam API supports multiple Indian languages (Hindi, English, etc.)
   - Check [Sarvam Documentation](https://docs.sarvam.ai) for supported languages
   - Emotion models optimized for English

## 🔐 Security & Privacy

- All processing happens **locally** (except Groq API calls)
- Video files are processed in-memory when possible
- No data is stored permanently (unless you implement storage)
- Use HTTPS in production

## 📝 License

This project uses open-source models and libraries. Check individual licenses:
- Whisper: MIT
- pyannote.audio: MIT
- DeepFace: MIT
- Groq API: Check Groq terms of service

## 🤝 Contributing

This is a production-ready template. Extend as needed:
- Add database storage
- Implement caching
- Add batch processing
- Enhance emotion models
- Add more languages

## 📧 Support

For issues:
1. Check Troubleshooting section
2. Verify all dependencies are installed
3. Check environment variables
4. Review logs for specific errors

---

**Built with**: Sarvam AI, pyannote.audio, DeepFace, Groq, FastAPI
