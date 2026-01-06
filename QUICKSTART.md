# 🚀 Quick Start Guide

## How to Run the Multimodal Video Transcription System

### Prerequisites Checklist

Before running, ensure you have:

- ✅ Python 3.9 or higher installed
- ✅ FFmpeg installed and in PATH
- ✅ All API keys ready (Sarvam, Gemini)
- ✅ Virtual environment created (recommended)

---

## Step 1: Install Dependencies

### Option A: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Direct Installation

```bash
pip install -r requirements.txt
```

**Note:** This may take 5-10 minutes as it downloads large ML models.

---

## Step 2: Configure Environment Variables

1. **Copy the template file:**
   ```bash
   # On Windows:
   copy env.template .env
   
   # On Linux/Mac:
   cp env.template .env
   ```

2. **Edit `.env` file** and add your API keys:
   ```env
   SARVAM_API_KEY=your_actual_sarvam_key_here
   GEMINI_API_KEY=your_actual_gemini_key_here
   ```
   
   **Note:** `HF_TOKEN` and `GROQ_API_KEY` are no longer required.

3. **Get your API keys:**
   - **Sarvam API Key**: https://dashboard.sarvam.ai/signin
   - **Gemini API Key**: https://makersuite.google.com/app/apikey
   
   **Note:** Speaker diarization now uses free clustering (no Hugging Face token needed!)

---

## Step 3: Start the Server

### Method 1: Direct Python Execution

```bash
python -m app.main
```

### Method 2: Using Uvicorn Directly

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Expected Output

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

The server is now running at **http://localhost:8000**

---

## Step 4: Test the API

### Option 1: Using the Interactive API Documentation (Easiest)

1. Open your browser and go to: **http://localhost:8000/docs**
2. You'll see the Swagger UI with all available endpoints
3. Click on `/transcribe` endpoint
4. Click "Try it out"
5. Click "Choose File" and select your video file
6. Click "Execute"
7. Wait for the response (this may take several minutes for long videos)

### Option 2: Using cURL (Command Line)

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Transcribe Video (JSON Response)
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "video=@path/to/your/video.mp4"
```

#### Transcribe Video (Text Response)
```bash
curl -X POST "http://localhost:8000/transcribe/text" \
  -F "video=@path/to/your/video.mp4"
```

### Option 3: Using Python Requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Transcribe video
with open("your_video.mp4", "rb") as video_file:
    files = {"video": video_file}
    response = requests.post(
        "http://localhost:8000/transcribe",
        files=files
    )
    result = response.json()
    print(result)
```

### Option 4: Using Postman

1. Open Postman
2. Create a new POST request to: `http://localhost:8000/transcribe`
3. Go to "Body" tab
4. Select "form-data"
5. Add key: `video` (type: File)
6. Select your video file
7. Click "Send"

---

## 📥 Input Requirements

### Supported Video Formats

- ✅ **MP4** (recommended)
- ✅ **AVI**
- ✅ **MOV**
- ✅ **MKV**
- ✅ **WebM**

### Video Requirements

- **Format**: Any video format supported by FFmpeg
- **Audio**: Must contain audio track
- **Duration**: No strict limit, but longer videos take more time
- **Resolution**: Any resolution (will be processed as-is)
- **File Size**: No strict limit, but larger files take longer to upload

### Recommended Video Specifications

- **Format**: MP4 (H.264 codec)
- **Audio**: AAC or MP3
- **Duration**: 1-30 minutes (optimal)
- **Resolution**: 720p or 1080p
- **Frame Rate**: 24-30 FPS

---

## 📤 Output Format

### JSON Response (`/transcribe`)

```json
{
  "transcript": [
    {
      "timestamp": "00:01:23",
      "speaker": "SPEAKER_00",
      "text": "I'm really excited about this project.",
      "emotion": "Joy",
      "tone": "Enthusiastic",
      "reaction": "Positive engagement",
      "audio_confidence": 0.92,
      "visual_confidence": 0.88
    },
    {
      "timestamp": "00:01:45",
      "speaker": "SPEAKER_01",
      "text": "That sounds great! When can we start?",
      "emotion": "Neutral",
      "tone": "Inquisitive",
      "reaction": "Question",
      "audio_confidence": 0.85,
      "visual_confidence": 0.75
    }
  ],
  "summary": {
    "total_speakers": 2,
    "total_segments": 15,
    "dominant_emotions": ["Joy", "Neutral", "Surprise"],
    "summary": "Conversation summary text..."
  },
  "metadata": {
    "duration": 330.5,
    "total_segments": 15,
    "video_path": "/tmp/video_xyz.mp4"
  }
}
```

### Text Response (`/transcribe/text`)

```
[00:01:23] SPEAKER_00: "I'm really excited about this project."
[Emotion: Joy, Tone: Enthusiastic, Reaction: Positive engagement]

[00:01:45] SPEAKER_01: "That sounds great! When can we start?"
[Emotion: Neutral, Tone: Inquisitive, Reaction: Question]
```

---

## ⏱️ Processing Time

Processing time depends on video length:

- **1 minute video**: ~2-5 minutes
- **5 minute video**: ~10-25 minutes
- **10 minute video**: ~20-50 minutes

**Real-time factor**: Approximately 2-5x (5 min video = 10-25 min processing)

Factors affecting speed:
- Video length
- Number of speakers
- Hardware (CPU/GPU)
- API response times (Sarvam, Gemini)

---

## 🔍 Available Endpoints

### 1. `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "Video transcription service is running"
}
```

### 2. `POST /transcribe`
Transcribe video and return JSON with enriched metadata.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `video` (file)

**Response:** JSON object with transcript, emotions, and metadata

### 3. `POST /transcribe/text`
Transcribe video and return formatted text.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `video` (file)

**Response:** Plain text transcript

### 4. `GET /`
API information endpoint.

### 5. `GET /docs`
Interactive API documentation (Swagger UI).

---

## 🐛 Troubleshooting

### Server won't start

**Error: "Module not found"**
```bash
# Make sure you're in the project directory
cd E:\personal\video_trancibe

# Reinstall dependencies
pip install -r requirements.txt
```

**Error: "Port already in use"**
```bash
# Change port in .env file
PORT=8001

# Or kill the process using port 8000
# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### API Key Errors

**Error: "SARVAM_API_KEY not set"**
- Check your `.env` file exists
- Verify the key is correct (no extra spaces)
- Restart the server after updating `.env`

**Error: "Speaker diarization issues"**
- Diarization uses free clustering and works offline
- No external API keys or model downloads needed
- Works best with 2-4 speakers

### Processing Errors

**Error: "FFmpeg not found"**
```bash
# Install FFmpeg
# Windows: Download from https://ffmpeg.org/download.html
# Linux: sudo apt-get install ffmpeg
# Mac: brew install ffmpeg

# Verify installation
ffmpeg -version
```

**Error: "No face detected"**
- This is normal if faces aren't visible
- Audio emotion detection will still work
- Processing will continue

---

## 📝 Example Workflow

1. **Start the server:**
   ```bash
   python -m app.main
   ```

2. **Open browser:** http://localhost:8000/docs

3. **Upload video:**
   - Click on `/transcribe` endpoint
   - Click "Try it out"
   - Select your video file
   - Click "Execute"

4. **Wait for processing** (progress shown in server logs)

5. **Get results:**
   - View JSON response in Swagger UI
   - Or copy the response for further processing

---

## 🎯 Next Steps

- Check the full [README.md](README.md) for detailed documentation
- Explore the code in `app/` directory
- Customize models and parameters
- Integrate with your own applications

---

## 💡 Tips

1. **Start with short videos** (1-2 minutes) to test the setup
2. **Use MP4 format** for best compatibility
3. **Check server logs** for detailed processing information
4. **Monitor API quotas** (Sarvam and Gemini have rate limits - Gemini: 5 req/min)
5. **Use GPU** if available for faster processing

---

**Need Help?** Check the [README.md](README.md) for more detailed information.
