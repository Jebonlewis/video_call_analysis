# 🔄 Complete Workflow & Tools Explanation

## 📋 Table of Contents
1. [Overview](#overview)
2. [Complete Workflow](#complete-workflow)
3. [Tools & Libraries Used](#tools--libraries-used)
4. [Data Flow](#data-flow)
5. [Step-by-Step Breakdown](#step-by-step-breakdown)

---

## 🎯 Overview

This is a **multimodal AI system** that processes video calls to create enriched transcripts with:
- **What was said** (transcription)
- **Who said it** (speaker diarization)
- **How it was said** (emotions, tone, reactions)

The system combines:
- **Signal processing** (audio/video analysis)
- **Machine learning** (emotion detection, clustering)
- **Large Language Models** (reasoning and refinement)

---

## 🔄 Complete Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Video File (.mp4)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: VIDEO INGESTION                                      │
│ ─────────────────────────────────────────────────────────── │
│ Tools: FFmpeg, OpenCV                                        │
│                                                               │
│ • Extract audio track → WAV file (16kHz, mono)              │
│ • Extract video frames → 2 frames per second                 │
│ • Get video metadata (duration, resolution)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: SPEECH-TO-TEXT TRANSCRIPTION                          │
│ ─────────────────────────────────────────────────────────── │
│ Tool: Sarvam AI API                                          │
│                                                               │
│ • Send audio to Sarvam API                                   │
│ • Receive transcript text                                    │
│ • Split transcript into sentence segments                    │
│ • Assign timestamps to each segment                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: SPEAKER DIARIZATION                                   │
│ ─────────────────────────────────────────────────────────── │
│ Tools: librosa, scikit-learn                                │
│                                                               │
│ • Extract audio features (MFCC, spectral, zero-crossing)    │
│ • Cluster audio frames by speaker characteristics           │
│ • Identify speaker segments with timestamps                  │
│ • Assign speaker labels to transcription segments            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: AUDIO EMOTION ANALYSIS                                │
│ ─────────────────────────────────────────────────────────── │
│ Tool: librosa                                                │
│                                                               │
│ • Analyze prosody features for each segment:                 │
│   - Pitch (fundamental frequency)                            │
│   - Tempo (speech rate)                                      │
│   - Energy (RMS amplitude)                                  │
│   - Spectral features                                        │
│ • Infer emotion from prosody patterns                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: VISUAL EMOTION DETECTION (Optional)                  │
│ ─────────────────────────────────────────────────────────── │
│ Tool: DeepFace                                               │
│                                                               │
│ • Detect faces in video frames                               │
│ • Analyze facial expressions                                │
│ • Classify emotions (Joy, Sadness, Anger, etc.)             │
│ • Extract confidence scores                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: MULTIMODAL FUSION                                    │
│ ─────────────────────────────────────────────────────────── │
│ Tool: Custom alignment logic                                │
│                                                               │
│ • Align all data by timestamps:                              │
│   - Transcription segments                                   │
│   - Speaker labels                                           │
│   - Audio emotions                                           │
│   - Visual emotions                                          │
│ • Merge audio + visual emotions                              │
│ • Create unified segments with all metadata                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: LLM REASONING & REFINEMENT                            │
│ ─────────────────────────────────────────────────────────── │
│ Tool: Groq API (LLaMA-3)                                     │
│                                                               │
│ • Send structured data to LLM:                               │
│   - Transcript text                                           │
│   - Raw emotion predictions                                  │
│   - Speaker information                                      │
│ • LLM performs:                                               │
│   - Emotion refinement (normalize labels)                   │
│   - Tone classification (Enthusiastic, Hesitant, etc.)       │
│   - Reaction classification (Question, Agreement, etc.)      │
│ • Generate final enriched metadata                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT: Enriched Transcript                │
│ ─────────────────────────────────────────────────────────── │
│ JSON format with:                                            │
│ • Timestamped segments                                       │
│ • Speaker labels                                             │
│ • Refined emotions                                           │
│ • Tone descriptions                                          │
│ • Reaction classifications                                   │
│ • Confidence scores                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tools & Libraries Used

### 1. **Video/Audio Processing**

#### **FFmpeg** (`ffmpeg-python`)
- **Purpose**: Extract audio from video files
- **Why**: Industry-standard, handles all video formats
- **Usage**: Converts video → WAV audio (16kHz, mono)
- **Location**: `app/ingestion/extract_audio.py`

#### **OpenCV** (`opencv-python`)
- **Purpose**: Extract video frames for visual analysis
- **Why**: Fast, reliable video processing
- **Usage**: Extracts frames at 2 FPS for emotion detection
- **Location**: `app/ingestion/extract_frames.py`

---

### 2. **Speech Recognition**

#### **Sarvam AI API** (`sarvamai`, `requests`)
- **Purpose**: Convert speech to text
- **Why**: High-quality ASR, supports Indian languages
- **Usage**: 
  - Sends audio file to API
  - Receives transcript text
  - Splits into sentence segments
- **Location**: `app/asr/sarvam_asr.py`
- **API Endpoint**: `https://api.sarvam.ai/speech-to-text`

---

### 3. **Speaker Diarization**

#### **librosa**
- **Purpose**: Extract audio features for speaker identification
- **Why**: Excellent audio analysis library
- **Usage**: 
  - Extracts MFCC (Mel-frequency cepstral coefficients)
  - Spectral features
  - Zero-crossing rate
- **Location**: `app/asr/diarization.py`

#### **scikit-learn**
- **Purpose**: Clustering algorithm for speaker separation
- **Why**: Simple, effective, no external dependencies
- **Usage**: 
  - KMeans or Agglomerative clustering
  - Groups audio frames by speaker characteristics
  - Auto-detects number of speakers
- **Location**: `app/asr/diarization.py`

**How it works:**
1. Extract audio features (MFCC, spectral, etc.)
2. Normalize features
3. Cluster frames into speaker groups
4. Convert clusters to time segments
5. Assign speaker labels

---

### 4. **Audio Emotion Analysis**

#### **librosa** (again)
- **Purpose**: Analyze prosody (how speech is delivered)
- **Why**: Comprehensive audio feature extraction
- **Usage**: 
  - **Pitch (F0)**: Fundamental frequency → excitement level
  - **Tempo**: Speech rate → urgency/calmness
  - **Energy (RMS)**: Volume → enthusiasm
  - **Spectral features**: Brightness, timbre
- **Location**: `app/audio_emotion/prosody.py`

**Emotion Inference:**
- High pitch + high energy + fast tempo → **Excitement/Joy**
- Low pitch + low energy → **Sadness**
- High pitch variation → **Surprise**
- Rule-based heuristics (refined by LLM later)

---

### 5. **Visual Emotion Detection**

#### **DeepFace**
- **Purpose**: Detect emotions from facial expressions
- **Why**: Pre-trained models, multiple emotion classes
- **Usage**: 
  - Detects faces in video frames
  - Analyzes facial expressions
  - Classifies: Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral
  - Returns confidence scores
- **Location**: `app/vision/facial_emotion.py`
- **Backend**: Uses TensorFlow/Keras models

**Note**: Currently optional/disabled in pipeline

---

### 6. **Multimodal Fusion**

#### **Custom Alignment Logic**
- **Purpose**: Combine all data sources by timestamps
- **Why**: Need to align different data streams
- **Usage**: 
  - Matches transcription segments with:
    - Speaker labels (by time overlap)
    - Audio emotions (by time range)
    - Visual emotions (by closest timestamp)
  - Merges audio + visual emotions
  - Creates unified segments
- **Location**: `app/fusion/align_and_merge.py`

**Fusion Strategy:**
- Visual emotion (if confidence > 0.7) → Primary
- Audio emotion (if confidence > 0.5) → Primary
- Otherwise → Combined or fallback

---

### 7. **LLM Reasoning**

#### **Groq API** (`groq`)
- **Purpose**: Refine emotions, classify tone/reactions
- **Why**: Fast inference, free tier available
- **Model**: LLaMA-3 70B (via Groq)
- **Usage**: 
  - Receives structured text + metadata
  - **Never processes raw audio/video** (LLM limitation)
  - Performs reasoning tasks:
    - Emotion normalization
    - Tone classification
    - Reaction detection
  - Returns refined metadata
- **Location**: `app/llm/groq_reasoning.py`

**LLM Input Format:**
```json
{
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "text": "I'm excited about this!",
      "audio_emotion": "Excitement",
      "visual_emotion": "Joy"
    }
  ]
}
```

**LLM Output:**
```json
{
  "emotion": "Joy",
  "tone": "Enthusiastic",
  "reaction": "Positive engagement"
}
```

---

### 8. **Web Framework**

#### **FastAPI** (`fastapi`)
- **Purpose**: REST API server
- **Why**: Modern, fast, auto-documentation
- **Usage**: 
  - `/transcribe` - Upload video, get JSON
  - `/transcribe/text` - Upload video, get text
  - `/health` - Health check
  - `/docs` - Interactive API docs
- **Location**: `app/main.py`

#### **Uvicorn**
- **Purpose**: ASGI server for FastAPI
- **Why**: High-performance async server

---

### 9. **Supporting Libraries**

#### **NumPy**
- Numerical operations, array handling

#### **PyTorch** (`torch`)
- Deep learning backend (for some models)

#### **TensorFlow**
- DeepFace backend

#### **python-dotenv**
- Load environment variables from `.env` file

#### **Pydantic**
- Data validation for API requests/responses

---

## 📊 Data Flow

### Input Data Structure
```
Video File (.mp4)
├── Audio Track (embedded)
└── Video Frames (embedded)
```

### Intermediate Data Structures

**After Step 1:**
```python
{
  "audio_path": "temp_audio.wav",
  "frames": [(timestamp, frame_array), ...],
  "duration": 23.69
}
```

**After Step 2:**
```python
{
  "text": "Good morning everyone...",
  "segments": [
    {
      "text": "Good morning everyone.",
      "start": 0.0,
      "end": 2.5,
      "words": []
    },
    ...
  ]
}
```

**After Step 3:**
```python
[
  {
    "text": "Good morning everyone.",
    "start": 0.0,
    "end": 2.5,
    "speaker": "SPEAKER_00"  # ← Added
  }
]
```

**After Step 4:**
```python
[
  {
    "text": "Good morning everyone.",
    "start": 0.0,
    "end": 2.5,
    "speaker": "SPEAKER_00",
    "audio_emotion": {
      "emotion": "Neutral",
      "confidence": 0.7,
      "features": {...}
    }
  }
]
```

**After Step 6 (Fusion):**
```python
[
  {
    "text": "Good morning everyone.",
    "speaker": "SPEAKER_00",
    "timestamp": "00:00:00",
    "audio_emotion": {...},
    "visual_emotion": {...},
    "merged_emotion": {
      "primary_emotion": "Neutral",
      "primary_source": "audio"
    }
  }
]
```

**After Step 7 (LLM Refinement):**
```python
[
  {
    "timestamp": "00:00:00",
    "speaker": "SPEAKER_00",
    "text": "Good morning everyone.",
    "emotion": "Neutral",        # ← Refined
    "tone": "Professional",      # ← Added
    "reaction": "Greeting",      # ← Added
    "audio_confidence": 0.7,
    "visual_confidence": 0.0
  }
]
```

---

## 🔍 Step-by-Step Breakdown

### Step 1: Video Ingestion
**File**: `app/ingestion/extract_audio.py`, `extract_frames.py`

**Process:**
1. FFmpeg extracts audio → WAV file (16kHz, mono)
2. OpenCV reads video, extracts frames at 2 FPS
3. Returns audio path, frame list, duration

**Why 2 FPS?**
- Balance between accuracy and processing speed
- Enough for emotion detection (faces don't change rapidly)

---

### Step 2: Speech-to-Text
**File**: `app/asr/sarvam_asr.py`

**Process:**
1. Read audio file
2. POST to Sarvam API with audio + model name
3. Receive transcript JSON
4. Parse transcript text
5. Split into sentences (regex-based)
6. Distribute sentences across audio duration
7. Create segments with timestamps

**Why sentence splitting?**
- Sarvam API returns full transcript, not segments
- Need segments for emotion analysis per utterance

---

### Step 3: Speaker Diarization
**File**: `app/asr/diarization.py`

**Process:**
1. Load audio with librosa
2. Extract features:
   - MFCC (13 coefficients) - captures vocal tract characteristics
   - Spectral centroid - brightness
   - Zero-crossing rate - speech activity
3. Normalize features (StandardScaler)
4. Auto-detect number of speakers (try 2-4, pick best)
5. Cluster frames (KMeans or Agglomerative)
6. Convert clusters to time segments
7. Merge short segments
8. Assign speakers to transcription segments by time overlap

**Why clustering works:**
- Different speakers have different vocal characteristics
- MFCC captures these differences
- Clustering groups similar voices together

---

### Step 4: Audio Emotion Analysis
**File**: `app/audio_emotion/prosody.py`

**Process:**
For each transcription segment:
1. Extract audio segment (start → end time)
2. Calculate prosody features:
   - **Pitch (F0)**: Using `librosa.pyin()`
   - **Tempo**: Beat tracking
   - **Energy**: RMS amplitude
   - **Spectral features**: Centroid, rolloff
3. Apply heuristics:
   - High pitch + high energy → Excitement
   - Low pitch + low energy → Sadness
   - High variation → Surprise
4. Return emotion label + confidence

**Why prosody?**
- How you say something matters as much as what you say
- Pitch, tempo, energy reveal emotional state
- Works even without understanding words

---

### Step 5: Visual Emotion Detection (Optional)
**File**: `app/vision/facial_emotion.py`

**Process:**
1. For each extracted frame:
2. DeepFace detects face
3. Analyzes facial expression
4. Classifies emotion (7 classes)
5. Returns emotion + confidence

**Note**: Currently commented out in pipeline

---

### Step 6: Multimodal Fusion
**File**: `app/fusion/align_and_merge.py`

**Process:**
1. For each transcription segment:
2. Find matching audio emotion (same time range)
3. Find closest visual emotion (by timestamp)
4. Merge emotions:
   - Prefer visual if high confidence (>0.7)
   - Otherwise prefer audio if confidence >0.5
   - Fallback to combined
5. Create aligned segments with all metadata

**Why alignment?**
- Different data sources have different timestamps
- Need to match them correctly
- Time tolerance: 0.5 seconds

---

### Step 7: LLM Reasoning
**File**: `app/llm/groq_reasoning.py`

**Process:**
1. Batch segments (10 at a time)
2. Create prompt with:
   - Transcript text
   - Raw emotion predictions
   - Speaker labels
3. Send to Groq API (LLaMA-3)
4. LLM performs:
   - **Emotion refinement**: Normalize labels, correct mistakes
   - **Tone classification**: Enthusiastic, Hesitant, Confident, etc.
   - **Reaction detection**: Question, Agreement, Disagreement, etc.
5. Parse JSON response
6. Merge with original segments

**Why LLM?**
- Rule-based emotion detection is limited
- LLM understands context and nuance
- Can infer tone and reactions from text
- Refines raw predictions

**Important**: LLM only sees text + metadata, never raw audio/video

---

## 🎯 Key Design Decisions

### 1. **Hybrid Architecture**
- Signal processing for low-level features
- LLM for high-level reasoning
- Best of both worlds

### 2. **Free/Open-Source Tools**
- No paid APIs (except Groq free tier)
- Works offline for most components
- No vendor lock-in

### 3. **Modular Design**
- Each step is independent
- Easy to replace components
- Easy to test

### 4. **Error Handling**
- Graceful degradation
- Continues even if one step fails
- Fallback values provided

### 5. **Timestamp Alignment**
- All data aligned by time
- Enables multimodal fusion
- Accurate metadata assignment

---

## 📈 Performance Characteristics

**Processing Time:**
- 1 min video: ~2-5 minutes
- 5 min video: ~10-25 minutes
- Real-time factor: 2-5x

**Bottlenecks:**
1. Sarvam API call (network latency)
2. Groq API calls (batch processing)
3. DeepFace (if enabled - slow)
4. Audio feature extraction (CPU-intensive)

**Optimization Opportunities:**
- Parallel processing
- Caching
- GPU acceleration
- Batch API calls

---

## 🔧 Configuration

**Environment Variables:**
- `SARVAM_API_KEY` - Speech-to-text API
- `GROQ_API_KEY` - LLM reasoning
- `SARVAM_MODEL` - Model version (default: saarika:v2.5)
- `PORT` - Server port (default: 8000)

**Pipeline Options:**
- `use_diarization` - Enable/disable speaker diarization
- `use_llm` - Enable/disable LLM refinement
- `sarvam_model` - Sarvam model version

---

## 🎓 Summary

This system combines:
- **Audio processing** (FFmpeg, librosa)
- **Video processing** (OpenCV, DeepFace)
- **Machine learning** (scikit-learn clustering)
- **Cloud APIs** (Sarvam, Groq)
- **Web framework** (FastAPI)

To create a **multimodal AI pipeline** that:
1. Transcribes speech
2. Identifies speakers
3. Detects emotions (audio + visual)
4. Infers tone and reactions
5. Produces enriched transcripts

All while being **free, open-source, and production-ready**!
