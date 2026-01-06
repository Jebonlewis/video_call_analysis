"""
FastAPI main application.
Provides REST API for video transcription service.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn

from pipeline import VideoTranscriptionPipeline


# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Video Transcription API",
    description="Transcribe video calls with emotion and metadata enrichment",
    version="1.0.0"
)

# Global pipeline instance (can be reused)
pipeline: Optional[VideoTranscriptionPipeline] = None


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    transcript: list
    summary: dict
    metadata: dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Video transcription service is running"
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_video(
    video: UploadFile = File(..., description="Video file to transcribe")
):
    """
    Transcribe video file and return enriched transcript.
    
    Args:
        video: Uploaded video file
    
    Returns:
        Enriched transcript with emotions and metadata
    """
    # Validate file type
    if not video.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    file_ext = Path(video.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    temp_video_path = None
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_video_path = temp_file.name
            
            # Write uploaded content
            content = await video.read()
            temp_file.write(content)
        
        # Initialize pipeline if needed
        global pipeline
        if pipeline is None:
            pipeline = VideoTranscriptionPipeline(
                sarvam_model=os.getenv("SARVAM_MODEL", "saarika:v2.5"),
                use_diarization=True,
                use_llm=True
            )
        
        # Process video
        result = pipeline.process(temp_video_path)
        
        return JSONResponse(content=result)
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File error: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup temp file
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass


@app.post("/transcribe/text")
async def transcribe_video_text(
    video: UploadFile = File(..., description="Video file to transcribe")
):
    """
    Transcribe video file and return formatted text transcript.
    
    Args:
        video: Uploaded video file
    
    Returns:
        Formatted text transcript
    """
    # Validate file type
    if not video.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    file_ext = Path(video.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    temp_video_path = None
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_video_path = temp_file.name
            
            # Write uploaded content
            content = await video.read()
            temp_file.write(content)
        
        # Initialize pipeline if needed
        global pipeline
        if pipeline is None:
            pipeline = VideoTranscriptionPipeline(
                sarvam_model=os.getenv("SARVAM_MODEL", "saarika:v2.5"),
                use_diarization=True,
                use_llm=True
            )
        
        # Process video
        result = pipeline.process(temp_video_path)
        
        # Format as text
        text_output = pipeline.format_transcript_text(result)
        
        return PlainTextResponse(content=text_output, media_type="text/plain")
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File error: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup temp file
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Multimodal Video Transcription API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "transcribe_json": "/transcribe",
            "transcribe_text": "/transcribe/text",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run server
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
