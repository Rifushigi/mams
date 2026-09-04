from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from dotenv import load_dotenv
import asyncio
import cv2
import numpy as np
import httpx
from datetime import datetime

from services.face_recognition import FaceRecognitionService
from services.anomaly_detection import AnomalyDetectionService
from services.engagement_prediction import EngagementPredictionService
from utils.auth import verify_token
from utils.errors import (
    ValidationError,
    NotFoundError,
    ConflictError,
    DatabaseError,
    MLServiceError,
)

load_dotenv()

app = FastAPI(title="MAMS ML Service")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
face_recognition = FaceRecognitionService()
anomaly_detection = AnomalyDetectionService()
engagement_prediction = EngagementPredictionService()


class FaceVerificationRequest(BaseModel):
    face_data: str  # Base64 encoded image
    user_id: str


class EngagementPredictionRequest(BaseModel):
    face_data: str  # Base64 encoded image
    user_id: str
    session_id: str


class AnomalyDetectionRequest(BaseModel):
    face_data: str  # Base64 encoded image
    user_id: str
    session_id: str


class VideoStreamRequest(BaseModel):
    stream_id: str
    location: str
    session_id: str


@app.post("/api/v1/face/verify")
async def verify_face(
    request: FaceVerificationRequest, token: str = Depends(verify_token)
):
    try:
        data = request.dict()
        face_data = data.get("face_data")
        location = data.get("user_id")

        if not face_data or not location:
            raise ValidationError("Missing required data")

        # Decode image and compute quality metrics
        image = face_recognition.decode_base64_image(face_data)
        metrics = face_recognition.compute_face_quality(image)

        result = await face_recognition.verify_face(face_data, location)
        return {
            "match": result["match"],
            "confidence": result["confidence"],
            "face_id": result["face_id"],
            "metrics": metrics,
        }
    except Exception as e:
        if isinstance(
            e, (ValidationError, NotFoundError, ConflictError, DatabaseError)
        ):
            raise HTTPException(status_code=e.status_code, detail=str(e))
        raise HTTPException(status_code=500, detail="Face verification failed")


@app.post("/api/v1/face/register")
async def register_face(
    request: FaceVerificationRequest, token: str = Depends(verify_token)
):
    try:
        data = request.dict()
        face_data = data.get("face_data")
        user_id = data.get("user_id")

        if not face_data or not user_id:
            raise ValidationError("Missing required data")

        # Decode image and compute quality metrics
        image = face_recognition.decode_base64_image(face_data)
        metrics = face_recognition.compute_face_quality(image)
        if metrics["score"] < 67:
            return {
                "success": False,
                "message": "Face quality too low for registration.",
                "metrics": metrics,
            }

        reg_result = await face_recognition.register_face(face_data, user_id)
        return {
            "success": reg_result["status"] in ["new", "updated"],
            "status": reg_result["status"],
            "user_id": reg_result["user_id"],
            "message": reg_result["message"],
            "metrics": metrics,
        }
    except Exception as e:
        if isinstance(
            e, (ValidationError, NotFoundError, ConflictError, DatabaseError)
        ):
            raise HTTPException(status_code=e.status_code, detail=str(e))
        raise HTTPException(status_code=500, detail="Face registration failed")


@app.post("/api/v1/engagement/predict")
async def predict_engagement(
    request: EngagementPredictionRequest, token: str = Depends(verify_token)
):
    try:
        prediction = await engagement_prediction.predict(
            request.face_data, request.user_id, request.session_id
        )
        return {
            "engagement": prediction["engagement"],
            "attention": prediction["attention"],
            "timestamp": prediction["timestamp"],
        }
    except Exception as e:
        if isinstance(
            e, (ValidationError, NotFoundError, ConflictError, DatabaseError)
        ):
            raise HTTPException(status_code=e.status_code, detail=str(e))
        raise HTTPException(status_code=500, detail="Engagement prediction failed")


@app.post("/api/v1/anomaly/detect")
async def detect_anomaly(
    request: AnomalyDetectionRequest, token: str = Depends(verify_token)
):
    try:
        result = await anomaly_detection.detect(
            request.face_data, request.user_id, request.session_id
        )
        return {
            "is_anomaly": result["is_anomaly"],
            "confidence": result["confidence"],
            "reason": result["reason"],
        }
    except Exception as e:
        if isinstance(
            e, (ValidationError, NotFoundError, ConflictError, DatabaseError)
        ):
            raise HTTPException(status_code=e.status_code, detail=str(e))
        raise HTTPException(status_code=500, detail="Anomaly detection failed")


@app.websocket("/ws/video-stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    stream_data = None

    try:
        # Receive stream metadata first
        metadata = await websocket.receive_json()
        stream_data = {
            "stream_id": metadata.get("stream_id"),
            "location": metadata.get("location"),
            "session_id": metadata.get("session_id"),
        }

        if not all(stream_data.values()):
            raise ValidationError("Missing stream metadata")

        while True:
            # Receive video frame
            frame_data = await websocket.receive_bytes()

            # Convert bytes to numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Process frame
            faces = face_recognition.detect_faces(frame)

            for face in faces:
                # Verify face
                result = face_recognition.verify_face(face)

                if result["match"]:
                    # Send check-in to backend
                    try:
                        async with httpx.AsyncClient() as client:
                            await client.post(
                                f"{os.getenv('BACKEND_URL')}/api/v1/attendance/auto-checkin",
                                json={
                                    "studentId": result["user_id"],
                                    "sessionId": stream_data["session_id"],
                                    "location": stream_data["location"],
                                    "confidence": result["confidence"],
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )
                    except Exception as e:
                        raise MLServiceError(
                            f"Failed to send check-in to backend: {str(e)}"
                        )

                # Send result back to client
                await websocket.send_json(
                    {
                        "type": "face_detected",
                        "data": {
                            "match": result["match"],
                            "confidence": result["confidence"],
                            "location": stream_data["location"],
                        },
                    }
                )

    except Exception as e:
        if isinstance(
            e,
            (
                ValidationError,
                NotFoundError,
                ConflictError,
                DatabaseError,
                MLServiceError,
            ),
        ):
            await websocket.send_json(
                {
                    "type": "error",
                    "data": {
                        "message": str(e),
                        "code": e.status_code if hasattr(e, "status_code") else 500,
                    },
                }
            )
        else:
            await websocket.send_json(
                {
                    "type": "error",
                    "data": {"message": "Internal server error", "code": 500},
                }
            )
    finally:
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
