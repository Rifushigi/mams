import face_recognition
import numpy as np
import base64
import cv2
import os
from typing import Dict, Any
import json
from datetime import datetime

from utils import crypto

ENCODINGS_PATH = "data/face_encodings.json"


class FaceRecognitionService:
    def __init__(self):
        self.face_encodings = {}
        self.face_locations = {}
        self.known_face_encodings = []
        self.known_face_ids = []
        self._needs_migration = False
        self.load_known_faces()

    def load_known_faces(self):
        """
        Load known face encodings from the encrypted store.

        """
        if not os.path.exists(ENCODINGS_PATH):
            return

        with open(ENCODINGS_PATH, "r") as f:
            payload = json.load(f)

        if crypto.is_encrypted(payload):
            document = crypto.decrypt(payload)
        else:
            document = payload
            self._needs_migration = True
            print(
                f"{ENCODINGS_PATH} is stored in plaintext. It will be encrypted on "
                f"the next write."
            )

        self.face_encodings = document.get("encodings", {})
        self.known_face_encodings = [
            np.array(encoding) for encoding in self.face_encodings.values()
        ]
        self.known_face_ids = list(self.face_encodings.keys())

    def save_known_faces(self):
        """
        Persist face encodings, encrypted with AES-256-GCM.

        """
        os.makedirs(os.path.dirname(ENCODINGS_PATH), exist_ok=True)
        envelope = crypto.encrypt({"encodings": self.face_encodings})

        temporary_path = f"{ENCODINGS_PATH}.tmp"
        with open(temporary_path, "w") as f:
            json.dump(envelope, f)
        os.replace(temporary_path, ENCODINGS_PATH)

        if getattr(self, "_needs_migration", False):
            print(f"{ENCODINGS_PATH} has been migrated to encrypted storage.")
            self._needs_migration = False

    def decode_base64_image(self, base64_string: str) -> np.ndarray:
        """Decode base64 image to numpy array"""
        try:
            # Remove data URL prefix if present
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]

            # Decode base64 string
            image_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")

    async def register_face(self, face_data: str, user_id: str) -> dict:
        """Register a new face or update if better quality. Returns dict with status and message."""
        try:
            # Decode image
            image = self.decode_base64_image(face_data)
            # Detect face
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                raise ValueError("No face detected in image")
            # Get face encoding
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            # Compute quality
            quality = self.compute_face_quality(image)
            # Check for duplicate face (by encoding)
            if self.known_face_encodings:
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                best_match_index = int(np.argmin(face_distances))
                best_match_distance = face_distances[best_match_index]
                if best_match_distance < 0.6:
                    matched_user_id = self.known_face_ids[best_match_index]
                    prev_quality = 0
                    if matched_user_id in self.face_encodings:
                        prev_image = None
                        # Try to reconstruct previous image quality if possible (not always available)
                        # For now, just use 100 if exists
                        prev_quality = 100
                    if quality["score"] > prev_quality:
                        # Overwrite with better quality
                        self.face_encodings[matched_user_id] = face_encoding.tolist()
                        self.save_known_faces()
                        return {
                            "status": "updated",
                            "user_id": matched_user_id,
                            "message": f"Face updated for user {matched_user_id} (better quality).",
                        }
                    else:
                        return {
                            "status": "duplicate",
                            "user_id": matched_user_id,
                            "message": f"Face already registered for user {matched_user_id} with equal or better quality.",
                        }
            # Register new face
            self.face_encodings[user_id] = face_encoding.tolist()
            self.known_face_encodings.append(face_encoding)
            self.known_face_ids.append(user_id)
            self.save_known_faces()
            return {
                "status": "new",
                "user_id": user_id,
                "message": f"Face registered for user {user_id}.",
            }
        except Exception as e:
            raise Exception(f"Face registration failed: {e}")

    def get_face_encoding(self, image: np.ndarray, face_location: tuple) -> np.ndarray:
        """Get face encoding for a specific face location in an image"""
        try:
            # Use face_recognition to get face encodings
            face_encodings = face_recognition.face_encodings(image, [face_location])
            if not face_encodings:
                raise ValueError("No face encoding found for the given face location")
            return face_encodings[0]
        except Exception as e:
            raise Exception(f"Failed to get face encoding: {e}")

    def detect_faces(self, image: np.ndarray) -> list:
        """Detect face locations in an image"""
        try:
            # Use face_recognition to detect face locations
            face_locations = face_recognition.face_locations(image)
            return face_locations
        except Exception as e:
            raise Exception(f"Failed to detect faces: {e}")

    def compute_face_quality(self, image: np.ndarray) -> dict:
        """Compute face quality metrics for the first detected face in the image."""
        face_locations = self.detect_faces(image)
        if not face_locations:
            return {
                "sharpness": 0,
                "brightness": 0,
                "face_size_pct": 0,
                "score": 0,
                "sharpness_good": False,
                "brightness_good": False,
                "face_size_good": False,
                "all_good": False,
            }
        top, right, bottom, left = face_locations[0]
        face_img = image[top:bottom, left:right]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        brightness = gray_face.mean()
        face_area = (right - left) * (bottom - top)
        frame_area = image.shape[0] * image.shape[1]
        face_size_pct = 100 * face_area / frame_area
        sharpness_good = sharpness > 80
        brightness_good = 80 < brightness < 200
        face_size_good = face_size_pct > 5
        all_good = sharpness_good and brightness_good and face_size_good
        score = (
            (int(sharpness_good) + int(brightness_good) + int(face_size_good)) / 3 * 100
        )
        return {
            "sharpness": sharpness,
            "brightness": brightness,
            "face_size_pct": face_size_pct,
            "score": score,
            "sharpness_good": sharpness_good,
            "brightness_good": brightness_good,
            "face_size_good": face_size_good,
            "all_good": all_good,
        }

    async def verify_face(self, face_data: str, user_id: str) -> Dict[str, Any]:
        """Verify a face against registered faces"""
        try:
            # Decode image
            image = self.decode_base64_image(face_data)

            # Detect face
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                raise ValueError("No face detected in image")

            # Get face encoding
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]

            # Compare with known faces
            if not self.known_face_encodings:
                return {"match": False, "confidence": 0.0, "face_id": None}

            # Calculate face distances
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            best_match_distance = face_distances[best_match_index]

            # Convert distance to confidence (0-1)
            confidence = 1 - best_match_distance

            # Check if it's a match
            is_match = (
                best_match_distance < 0.6
                and self.known_face_ids[best_match_index] == user_id
            )

            return {
                "match": is_match,
                "confidence": float(confidence),
                "face_id": self.known_face_ids[best_match_index] if is_match else None,
            }
        except Exception as e:
            raise Exception(f"Face verification failed: {e}")

    def get_face_landmarks(self, image: np.ndarray) -> Dict[str, Any]:
        """Get facial landmarks"""
        try:
            face_landmarks_list = face_recognition.face_landmarks(image)
            if not face_landmarks_list:
                return {}
            return face_landmarks_list[0]
        except Exception as e:
            raise Exception(f"Failed to get face landmarks: {e}")
