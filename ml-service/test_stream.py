import cv2
import numpy as np
import asyncio
import websockets
import json
import base64
from datetime import datetime
import os
from dotenv import load_dotenv
from services.face_recognition import FaceRecognitionService
import requests
import time
import random
import string
import face_recognition

load_dotenv()

FACE_ENCODINGS_PATH = "data/face_encodings.json"

face_service = FaceRecognitionService()
print("Loaded users:", face_service.known_face_ids)


def save_face_encodings(encodings):
    """Persist the harness's working set through the service's encrypted store."""
    face_service.face_encodings = encodings
    face_service.save_known_faces()

# Test configuration
STREAM_ID = "test_stream"
LOCATION = "test_location"
SESSION_ID = "test_session"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000")


async def process_frame(frame):
    import io

    # Convert frame to RGB for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_service.detect_faces(rgb_frame)

    # Process each face
    for face_location in face_locations:
        # Draw rectangle around face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Crop the face region
        face_image = rgb_frame[top:bottom, left:right]
        # Encode the face region as JPEG
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        face_base64 = base64.b64encode(buffer).decode("utf-8")

        # Try to match against all registered users
        try:
            # If no known faces, skip
            if not face_service.known_face_encodings:
                text = "No faces registered"
                color = (0, 0, 255)
            else:
                # Use a dummy user_id (will be ignored in is_match logic)
                result = await face_service.verify_face(face_base64, "dummy")
                if result["match"]:
                    text = f"Match: {result['face_id']} ({result['confidence']:.2f})"
                    color = (0, 255, 0)
                else:
                    text = "No Match"
                    color = (0, 0, 255)
        except Exception as e:
            print(f"Verification error: {e}")
            text = "Error"
            color = (0, 0, 255)

        cv2.putText(
            frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

        # If face is verified, send check-in to backend
        if "result" in locals() and result.get("match"):
            try:
                async with websockets.connect(f"{BACKEND_URL}/ws") as websocket:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "check_in",
                                "data": {
                                    "studentId": result["face_id"],
                                    "sessionId": SESSION_ID,
                                    "location": LOCATION,
                                    "confidence": result["confidence"],
                                    "timestamp": datetime.now().isoformat(),
                                },
                            }
                        )
                    )
            except Exception as e:
                print(f"Failed to send check-in: {e}")

    return frame


async def register_face(frame, user_id):
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_service.detect_faces(rgb_frame)

    if not face_locations:
        print("No face detected in frame")
        return False

    # Get the first face
    face_location = face_locations[0]
    top, right, bottom, left = face_location
    face_image = rgb_frame[top:bottom, left:right]
    # Encode the face region as JPEG
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
    face_base64 = base64.b64encode(buffer).decode("utf-8")

    # Register face
    try:
        face_id = await face_service.register_face(face_base64, user_id)
        print(f"Face registered successfully with ID: {face_id}")
        return True
    except Exception as e:
        print(f"Failed to register face: {e}")
        return False


def generate_user_id():
    year = str(random.randint(20, 24))  # e.g., 21 for 2021-2024
    dept = random.choice(["CSC", "EEE", "MTH", "PHY", "BIO"])
    digits = "".join(random.choices(string.digits, k=4))
    return f"BU{year}{dept}{digits}"


async def registration_mode():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    print("[Registration Mode] Press 'r' to register a new face, 'q' to quit.")
    confirm_message = None
    confirm_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_service.detect_faces(rgb_frame)
        score = 0
        user_id = None
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_img = rgb_frame[top:bottom, left:right]
            # Metrics
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            brightness = gray_face.mean()
            face_area = (right - left) * (bottom - top)
            frame_area = frame.shape[0] * frame.shape[1]
            face_size_pct = 100 * face_area / frame_area
            sharpness_good = sharpness > 80
            brightness_good = 80 < brightness < 200
            face_size_good = face_size_pct > 5
            all_good = sharpness_good and brightness_good and face_size_good
            score = (
                (int(sharpness_good) + int(brightness_good) + int(face_size_good))
                / 3
                * 100
            )
            if score == 100:
                score_color = (0, 255, 0)  # Green
            elif score >= 67:
                score_color = (0, 165, 255)  # Orange (BGR)
            else:
                score_color = (0, 0, 255)  # Red
            metrics = [
                f"Sharpness: {sharpness:.0f} ({'OK' if sharpness_good else 'Low'})",
                f"Brightness: {brightness:.0f} ({'OK' if brightness_good else 'Bad'})",
                f"Face Size: {face_size_pct:.1f}% ({'OK' if face_size_good else 'Small'})",
                f"Face Quality: {score:.0f}%",
            ]
            box_color = (0, 255, 0) if all_good else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            for i, line in enumerate(metrics):
                y_offset = top - 30 - 15 * i
                if y_offset < 10:
                    y_offset = top + 20 + 15 * i
                color = score_color if "Face Quality" in line else box_color
                cv2.putText(
                    frame,
                    line,
                    (left, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
        # Show confirmation or error overlays
        if confirm_message and (time.time() - confirm_time < 2):
            cv2.putText(
                frame,
                confirm_message,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
        cv2.imshow("Face Registration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            if not face_locations:
                confirm_message = "No face detected for registration."
                confirm_time = time.time()
                continue
            if score < 67:
                confirm_message = "Face quality too low for registration."
                confirm_time = time.time()
                continue
            user_id = generate_user_id()
            # Register the first detected face
            top, right, bottom, left = face_locations[0]
            face_image = rgb_frame[top:bottom, left:right]
            _, buffer = cv2.imencode(
                ".jpg", cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            )
            face_base64 = base64.b64encode(buffer).decode("utf-8")
            try:
                # Simulate backend registration logic locally
                # Check for duplicate face (by encoding)
                face_encoding = face_service.get_face_encoding(
                    rgb_frame, face_locations[0]
                )
                quality = face_service.compute_face_quality(rgb_frame)
                is_duplicate = False
                matched_user_id = None
                if face_service.known_face_encodings:
                    import numpy as np

                    face_distances = face_recognition.face_distance(
                        face_service.known_face_encodings, face_encoding
                    )
                    best_match_index = int(np.argmin(face_distances))
                    best_match_distance = face_distances[best_match_index]
                    if best_match_distance < 0.3:
                        is_duplicate = True
                        matched_user_id = face_service.known_face_ids[best_match_index]
                        prev_quality = 100  # Simulate previous quality as 100
                        if quality["score"] > prev_quality:
                            # Overwrite with better quality
                            face_service.face_encodings[matched_user_id] = (
                                face_encoding.tolist()
                            )
                            save_face_encodings(face_service.face_encodings)
                            confirm_message = f"Face updated for user {matched_user_id} (better quality)."
                        else:
                            confirm_message = f"Face already registered for user {matched_user_id} with equal or better quality."
                        confirm_time = time.time()
                        print(confirm_message)
                if not is_duplicate:
                    # Generate a unique matric number
                    while True:
                        user_id = generate_user_id()
                        if user_id not in face_service.face_encodings:
                            break
                    face_service.face_encodings[user_id] = face_encoding.tolist()
                    face_service.known_face_encodings.append(face_encoding)
                    face_service.known_face_ids.append(user_id)
                    save_face_encodings(face_service.face_encodings)
                    confirm_message = f"Face registered for user: {user_id}"
                    confirm_time = time.time()
                    print(confirm_message)
            except Exception as e:
                confirm_message = f"Failed to register face: {e}"
                confirm_time = time.time()
                print(f"Failed to register face: {e}")
    cap.release()
    cv2.destroyAllWindows()


async def recognition_mode():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    print(
        "[Recognition Mode] Recognizing faces. Press 'q' to quit. Press 'i' to toggle payload info overlay."
    )
    show_payload_info = False
    recognized_counter = {}
    already_checked_in = set()
    frame_count = 0
    total_time = 0
    start_time = time.time()
    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_service.detect_faces(rgb_frame)
        for face_location in face_locations:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            face_image = rgb_frame[top:bottom, left:right]
            _, buffer = cv2.imencode(
                ".jpg", cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            )
            face_base64 = base64.b64encode(buffer).decode("utf-8")
            try:
                if not face_service.known_face_encodings:
                    text = "No faces registered"
                    color = (0, 0, 255)
                else:
                    # Recognition logic: match against any user
                    face_encoding = face_service.get_face_encoding(
                        rgb_frame, face_location
                    )
                    import numpy as np

                    face_distances = face_recognition.face_distance(
                        face_service.known_face_encodings, face_encoding
                    )
                    best_match_index = int(np.argmin(face_distances))
                    best_match_distance = face_distances[best_match_index]
                    if best_match_distance < 0.3:
                        matched_user_id = face_service.known_face_ids[best_match_index]
                        text = f"Match: {matched_user_id} (distance: {best_match_distance:.2f})"
                        color = (0, 255, 0)
                        # Info overlay for payload
                        if show_payload_info:
                            payload = {
                                "studentId": matched_user_id,
                                "sessionId": SESSION_ID,
                                "location": LOCATION,
                                "confidence": 1 - best_match_distance,
                                "timestamp": datetime.now().isoformat(),
                            }
                            payload_lines = [f"{k}: {v}" for k, v in payload.items()]
                            for i, line in enumerate(payload_lines):
                                y_offset = top - 30 - 15 * i
                                if y_offset < 10:
                                    y_offset = top + 20 + 15 * i
                                cv2.putText(
                                    frame,
                                    line,
                                    (left, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 0),
                                    1,
                                )
                        # Count consecutive recognitions
                        recognized_counter.setdefault(matched_user_id, 0)
                        recognized_counter[matched_user_id] += 1
                        # Simulate backend call only once per user per session
                        if (
                            recognized_counter[matched_user_id] == 10
                            and matched_user_id not in already_checked_in
                        ):
                            payload = {
                                "studentId": matched_user_id,
                                "sessionId": SESSION_ID,
                                "location": LOCATION,
                                "confidence": 1 - best_match_distance,
                                "timestamp": datetime.now().isoformat(),
                            }
                            print(
                                f"[Simulated Request] POST {BACKEND_URL}/api/v1/attendance/auto-checkin"
                            )
                            print(f"Payload: {payload}")
                            # Simulate backend response
                            backend_response = {
                                "success": True,
                                "message": f"Check-in successful for {matched_user_id}",
                                "attendance": {
                                    "studentId": matched_user_id,
                                    "sessionId": SESSION_ID,
                                    "status": "checked-in",
                                    "timestamp": payload["timestamp"],
                                },
                            }
                            print(f"[Simulated Response] {backend_response}")
                            already_checked_in.add(matched_user_id)
                    else:
                        text = "No Match"
                        color = (0, 0, 255)
                        # Reset counters for all users
                        recognized_counter = {}
            except Exception as e:
                print(f"Verification error: {e}")
                text = "Error"
                color = (0, 0, 255)
            cv2.putText(
                frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        # After all face processing and before cv2.imshow:
        frame_end = time.time()
        frame_time = frame_end - frame_start
        total_time += frame_time
        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Frame time: {frame_time*1000:.1f} ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("i"):
            show_payload_info = not show_payload_info
            print(
                f"[Info Overlay] Payload info {'enabled' if show_payload_info else 'disabled'}."
            )
    cap.release()
    cv2.destroyAllWindows()


def clear_face_encodings():
    import os

    if os.path.exists(FACE_ENCODINGS_PATH):
        os.remove(FACE_ENCODINGS_PATH)
        print(f"{FACE_ENCODINGS_PATH} has been deleted. You can now re-register faces.")
    else:
        print(f"{FACE_ENCODINGS_PATH} does not exist.")


async def main():
    print("Select mode:")
    print("1. Registration mode")
    print("2. Recognition mode")
    mode = input("Enter 1 or 2: ").strip()
    if mode == "1":
        await registration_mode()
    elif mode == "2":
        await recognition_mode()
    else:
        print("Invalid selection. Exiting.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "clear_db":
        clear_face_encodings()
    else:
        asyncio.run(main())
