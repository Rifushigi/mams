# MAMS: A Multimodal Attendance Management System

A multimodal attendance monitoring system for higher-education institutions.

---

## Table of Contents

1. [Description](#1-description)
2. [Dataset Information](#2-dataset-information)
3. [Code Information](#3-code-information)
4. [Usage Instructions](#4-usage-instructions)
5. [Requirements](#5-requirements)
6. [Methodology](#6-methodology)

---

## 1. Description

MAMS replaces manual roll-call with automated attendance
capture. A student's presence is established through one or more independent
verification channels, each of which produces an auditable attendance record
carrying the method used and a confidence score.

The system is organised as three independently deployable components:

| Component | Role |
| --- | --- |
| `backend/` | REST API, authentication, session and attendance lifecycle, reporting, notifications |
| `ml-service/` | Face registration and verification, anomaly detection |
| `mobile/trace/` | Cross-platform client for students, lecturers, and administrators |

The system provides:

- Role-based access control across three roles (`admin`, `lecturer`, `student`).
- Email and push notification pipelines for session start and end, check-in
  confirmation, anomaly alerts, and low-attendance warnings.
- Per-class and per-student attendance reporting with CSV export.
- Face quality gating at registration time to reject low-quality enrolments.

---

## 2. Dataset Information

### 2.1 Availability statement

**No third-party, public, or human-subject dataset is distributed with this
repository.** All data consumed by MAMS is produced at runtime by its own
users and devices. Consequently there is no dataset to download, and no data
files are tracked in version control.

### 2.2 Data schema (MongoDB)

| Collection | Key fields |
| --- | --- |
| `User` | `email` (unique), `firstName`, `lastName`, `password` (bcrypt hash), `role`, `isVerified`, `avatar`, `fcmToken`, `googleId`, `createdAt`, `updatedAt`, `deletedAt` |
| `Students` | `userId`, `matricNo` (unique), `program`, `level` (100–700), `faceModelId`, `nfcUid`, `bleToken` |
| `Lecturers` | `userId`, `staffId` (unique), `college` |
| `Class` | `title`, `className`, `courseCode`, `lecturerId`, `semester`, `year`, `beaconIds[]` |
| `Enrollments` | `studentId`, `classId`, `enrolledAt` |
| `AttendanceSession` | `classId`, `startTime`, `endTime`, `status` (`ongoing` \| `completed`) |
| `AttendanceLog` | `sessionId`, `studentId`, `checkedInAt`, `method` (`face` \| `nfc` \| `ble` \| `geofence`), `confidenceScore` (0–1), `isAnomaly` |
| `Session` | `userId`, `deviceId` (unique), `accessToken`, `refreshToken`, `isActive`, `lastActivity` |
| `Verification` | `userId`, `email`, `otp` (`code`, `createdAt`, `expiresAt`, `usedAt`), `verificationToken`, `isVerified`, `verifiedAt` |
| `NotificationPreferences` | `userId`, `email.*` and `push.*` toggles for `sessionStart`, `sessionEnd`, `checkIn`, `anomaly`, `lowAttendance` |

`AttendanceLog` is the primary unit of analysis for any downstream study: it is
the append-only record linking a student to a session together with the
verification method and its confidence.

### 2.3 Machine learning artifacts

The ML service persists its state as local files under `ml-service/data/`. This directory is created on first write.

| Path | Contents | Format |
| --- | --- | --- |
| `data/face_encodings.json` | Mapping of user identifier to a 128-dimensional face embedding | AES-256-GCM envelope, `{"version", "algorithm", "nonce", "ciphertext"}`; the plaintext within is `{"encodings": {user_id: float[128]}}` |
| `data/anomaly_models.json` | Per-user Isolation Forest state and retained feature history | JSON |

---

## 3. Code Information

### 3.1 Technology stack

| Layer | Technologies |
| --- | --- |
| Mobile client | React Native 0.79, React 19, Expo SDK 53, Expo Router 5, MobX 6, TypeScript 5.8 |
| Device access | `react-native-ble-plx`, `expo-camera`, `expo-location`, `expo-local-authentication`, `expo-secure-store` |
| API | Node.js 20, Express 4, TypeScript 5.8 (strict), Mongoose 8, `express-validator`, JWT, bcrypt, Socket.IO 4 |
| ML service | Python 3.11, FastAPI, Uvicorn, `face_recognition` (dlib), OpenCV, scikit-learn, NumPy, pandas |
| Persistence | MongoDB; JSON and SavedModel files for ML artifacts |
| External services | Cloudinary, SMTP, Firebase Cloud Messaging |
| Orchestration | Docker, Docker Compose |

A Postman collection covering these endpoints is provided at
[docs/Trace.postman_collection.json](docs/Trace.postman_collection.json).

---

## 4. Usage Instructions

### 4.1 Clone the repo

```bash
git clone https://github.com/Rifushigi/mams.git
cd mams
```

### 4.2 Configure the backend

```bash
cd backend
cp .env.example .env
```

The following variables are validated at startup by
`isEnvDefined()` and the process will terminate if any are absent:

| Variable |
| --- |
| `BASE_URL`, `PORT`, `ENV` |
| `LOCAL_DATABASE_URL`, `PRD_DATABASE_URL` |
| `ACCESS_TOKEN_SECRET`, `REFRESH_TOKEN_SECRET` |
| `ACCESS_TOKEN_DURATION`, `REFRESH_TOKEN_DURATION` |
| `EMAIL_FROM`, `EMAIL_EXP`, `OTP_EXP` |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `SMTP_SECURE` |
| `CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET`, `CLOUDINARY_DIRECTORY` |
| `FIREBASE_PROJECT_ID`, `FIREBASE_CLIENT_EMAIL`, `FIREBASE_PRIVATE_KEY` |
| `ML_SERVICE_URL` |

### 4.3 Run the backend

```bash
cd backend
pnpm install          # or npm install
pnpm dev              
```

Production build and serve:

```bash
pnpm build            
pnpm serve            
```

On first successful database connection the service creates a default
administrator account through `createDefaultAdmin()` before binding to
`0.0.0.0:${PORT}` (default `3000`).

### 4.4 Run the ML service

```bash
cd ml-service
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install httpx                  
cp .env.example .env
python -m utils.crypto             # prints a fresh FACE_ENCODING_KEY
python app.py                      # Uvicorn on 0.0.0.0:5000
```

Configure the following before starting:

| Variable | Purpose |
| --- | --- |
| `FACE_ENCODING_KEY` | Base64-encoded 256-bit key encrypting the biometric template store. Generate with `python -m utils.crypto` |
| `ACCESS_TOKEN_SECRET` | Shared with the backend, used to verify access tokens presented by end users |
| `ML_SERVICE_TOKEN` | Service credential for backend-to-ML calls carrying no end-user context |
| `BACKEND_URL` | Base URL for posting automatic check-ins to `${BACKEND_URL}/api/v1/attendance/auto-checkin` |

**Encryption key.** `FACE_ENCODING_KEY` is required to read or write enrolments.

Note:

- **The key is not recoverable.** If it is lost or changed, every enrolled
  template is unreadable and all students must re-enrol. 

### 4.5 Run the mobile client

```bash
cd mobile/trace
npm install
npm start             # Expo development server
```

Platform-specific entry points:

```bash
npm run android
npm run ios
npm run web
```

### 4.6 Load the face recognition data

Faces are enrolled through the ML service. Registration is rejected when the
computed quality score falls below 67.

```bash
curl -X POST http://localhost:5000/api/v1/face/register \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "BU21CSC1048", "face_data": "<base64-encoded-jpeg>"}'
```

Verification returns a match decision, a confidence value, the matched
identifier, and the quality metrics computed for the submitted frame:

```bash
curl -X POST http://localhost:5000/api/v1/face/verify \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "BU21CSC1048", "face_data": "<base64-encoded-jpeg>"}'
```

Encodings accumulate in `ml-service/data/face_encodings.json`, encrypted under
`FACE_ENCODING_KEY`, and are decrypted and reloaded automatically on service
start.


### 4.7 Run the BLE geofencing simulation

The standalone simulation requires no database or running services and
demonstrates the classroom-resolution algorithm in isolation:

```bash
node simulate_ble_geofence.js
```

The same algorithm is exposed over HTTP against live class records:

```bash
curl -X POST http://localhost:3000/api/v1/classes/detect-by-beacons \
  -H "Content-Type: application/json" \
  -d '{"beaconIds": ["beacon-uuid-1", "beacon-uuid-2", "beacon-uuid-3"]}'
```

---

## 5. Requirements

### 5.1 Platform prerequisites

| Requirement | Version | Notes |
| --- | --- | --- |
| Node.js | 20 LTS or later | Backend targets ES2024 with `NodeNext` module resolution |
| pnpm or npm | Current | A `pnpm-workspace.yaml` is present in `backend/` |
| Python | 3.11 | Matches the ML service container image |
| MongoDB | 6 or later | `mongo:latest` under Docker Compose |
| CMake and a C++ toolchain | Current | Required to build `dlib`, a transitive dependency of `face-recognition` |
| Expo CLI | Bundled with `expo` | Installed with the mobile dependencies |
| Android Studio or Xcode | Current | Only for native device builds; BLE and NFC are unavailable in Expo Go |

### 5.2 Backend dependencies


`express`, `mongoose`, `socket.io`, `jsonwebtoken`, `bcrypt`,
`express-validator`, `cors`, `cookie-parser`, `body-parser`, `morgan`, `multer`,
`cloudinary`, `nodemailer`, `ejs`, `firebase-admin`, `csv-writer`, `date-fns`,
`axios`, `dotenv`.

Development: `typescript`, `tsx`, `nodemon`, and the corresponding `@types`
packages.

### 5.3 ML service dependencies

```text
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
numpy==1.24.3
opencv-python==4.8.1.78
face-recognition==1.3.0
scikit-learn==1.3.2
tensorflow==2.14.0
pandas==2.1.3
python-jose==3.3.0
cryptography==42.0.5
pydantic==2.5.2
python-dotenv==1.0.0
requests==2.31.0
websockets
dotenv
```

### 5.4 Mobile dependencies

 Primary
packages: `expo` (SDK 53), `expo-router`, `react-native` 0.79, `react` 19,
`mobx` and `mobx-react-lite`, `@react-navigation/*`, `react-native-ble-plx`,
`react-native-permissions`, `expo-camera`, `expo-location`,
`expo-local-authentication`, `expo-secure-store`, `axios`,
`react-native-chart-kit`, `react-native-reanimated`.

### 5.5 External service accounts

A MongoDB deployment, an SMTP account, a Cloudinary account, and a Firebase
project with Cloud Messaging enabled are required for full functionality.

---

## 6. Methodology

### 6.1 Face quality assessment

| Metric | Definition | Acceptance criterion |
| --- | --- | --- |
| Sharpness | Variance of the Laplacian of the grayscale crop | `> 80` |
| Brightness | Mean grayscale pixel intensity of the crop | `80 < b < 200` |
| Face size | Face bounding-box area as a percentage of frame area | `> 5` |

The overall quality score is the proportion of satisfied criteria expressed as a
percentage, yielding one of four discrete values: 0, 33, 67, or 100.
Registration is refused below 67, so at least two of the three criteria must
hold. The thresholds were selected empirically.

### 6.2 Face encoding and enrolment

A detected face is encoded as a 128-dimensional vector using the
`face_recognition` library, which wraps dlib's ResNet-based face embedding
model. Encodings are keyed by user identifier and persisted to
`data/face_encodings.json`.

Duplicate enrolment is handled by comparing a candidate encoding against every
stored encoding. When the minimum Euclidean distance is below 0.6, the candidate
is treated as the same individual as the matched record; the stored encoding is
overwritten only if the candidate's quality score exceeds that of the incumbent,
and is otherwise rejected as a duplicate.

### 6.3 Face verification

Verification computes the Euclidean distance between the probe encoding and all
enrolled encodings, then takes the nearest neighbour. Confidence is reported as
`1 - d`, where `d` is the best-match distance. A verification succeeds only when
both conditions hold:

1. The best-match distance is below the acceptance threshold.
2. The identifier of the nearest neighbour equals the claimed `user_id`.

The second condition makes this a one-to-one verification rather than a
one-to-many identification, which limits false accepts as the
population of enrolled student grows.

### 6.4 BLE proximity geofencing

Each `Class` record carries a set of BLE beacon identifiers installed in its
room. A client submits the set of identifiers observed in a scan, and the server
resolves the classroom by set intersection.

1. For every class, compute the cardinality of the intersection between the
   scanned identifiers and the class's registered beacons.
2. Discard any class whose intersection contains fewer than three beacons.
3. Return the surviving class with the largest intersection; if none survives,
   report no match.

The three-beacon quorum is a fixed constant. It provides tolerance to individual
beacon dropout while making a spoof that presents a single captured identifier
insufficient to establish presence of a student.

### 6.5 Anomaly detection

1. Facial landmarks are extracted for the frame.
2. Features are formed from all pairwise Euclidean distances between landmark
   points within each landmark group.
3. The feature vector is appended to a rolling window retaining the ten most
   recent observations for that user.
4. Once five observations are available, the forest is refitted on the window
   and the current vector is scored.
5. A prediction of `-1` marks an anomaly; the reported confidence is the
   absolute value of `score_samples`.

### 6.6 Attendance recording and anomaly flagging

For automatic check-ins originating from the ML service:

1. Reject the request if `studentId`, `sessionId`, or `location` is absent, or
   if the confidence score falls outside `[0, 1]`.
2. Enforce a minimum interval of 60 seconds between consecutive check-ins for
   the same student, tracked in memory.
3. Require an attendance session with status `ongoing`.
4. Reject a second check-in by the same student within the same session,
   enforcing at most one attendance record per student per session.
5. Confirm the student record exists.
6. Flag the log as anomalous when the confidence score is below 0.8.
7. Persist the `AttendanceLog` and dispatch notifications.

Manual check-ins bypass the confidence pathway and are recorded with a
confidence of 1.0 and `isAnomaly` set to false.

### 6.7 Reporting and aggregation

- Total sessions and total students.
- Average attendance, computed as total logs divided by total sessions.
- A breakdown of check-in counts by verification method.
- A count of logs flagged as anomalous.
- CSV export via `csv-writer`, with dates formatted through `date-fns`.

A low-attendance notification is sent to the lecturer when the ratio of
checked-in students to total students falls below 0.7.

---
