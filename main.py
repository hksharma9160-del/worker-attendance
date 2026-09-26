from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import sqlite3, base64, uuid, io, csv, json
import numpy as np
import cv2

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
WORKER_PHOTOS = DATA / "workers"
ATT_PHOTOS = DATA / "attendance"
MODELS = BASE / "models"
DB = DATA / "attendance.db"
YUNET = MODELS / "face_detection_yunet_2023mar.onnx"
SFACE = MODELS / "face_recognition_sface_2021dec.onnx"

for p in [DATA, WORKER_PHOTOS, ATT_PHOTOS, MODELS]:
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Worker Photo Attendance")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    c = conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS workers (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      worker_code TEXT UNIQUE NOT NULL,
      name TEXT NOT NULL,
      department TEXT DEFAULT '',
      photo_path TEXT NOT NULL,
      embedding TEXT,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS attendance (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      worker_id INTEGER NOT NULL,
      attendance_date TEXT NOT NULL,
      attendance_time TEXT NOT NULL,
      photo_path TEXT NOT NULL,
      method TEXT NOT NULL,
      confidence REAL,
      created_at TEXT NOT NULL,
      UNIQUE(worker_id, attendance_date),
      FOREIGN KEY(worker_id) REFERENCES workers(id)
    );
    """)
    c.commit(); c.close()

init_db()

class WorkerIn(BaseModel):
    worker_code: str
    name: str
    department: str = ""
    photo: str

class AttendanceIn(BaseModel):
    worker_id: int
    photo: str

class RecognizeIn(BaseModel):
    photo: str


def decode_photo(data_url: str):
    try:
        raw = data_url.split(',', 1)[1] if ',' in data_url else data_url
        data = base64.b64decode(raw)
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('invalid image')
        return data, img
    except Exception:
        raise HTTPException(400, "फोटो पढ़ी नहीं जा सकी।")


def save_bytes(data: bytes, folder: Path, stem: str):
    path = folder / f"{stem}.jpg"
    path.write_bytes(data)
    return str(path.relative_to(BASE))


def face_models_ready():
    return YUNET.exists() and SFACE.exists()


def get_embedding(img):
    if not face_models_ready():
        return None
    h, w = img.shape[:2]
    detector = cv2.FaceDetectorYN_create(str(YUNET), "", (w, h), 0.8, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF_create(str(SFACE), "")
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
    # use largest detected face
    face = max(faces, key=lambda f: f[2] * f[3])
    aligned = recognizer.alignCrop(img, face)
    feat = recognizer.feature(aligned).flatten().astype(float)
    return feat


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else -1.0

@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")

@app.get("/api/status")
def status():
    c = conn()
    total = c.execute("SELECT COUNT(*) n FROM workers").fetchone()["n"]
    today = datetime.now().strftime("%Y-%m-%d")
    present = c.execute("SELECT COUNT(*) n FROM attendance WHERE attendance_date=?", (today,)).fetchone()["n"]
    c.close()
    return {"workers": total, "present_today": present, "face_recognition_ready": face_models_ready()}

@app.get("/api/workers")
def workers():
    c = conn()
    rows = c.execute("SELECT id, worker_code, name, department, created_at FROM workers ORDER BY name").fetchall()
    c.close()
    return [dict(r) for r in rows]

@app.post("/api/workers")
def add_worker(payload: WorkerIn):
    code = payload.worker_code.strip(); name = payload.name.strip()
    if not code or not name:
        raise HTTPException(400, "नाम और Worker ID जरूरी हैं।")
    data, img = decode_photo(payload.photo)
    embedding = get_embedding(img)
    photo_rel = save_bytes(data, WORKER_PHOTOS, f"{code}_{uuid.uuid4().hex[:8]}")
    try:
        c = conn()
        c.execute("INSERT INTO workers(worker_code,name,department,photo_path,embedding,created_at) VALUES(?,?,?,?,?,?)",
                  (code, name, payload.department.strip(), photo_rel,
                   json.dumps(embedding.tolist()) if embedding is not None else None,
                   datetime.now().isoformat(timespec='seconds')))
        c.commit(); wid = c.execute("SELECT last_insert_rowid()").fetchone()[0]; c.close()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "यह Worker ID पहले से मौजूद है।")
    return {"ok": True, "id": wid, "face_saved": embedding is not None}


def mark_attendance(worker_id: int, photo: str, method: str, confidence=None):
    data, _ = decode_photo(photo)
    c = conn()
    w = c.execute("SELECT * FROM workers WHERE id=?", (worker_id,)).fetchone()
    if not w:
        c.close(); raise HTTPException(404, "Worker नहीं मिला।")
    now = datetime.now(); d = now.strftime("%Y-%m-%d"); t = now.strftime("%H:%M:%S")
    rel = save_bytes(data, ATT_PHOTOS, f"{d}_{worker_id}_{uuid.uuid4().hex[:8]}")
    try:
        c.execute("INSERT INTO attendance(worker_id,attendance_date,attendance_time,photo_path,method,confidence,created_at) VALUES(?,?,?,?,?,?,?)",
                  (worker_id, d, t, rel, method, confidence, now.isoformat(timespec='seconds')))
        c.commit()
    except sqlite3.IntegrityError:
        c.close(); raise HTTPException(409, f"{w['name']} की आज की attendance पहले ही लग चुकी है।")
    c.close()
    return {"ok": True, "worker": w["name"], "date": d, "time": t, "method": method, "confidence": confidence}

@app.post("/api/attendance")
def attendance(payload: AttendanceIn):
    return mark_attendance(payload.worker_id, payload.photo, "manual_photo")

@app.post("/api/recognize")
def recognize(payload: RecognizeIn):
    if not face_models_ready():
        raise HTTPException(503, "Face recognition models install नहीं हैं। README में download_models.py चलाएँ।")
    _, img = decode_photo(payload.photo)
    emb = get_embedding(img)
    if emb is None:
        raise HTTPException(400, "फोटो में साफ चेहरा नहीं मिला।")
    c = conn(); rows = c.execute("SELECT * FROM workers WHERE embedding IS NOT NULL").fetchall(); c.close()
    best = None
    for r in rows:
        try:
            score = cosine(emb, json.loads(r["embedding"]))
        except Exception:
            continue
        if best is None or score > best[0]:
            best = (score, r)
    # Conservative default for SFace cosine similarity; tune with your own workforce images.
    threshold = 0.45
    if best is None or best[0] < threshold:
        raise HTTPException(404, "चेहरा किसी registered worker से match नहीं हुआ।")
    return mark_attendance(best[1]["id"], payload.photo, "face_recognition", round(best[0], 4))

@app.get("/api/attendance")
def attendance_list(date: str | None = None):
    c = conn()
    if date:
        rows = c.execute("""SELECT a.id,w.worker_code,w.name,w.department,a.attendance_date,a.attendance_time,a.method,a.confidence
                          FROM attendance a JOIN workers w ON w.id=a.worker_id
                          WHERE a.attendance_date=? ORDER BY a.attendance_time DESC""", (date,)).fetchall()
    else:
        rows = c.execute("""SELECT a.id,w.worker_code,w.name,w.department,a.attendance_date,a.attendance_time,a.method,a.confidence
                          FROM attendance a JOIN workers w ON w.id=a.worker_id
                          ORDER BY a.attendance_date DESC,a.attendance_time DESC LIMIT 500""").fetchall()
    c.close(); return [dict(r) for r in rows]

@app.get("/api/export.csv")
def export_csv():
    c = conn(); rows = c.execute("""SELECT w.worker_code,w.name,w.department,a.attendance_date,a.attendance_time,a.method,a.confidence
                                  FROM attendance a JOIN workers w ON w.id=a.worker_id
                                  ORDER BY a.attendance_date DESC,a.attendance_time DESC""").fetchall(); c.close()
    sio = io.StringIO(); writer = csv.writer(sio)
    writer.writerow(["Worker ID","Name","Department","Date","Time","Method","Confidence"])
    for r in rows: writer.writerow(list(r))
    data = sio.getvalue().encode('utf-8-sig')
    return StreamingResponse(io.BytesIO(data), media_type="text/csv", headers={"Content-Disposition":"attachment; filename=attendance.csv"})
