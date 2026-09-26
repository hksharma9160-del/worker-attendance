from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os, sqlite3, base64, uuid, io, csv, json
import numpy as np
import cv2

try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
WORKER_PHOTOS = DATA / "workers"
ATT_PHOTOS = DATA / "attendance"
MODELS = BASE / "models"
DB = DATA / "attendance.db"
YUNET = MODELS / "face_detection_yunet_2023mar.onnx"
SFACE = MODELS / "face_recognition_sface_2021dec.onnx"
IST = ZoneInfo("Asia/Kolkata")

for p in [DATA, WORKER_PHOTOS, ATT_PHOTOS, MODELS]:
    p.mkdir(parents=True, exist_ok=True)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and create_client)
supabase: Client | None = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) if USE_SUPABASE else None

app = FastAPI(title="Worker Photo Attendance")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


def now_ist():
    return datetime.now(IST)


def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init_local_db():
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

if not USE_SUPABASE:
    init_local_db()


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


def save_local_bytes(data: bytes, folder: Path, stem: str):
    path = folder / f"{stem}.jpg"
    path.write_bytes(data)
    return str(path.relative_to(BASE))


def ensure_buckets():
    if not USE_SUPABASE:
        return
    # Best-effort bucket creation. If already present, Supabase returns an error which we ignore.
    for bucket in ("worker-photos", "attendance-photos"):
        try:
            supabase.storage.create_bucket(bucket, options={"public": False})
        except Exception:
            pass

ensure_buckets()


def save_photo(data: bytes, kind: str, stem: str):
    if not USE_SUPABASE:
        folder = WORKER_PHOTOS if kind == "worker" else ATT_PHOTOS
        return save_local_bytes(data, folder, stem)
    bucket = "worker-photos" if kind == "worker" else "attendance-photos"
    object_path = f"{now_ist().strftime('%Y/%m')}/{stem}.jpg"
    try:
        supabase.storage.from_(bucket).upload(
            object_path,
            data,
            file_options={"content-type": "image/jpeg", "upsert": "false"},
        )
        return f"{bucket}/{object_path}"
    except Exception as e:
        raise HTTPException(500, f"फोटो cloud storage में save नहीं हो सकी: {str(e)[:180]}")


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
    face = max(faces, key=lambda f: f[2] * f[3])
    aligned = recognizer.alignCrop(img, face)
    feat = recognizer.feature(aligned).flatten().astype(float)
    return feat


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else -1.0


def sb_data(resp):
    return getattr(resp, "data", None) or []


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/status")
def status():
    today = now_ist().strftime("%Y-%m-%d")
    if USE_SUPABASE:
        try:
            workers = sb_data(supabase.table("workers").select("id").execute())
            present = sb_data(supabase.table("attendance").select("id").eq("attendance_date", today).execute())
            return {"workers": len(workers), "present_today": len(present), "face_recognition_ready": face_models_ready(), "storage": "supabase"}
        except Exception as e:
            raise HTTPException(500, f"Supabase database error: {str(e)[:180]}")
    c = conn()
    total = c.execute("SELECT COUNT(*) n FROM workers").fetchone()["n"]
    present = c.execute("SELECT COUNT(*) n FROM attendance WHERE attendance_date=?", (today,)).fetchone()["n"]
    c.close()
    return {"workers": total, "present_today": present, "face_recognition_ready": face_models_ready(), "storage": "local"}


@app.get("/api/workers")
def workers():
    if USE_SUPABASE:
        try:
            rows = sb_data(supabase.table("workers").select("id,worker_code,name,department,created_at").order("name").execute())
            return rows
        except Exception as e:
            raise HTTPException(500, f"Supabase database error: {str(e)[:180]}")
    c = conn(); rows = c.execute("SELECT id, worker_code, name, department, created_at FROM workers ORDER BY name").fetchall(); c.close()
    return [dict(r) for r in rows]


@app.post("/api/workers")
def add_worker(payload: WorkerIn):
    code = payload.worker_code.strip(); name = payload.name.strip()
    if not code or not name:
        raise HTTPException(400, "नाम और Worker ID जरूरी हैं।")
    data, img = decode_photo(payload.photo)
    embedding = get_embedding(img)
    photo_rel = save_photo(data, "worker", f"{code}_{uuid.uuid4().hex[:8]}")
    created = now_ist().isoformat(timespec='seconds')
    if USE_SUPABASE:
        try:
            row = {
                "worker_code": code, "name": name, "department": payload.department.strip(),
                "photo_path": photo_rel, "embedding": embedding.tolist() if embedding is not None else None,
                "created_at": created,
            }
            inserted = sb_data(supabase.table("workers").insert(row).execute())
            return {"ok": True, "id": inserted[0]["id"] if inserted else None, "face_saved": embedding is not None}
        except Exception as e:
            msg = str(e)
            if "duplicate" in msg.lower() or "23505" in msg:
                raise HTTPException(409, "यह Worker ID पहले से मौजूद है।")
            raise HTTPException(500, f"Worker save नहीं हुआ: {msg[:180]}")
    try:
        c = conn()
        c.execute("INSERT INTO workers(worker_code,name,department,photo_path,embedding,created_at) VALUES(?,?,?,?,?,?)",
                  (code, name, payload.department.strip(), photo_rel,
                   json.dumps(embedding.tolist()) if embedding is not None else None, created))
        c.commit(); wid = c.execute("SELECT last_insert_rowid()").fetchone()[0]; c.close()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "यह Worker ID पहले से मौजूद है।")
    return {"ok": True, "id": wid, "face_saved": embedding is not None}


def get_worker(worker_id: int):
    if USE_SUPABASE:
        rows = sb_data(supabase.table("workers").select("*").eq("id", worker_id).limit(1).execute())
        return rows[0] if rows else None
    c = conn(); row = c.execute("SELECT * FROM workers WHERE id=?", (worker_id,)).fetchone(); c.close()
    return dict(row) if row else None


def mark_attendance(worker_id: int, photo: str, method: str, confidence=None):
    data, _ = decode_photo(photo)
    w = get_worker(worker_id)
    if not w:
        raise HTTPException(404, "Worker नहीं मिला।")
    now = now_ist(); d = now.strftime("%Y-%m-%d"); t = now.strftime("%H:%M:%S")
    rel = save_photo(data, "attendance", f"{d}_{worker_id}_{uuid.uuid4().hex[:8]}")
    if USE_SUPABASE:
        try:
            row = {
                "worker_id": worker_id, "attendance_date": d, "attendance_time": t,
                "photo_path": rel, "method": method, "confidence": confidence,
                "created_at": now.isoformat(timespec='seconds'),
            }
            supabase.table("attendance").insert(row).execute()
        except Exception as e:
            msg = str(e)
            if "duplicate" in msg.lower() or "23505" in msg:
                raise HTTPException(409, f"{w['name']} की आज की attendance पहले ही लग चुकी है।")
            raise HTTPException(500, f"Attendance save नहीं हुई: {msg[:180]}")
        return {"ok": True, "worker": w["name"], "date": d, "time": t, "method": method, "confidence": confidence}
    c = conn()
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
    if USE_SUPABASE:
        rows = sb_data(supabase.table("workers").select("*").not_.is_("embedding", "null").execute())
    else:
        c = conn(); rows = [dict(r) for r in c.execute("SELECT * FROM workers WHERE embedding IS NOT NULL").fetchall()]; c.close()
    best = None
    for r in rows:
        try:
            stored = r["embedding"] if isinstance(r["embedding"], list) else json.loads(r["embedding"])
            score = cosine(emb, stored)
        except Exception:
            continue
        if best is None or score > best[0]:
            best = (score, r)
    threshold = 0.45
    if best is None or best[0] < threshold:
        raise HTTPException(404, "चेहरा किसी registered worker से match नहीं हुआ।")
    return mark_attendance(best[1]["id"], payload.photo, "face_recognition", round(best[0], 4))


@app.get("/api/attendance")
def attendance_list(date: str | None = None):
    if USE_SUPABASE:
        try:
            q = supabase.table("attendance").select("id,worker_id,attendance_date,attendance_time,method,confidence,workers(worker_code,name,department)")
            if date:
                q = q.eq("attendance_date", date)
            rows = sb_data(q.order("attendance_date", desc=True).order("attendance_time", desc=True).limit(500).execute())
            out = []
            for r in rows:
                w = r.get("workers") or {}
                out.append({
                    "id": r.get("id"), "worker_code": w.get("worker_code"), "name": w.get("name"),
                    "department": w.get("department"), "attendance_date": r.get("attendance_date"),
                    "attendance_time": r.get("attendance_time"), "method": r.get("method"), "confidence": r.get("confidence")
                })
            return out
        except Exception as e:
            raise HTTPException(500, f"Attendance list नहीं मिली: {str(e)[:180]}")
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
    rows = attendance_list(None)
    sio = io.StringIO(); writer = csv.writer(sio)
    writer.writerow(["Worker ID","Name","Department","Date","Time","Method","Confidence"])
    for r in rows:
        writer.writerow([r.get("worker_code"), r.get("name"), r.get("department"), r.get("attendance_date"), r.get("attendance_time"), r.get("method"), r.get("confidence")])
    data = sio.getvalue().encode('utf-8-sig')
    return StreamingResponse(io.BytesIO(data), media_type="text/csv", headers={"Content-Disposition":"attachment; filename=attendance.csv"})
