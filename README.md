# Worker Attendance — Permanent Supabase Version (V2)

यह clean version मोबाइल/लैपटॉप दोनों पर चलता है और Render restart/redeploy के बाद भी worker/attendance data Supabase में सुरक्षित रखता है।

## Features
- Mobile camera + photo attendance
- Worker registration
- Same worker की daily duplicate attendance block
- Daily report + CSV export
- Supabase Postgres में permanent worker/attendance data
- Supabase Storage में worker/attendance photos
- `SUPABASE_SECRET_KEY` support (`SUPABASE_SERVICE_ROLE_KEY` fallback भी है)
- `/api/status` से साफ पता चलता है कि storage `supabase` है या `local`
- Optional face recognition models

## 1) Supabase SQL
Supabase → SQL Editor में `supabase_schema.sql` पूरा paste करके Run करें।

## 2) Render Environment
Render service → Environment में ये variables रखें:

- `SUPABASE_URL` = Supabase Project URL
- `SUPABASE_SECRET_KEY` = Supabase Secret key (`sb_secret_...`)

Secret key browser/frontend में कभी न डालें।

## 3) GitHub Upload
Repository root में ये files/folders होने चाहिए:

- `main.py`
- `requirements.txt`
- `static/`
- `supabase_schema.sql`
- `render.yaml` (optional)

`static` के अंदर `index.html`, `icon.svg`, `manifest.webmanifest`, `sw.js` सीधे होने चाहिए — `static/static` नहीं।

## 4) Render Commands
Build command:

```text
pip install -r requirements.txt
```

Start command:

```text
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## 5) Permanent-storage check
Deploy के बाद खोलें:

```text
https://YOUR-APP.onrender.com/api/status
```

सही setup पर response में यह दिखना चाहिए:

```json
"storage": "supabase",
"permanent": true,
"key_source": "SUPABASE_SECRET_KEY"
```

अगर `storage: local` दिखे तो Render Environment variables सही नहीं हैं।

## 6) Test
1. App में नया test worker register करें।
2. उसकी attendance लगाएँ।
3. Render में एक बार redeploy करें।
4. App फिर खोलें। Worker और attendance record मौजूद रहने चाहिए।

## Local laptop use
Local fallback के लिए:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Supabase env variables न हों तो local SQLite fallback चलता है; Render production में permanent data के लिए Supabase variables जरूरी हैं।
