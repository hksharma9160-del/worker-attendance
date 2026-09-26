# Worker Photo Attendance — Mobile Friendly Version

यह mobile-first web app worker registration, live camera photo, daily attendance, duplicate रोकना, report और CSV export देता है। OpenCV YuNet + SFace models जोड़ने पर face recognition से worker auto-identify भी किया जा सकता है।

## Mobile version में क्या नया है
- Phone-first responsive UI और bottom navigation
- बड़े touch buttons और full-screen style camera
- Front / back camera switch
- Photo retake preview
- Mobile attendance cards/report
- PWA manifest: HTTPS पर browser से Home Screen पर install किया जा सकता है
- Desktop पर भी वही app चलता है

## 1) Install
Python 3.10+ install करें। Project folder में:

```bash
pip install -r requirements.txt
```

## 2) Face Recognition models (optional)

```bash
python download_models.py
```

App restart करें। Models के बिना manual-photo attendance काम करती रहेगी।

## 3) Run
Windows:

```bat
run.bat
```

Mac/Linux:

```bash
./run.sh
```

PC पर खोलें: `http://localhost:8000`

## Mobile पर कैसे चलाएँ
App server phone से reachable होना चाहिए। Production/mobile camera के लिए **HTTPS strongly recommended है** क्योंकि modern mobile browsers non-secure LAN URLs (`http://192.168.x.x:8000`) पर camera permission रोक सकते हैं।

Best setup:
1. App को HTTPS वाले cloud/VPS/domain पर deploy करें.
2. Phone में उस HTTPS link को Chrome/Safari में खोलें.
3. Camera permission Allow करें.
4. Supported browser में **Add to Home Screen / Install App** करें.

Development testing के लिए USB/HTTPS reverse proxy या local trusted HTTPS setup इस्तेमाल किया जा सकता है।

## Features
- Worker ID, नाम, department/site registration
- Registration live photo
- Attendance live photo evidence
- एक worker की एक दिन में duplicate attendance blocked
- Optional automatic face matching
- आज की present count
- Date-wise attendance report
- CSV export
- SQLite database
- Installable mobile web app shell (PWA)

## Production में जरूरी अगले upgrades
- Admin login और supervisor roles
- HTTPS + encrypted backup
- GPS/site geofence (worker consent के साथ)
- Liveness / anti-spoofing
- IN/OUT, shift, late, half-day, overtime
- Multi-site support
- Payroll integration
- Worker privacy/consent notice और photo retention policy

## Important
Face recognition को real workforce पर लगाने से पहले camera, lighting और different phones पर threshold test करें। केवल face match पर payroll/disciplinary decision automate न करें; manual review रखें।
