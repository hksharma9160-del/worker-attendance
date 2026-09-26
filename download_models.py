from pathlib import Path
from urllib.request import urlretrieve
BASE = Path(__file__).resolve().parent
M = BASE / "models"; M.mkdir(exist_ok=True)
files = {
  "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
  "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
for name,url in files.items():
    out=M/name
    if out.exists(): print(name, "already exists"); continue
    print("Downloading", name)
    urlretrieve(url,out)
print("Done. Restart the app.")
