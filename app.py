from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import gdown
import tempfile
import uuid
import cv2
import numpy as np
import base64

app = Flask(__name__)

# โหลดโมเดล YOLO จาก Google Drive ถ้ายังไม่มีใน /tmp
MODEL_PATH = "/tmp/yolov11.pt"
DRIVE_URL = "https://drive.google.com/uc?id=1BoLD1112mW0h0g3SXHjpGvKuskG_0STp"

if not os.path.exists(MODEL_PATH):
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# โหลดโมเดล YOLO
model = YOLO(MODEL_PATH)

# Mapping class → disease
diseases = {
    0: {"id": 1, "name": "Early Blight", "treatment": "ถอนต้นที่เป็นโรค..."},
    1: {"id": 2, "name": "Late Blight", "treatment": "ลดการให้น้ำแบบพ่นฝอย..."},
    2: {"id": 3, "name": "Septoria Leaf Spot", "treatment": "กำจัดเศษพืชรอบแปลง..."},
    9: {"id": 10, "name": "Healthy", "treatment": "ไม่ต้องทำการรักษา"}
}

@app.route("/predict", methods=["POST"])
def predict():
    # ✅ รับข้อมูลเป็น JSON
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "ไม่พบข้อมูลรูปภาพใน JSON payload"}), 400

    try:
        # ✅ แปลง base64 string กลับมาเป็นไฟล์ภาพ
        base64_image = data["image"]
        image_bytes = base64.b64decode(base64_image)
        
        # ✅ ใช้ numpy และ cv2 ในการอ่านข้อมูล binary
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "ไม่สามารถอ่านไฟล์ภาพจาก base64 ได้"}), 400
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ทำการทำนาย (ส่วนนี้เหมือนเดิม)
        results = model(img)[0]

        if results.boxes and len(results.boxes) > 0:
            best = max(results.boxes, key=lambda b: float(b.conf[0]))
            cls = int(best.cls[0])
            conf = float(best.conf[0])
            info = diseases.get(cls, {"id": 0, "name": "ไม่พบข้อมูล", "treatment": "ไม่พบข้อมูล"})
            confidence_str = f"{conf:.2%}"
        else:
            info = diseases[9]
            confidence_str = "100%"

        return jsonify({
            "disease_id": info["id"],
            "disease_name": info["name"],
            "confidence": confidence_str,
            "treatment": info["treatment"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # ลบไฟล์ temp ทิ้ง
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

