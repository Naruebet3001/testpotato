from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import gdown
import tempfile
import uuid

app = Flask(__name__)

# ✅ โหลดโมเดล YOLO จาก Google Drive ถ้ายังไม่มีใน /tmp
MODEL_PATH = "/tmp/yolov11.pt"
DRIVE_URL = "https://drive.google.com/file/d/1eoE4rg1uKktf1nbTSOddAGGD5TfCtTPW"

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
    if "file" not in request.files:
        return jsonify({"error": "ไม่พบไฟล์ภาพ"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "ชื่อไฟล์ไม่ถูกต้อง"}), 400

    # ✅ ใช้ tempfile + uuid เพื่อป้องกัน path แปลก ๆ จาก client
    tmp_dir = tempfile.gettempdir()
    tmp_filename = f"{uuid.uuid4().hex}.jpg"
    tmp_path = os.path.join(tmp_dir, tmp_filename)
    file.save(tmp_path)

    try:
        results = model(tmp_path)[0]

        if results.boxes and len(results.boxes) > 0:
            best = max(results.boxes, key=lambda b: float(b.conf[0]))
            cls = int(best.cls[0])
            conf = float(best.conf[0])
            info = diseases.get(cls, {"id": 0, "name": "ไม่พบข้อมูล", "treatment": "ไม่พบข้อมูล"})
            confidence_str = f"{conf:.2%}"
        else:
            info = diseases[9]  # Healthy
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
        # ✅ ลบไฟล์ temp ทิ้ง
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

