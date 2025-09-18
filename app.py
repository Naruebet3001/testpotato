from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import gdown
import tempfile
import uuid
import cv2
import numpy as np

app = Flask(__name__)

# โหลดโมเดล YOLO จาก Google Drive ถ้ายังไม่มีใน /tmp
MODEL_PATH = "/tmp/yolov11.pt"
DRIVE_URL = "https://drive.google.com/uc?id=1BoLD1112mW0h0g3SXHjpGvKuskG_0STp"

if not os.path.exists(MODEL_PATH):
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# โหลดโมเดล YOLO
model = YOLO(MODEL_PATH)

# Mapping class → disease
# ...
diseases = {
    0: {"id": 1, "name": "Bacterial Spot", "treatment": "ใช้สารชีวภัณฑ์..."},
    1: {"id": 2, "name": "Early Blight", "treatment": "ถอนต้นที่เป็นโรค..."},
    2: {"id": 3, "name": "Late Blight", "treatment": "ลดการให้น้ำแบบพ่นฝอย..."},
    3: {"id": 4, "name": "Leaf Mold", "treatment": "ปรับปรุงการถ่ายเทอากาศ..."},
    4: {"id": 5, "name": "Septoria Leaf Spot", "treatment": "กำจัดเศษพืชรอบแปลง..."},
    5: {"id": 6, "name": "Spider Mites", "treatment": "พ่นน้ำไล่..."},
    6: {"id": 7, "name": "Target Spot", "treatment": "กำจัดวัชพืช..."},
    7: {"id": 8, "name": "Tomato Yellow Leaf Curl Virus", "treatment": "กำจัดแมลงหวี่ขาว..."},
    8: {"id": 9, "name": "Tomato Mosaic Virus", "treatment": "ทำลายต้นที่เป็นโรค..."},
    9: {"id": 10, "name": "Healthy", "treatment": "ไม่ต้องทำการรักษา"}
}
# ...

@app.route("/predict", methods=["POST"])
def predict():
    # Initialize tmp_path to None outside the try block
    tmp_path = None

    if "file" not in request.files:
        return jsonify({"error": "ไม่พบไฟล์ภาพ"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "ชื่อไฟล์ไม่ถูกต้อง"}), 400

    try:
        tmp_dir = tempfile.gettempdir()
        tmp_filename = f"{uuid.uuid4().hex}.jpg"
        tmp_path = os.path.join(tmp_dir, tmp_filename)
        file.save(tmp_path)

        # ✅ Read the image with OpenCV and convert it to a matrix
        img = cv2.imread(tmp_path)
        if img is None:
            return jsonify({"error": "ไม่สามารถอ่านไฟล์ภาพได้"}), 400
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform prediction
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
        # Check if tmp_path exists before trying to remove it
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))



