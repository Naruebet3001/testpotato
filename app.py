# app.py
from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

# โหลดโมเดล YOLO
model = YOLO("https://drive.google.com/file/d/110kAb82an0NQ_OYLea8n1kYwOQ9BoqK5/view?usp=sharing")  # ตรวจสอบว่าโมเดลอยู่ในโฟลเดอร์เดียวกับ app.py

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

    # บันทึกไฟล์ชั่วคราว
    tmp_path = os.path.join("/tmp", file.filename)
    file.save(tmp_path)

    try:
        results = model(tmp_path)[0]

        if results.boxes and len(results.boxes) > 0:
            best = max(results.boxes, key=lambda b: float(b.conf[0]))
            cls = int(best.cls[0])
            conf = float(best.conf[0])
            info = diseases.get(cls, {"id": 0, "name": "ไม่พบข้อมูล", "treatment": "ไม่พบข้อมูล"})
        else:
            info = diseases[9]  # Healthy

        os.remove(tmp_path)
        return jsonify({
            "disease_id": info["id"],
            "disease_name": info["name"],
            "confidence": f"{conf:.2%}" if results.boxes else "100%",
            "treatment": info["treatment"]
        })

    except Exception as e:
        os.remove(tmp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

