import cv2
from flask import Flask, jsonify, Response, stream_with_context
from ultralytics import YOLO
import time

# ======= Global Variable untuk Delay =======
last_detected_time = 0
cooldown_seconds = 3
last_people_count = 0
allow_display = False

# ======= Tanya Index Kamera dari User =======
def minta_input_index_kamera():
    try:
        index = int(input("📷 Masukkan index kamera yang ingin digunakan (device rio 0/1, anyboard 4): "))
        return index
    except ValueError:
        print("❌ Input tidak valid. Gunakan angka saja.")
        exit(1)

CAMERA_INDEX = minta_input_index_kamera()

# ======= Inisialisasi Kamera =======
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not camera.isOpened():
    raise RuntimeError(f"❌ Kamera index {CAMERA_INDEX} tidak dapat dibuka.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ======= Load Model =======
model = YOLO("yolov8n.pt")
print(model.names)

# ======= Setup Flask =======
app = Flask(__name__)

def gen_frames():
    global last_detected_time, last_people_count, allow_display

    while True:
        success, frame = camera.read()
        if not success:
            print("⚠️ Gagal membaca frame dari kamera.")
            break

        results = model(frame)
        count_now = 0
        annotated_frame = frame.copy()

        # Hitung jumlah orang
        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            cls = int(cls)
            if model.names[cls] == "person" and conf >= 0.7:
                count_now += 1

        current_time = time.time()

        if count_now != last_people_count:
            last_detected_time = current_time
            allow_display = False
            last_people_count = count_now

        if not allow_display and (current_time - last_detected_time >= cooldown_seconds):
            allow_display = True

        # Gambar jika delay selesai
        if allow_display:
            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                cls = int(cls)
                if model.names[cls] == "person" and conf >= 0.7:
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"person {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.putText(annotated_frame, f"Total people: {count_now}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Kirim frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(stream_with_context(gen_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect')
def detect_people():
    global last_detected_time, last_people_count

    success, frame = camera.read()
    if not success:
        return jsonify({"error": "kamera tidak tersedia"}), 500

    results = model(frame)
    count_now = sum(1 for box in results[0].boxes.data if model.names[int(box[5])] == "person" and box[4] >= 0.6)

    current_time = time.time()

    if last_people_count == 0 and count_now > 0:
        if current_time - last_detected_time < cooldown_seconds:
            return jsonify({"people": 0, "status": "Waiting for cooldown..."})
        else:
            last_people_count = count_now
            last_detected_time = current_time
            return jsonify({"people": count_now, "status": "New detection after cooldown"})

    last_people_count = count_now
    last_detected_time = current_time
    return jsonify({"people": count_now, "status": "Stable"})

@app.route('/snapshot')
def snapshot():
    success, frame = camera.read()
    if not success:
        return "Camera error", 500
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    print("🚀 Menjalankan server deteksi di http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)
