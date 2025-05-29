import cv2
from flask import Flask, jsonify, Response, stream_with_context

# === Minta index kamera dari user ===
def minta_input_index_kamera():
    try:
        index = int(input("📷 Masukkan index kamera (misal 0 atau 1): "))
        return index
    except ValueError:
        print("❌ Input tidak valid. Gunakan angka.")
        exit(1)

CAMERA_INDEX = minta_input_index_kamera()

# === Inisialisasi Kamera ===
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not camera.isOpened():
    raise RuntimeError(f"❌ Kamera index {CAMERA_INDEX} tidak bisa dibuka.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === Load Face Detector Haar Cascade ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Setup Flask ===
app = Flask(__name__)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(frame, f"Total faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(stream_with_context(gen_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect')
def detect_people():
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "kamera tidak tersedia"}), 500

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    return jsonify({"people": len(faces)})

@app.route('/snapshot')
def snapshot():
    success, frame = camera.read()
    if not success:
        return "Camera error", 500
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    print("🚀 Haar Cascade Server aktif di http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)
