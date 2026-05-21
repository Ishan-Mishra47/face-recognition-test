from flask import Flask, render_template, Response, jsonify
import cv2
import sys
import threading
import os 

sys.path.append(os.path.join(os.path.dirname(__file__), "main"))
import face_rec

app = Flask(__name__)

counter = 0
def generate_frames():
    global counter
    while True:
        ret, frame = face_rec.capture.read()
        if ret:
            if counter%30==0:
                try:
                    threading.Thread(target=face_rec.check_face, args=(frame.copy(),)).start()
                except Exception:
                    pass
            counter+=1
            ret, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({"match": face_rec.face_check})

if __name__ == "__main__":
    app.run(debug=False)