from flask import Flask, render_template, Response
import cv2
import threading
import winsound
import atexit

# Initialize Flask app
app = Flask(__name__)

# Setup webcam
cap = cv2.VideoCapture(0)

# Verify camera works
if not cap.isOpened():
    raise RuntimeError("❌ Cannot access webcam. Make sure it's connected or not being used by another program.")

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Drowsiness detection variables
closed_eyes_count = 0
alarm_triggered = False
threshold_frames = 20

# Alarm sound function
def sound_alarm():
    try:
        winsound.PlaySound("alarm.wav", winsound.SND_FILENAME)
    except Exception as e:
        print("Error playing alarm:", e)

# Frame generator for video streaming
def generate_frames():
    global closed_eyes_count, alarm_triggered
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes_detected = False

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                eyes_detected = True
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        if not eyes_detected:
            closed_eyes_count += 1
        else:
            closed_eyes_count = 0
            alarm_triggered = False

        if closed_eyes_count >= threshold_frames:
            if not alarm_triggered:
                alarm_triggered = True
                threading.Thread(target=sound_alarm).start()
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Release camera on exit
@atexit.register
def cleanup():
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True)
