from flask import Flask, render_template, request, jsonify, Response
import cv2
import os
import numpy as np

app = Flask(__name__)

# Load models
faceNet = cv2.dnn.readNet("models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt")
ageNet = cv2.dnn.readNet("models/age_net.caffemodel", "models/age_deploy.prototxt")
genderNet = cv2.dnn.readNet("models/gender_net.caffemodel", "models/gender_deploy.prototxt")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

def detect_age_gender(image):
    frame = image.copy()
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            x1, y1, x2, y2 = box.astype(int)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            blob_face = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob_face)
            gender = genderList[genderNet.forward()[0].argmax()]

            ageNet.setInput(blob_face)
            age = ageList[ageNet.forward()[0].argmax()]

            results.append({'box': [int(x1), int(y1), int(x2), int(y2)], 'gender': gender, 'age': age})

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    file_path = 'temp.jpg'
    file.save(file_path)

    image = cv2.imread(file_path)
    results = detect_age_gender(image)

    return jsonify({'results': results})

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_age_gender(frame)

        for result in results:
            box = result['box']
            gender = result['gender']
            age = result['age']

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{gender}, {age}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)
