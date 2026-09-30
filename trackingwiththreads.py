import cv2
import mediapipe as mp
import serial
import threading
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
cap = cv2.VideoCapture(1)

ArduinoSerial = serial.Serial('com4', 9600)

latest_results = None
results_lock = threading.Lock()


def send_data_thread():
    while True:
        with results_lock:
            results = latest_results
        if results and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmark = face_landmarks.landmark[0]
                x = int(landmark.x * 800)
                y = int(landmark.y * 400)
                z = 0
                string = 'X{0:d}Y{1:d}Z{2:d}'.format(x, y, z)
                ArduinoSerial.write(string.encode('utf-8'))
                print(string)
        time.sleep(0.05)


sender = threading.Thread(target=send_data_thread, daemon=True)
sender.start()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (800, 400))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    with results_lock:
        latest_results = results

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('Face Mesh', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
