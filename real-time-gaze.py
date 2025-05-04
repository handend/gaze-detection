import cv2
import mediapipe as mp
import joblib
import numpy as np

# Modeli yükle
model = joblib.load("gaze_model.pkl")

# MediaPipe başlat
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Kamera
cap = cv2.VideoCapture(0)

# Ekran boyutu
screen_w, screen_h = 1920, 1080

# Gözler arasındaki mesafeyi hesapla
def get_iris_coords(landmarks, shape):
    ih, iw = shape
    left = landmarks[468]  # Sol göz
    right = landmarks[473]  # Sağ göz

    left_x, left_y = int(left.x * iw), int(left.y * ih)
    right_x, right_y = int(right.x * iw), int(right.y * ih)

    eye_distance = np.linalg.norm([left_x - right_x, left_y - right_y])

    return left_x, left_y, right_x, right_y, eye_distance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark
        left_x, left_y, right_x, right_y, eye_distance = get_iris_coords(mesh_points, frame.shape[:2])

        # Tahmin için giriş oluştur
        input_data = np.array([[left_x, left_y, right_x, right_y, 50, 50, eye_distance]])  # Burada screen_distance ve pupil_distance sabit
        predicted = model.predict(input_data)[0]

        # Tahmin edilen nokta ekran üzerinde çizilecek
        px, py = int(predicted[0] * screen_w), int(predicted[1] * screen_h)
        cv2.circle(frame, (px, py), 10, (0, 255, 0), -1)
        cv2.putText(frame, f"Gaze: ({px}, {py})", (px + 20, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Ekranı göster
    cv2.imshow("Real-Time Gaze Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
