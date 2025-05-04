import cv2
import mediapipe as mp

# MediaPipe tanımlamaları
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True, # Iris noktalarını da verir
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# Kamera başlat
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü çevir ve RGB'ye dönüştür
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Yüz mesh tespiti
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            # Sol iris merkezi: 468, sağ iris merkezi: 473
            left_iris = face_landmarks.landmark[468]
            right_iris = face_landmarks.landmark[473]

            left_coords = (int(left_iris.x * w), int(left_iris.y * h))
            right_coords = (int(right_iris.x * w), int(right_iris.y * h))

            # Göz bebeği üzerine daire çiz
            cv2.circle(frame, left_coords, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_coords, 3, (0, 255, 0), -1)

            # Koordinatları yazdır
            cv2.putText(frame, f"L: {left_coords}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            cv2.putText(frame, f"R: {right_coords}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

    cv2.imshow("Iris Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC ile çık
        break

cap.release()
cv2.destroyAllWindows()
