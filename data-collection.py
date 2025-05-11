import cv2
import mediapipe as mp
import time
import tkinter as tk
from tkinter import simpledialog
import numpy as np
import csv
import os
import screeninfo

mp_face_mesh = mp.solutions.face_mesh

def get_user_input():
    root = tk.Tk()
    root.withdraw()
    user_name = simpledialog.askstring("Kullanıcı Adı", "Adınızı girin:")
    screen_distance = simpledialog.askfloat("Ekran Mesafesi", "Ekrana olan mesafenizi (cm) girin:")
    return user_name, screen_distance

def show_calibration_button():
    def start_calibration():
        user_name, screen_distance = get_user_input()
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        calibration_screen(user_name, screen_distance, timestamp)

    root = tk.Tk()
    root.title("Kalibrasyon")
    root.geometry("300x150")
    label = tk.Label(root, text="Kalibrasyona başlamak için butona basın", font=("Helvetica", 12))
    label.pack(pady=20)

    button = tk.Button(root, text="Kalibrasyona Başla", font=("Helvetica", 14), command=start_calibration)
    button.pack(pady=10)

    root.mainloop()

def calibration_screen(user_name, screen_distance, timestamp):
    # Gerçek ekran çözünürlüğünü al
    screen = screeninfo.get_monitors()[0]
    screen_width = screen.width
    screen_height = screen.height

    padding_x = 100
    padding_y = 100
    cols = 5
    rows = 3

    available_width = screen_width - 2 * padding_x
    available_height = screen_height - 2 * padding_y

    x_spacing = available_width // (cols - 1)
    y_spacing = available_height // (rows - 1)

    calibration_points = [(padding_x + col * x_spacing, padding_y + row * y_spacing)
                          for row in range(rows) for col in range(cols)]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı. Lütfen başka bir uygulamada açık olmadığından emin olun.")
        return

    calibration_data = []
    window_name = "Kalibrasyon"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    show_message = True  # İlk noktada mesajı göstermek için kontrol

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        for point in calibration_points:
            collected = False
            while not collected:
                ret, frame = cap.read()
                if not ret:
                    break

                flipped = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                calib_screen = np.full((screen_height, screen_width, 3), 128, dtype=np.uint8)

                # İlk noktada "Hazırsanız SPACE'e basın" mesajını göster
                if show_message:
                    cv2.putText(calib_screen, "Hazirsaniz SPACE'e basin", (screen_width // 2 - 150, screen_height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    show_message = False  # Mesajı yalnızca bir kez göster

                # Şu anki nokta için çizim yapıyoruz
                cv2.circle(calib_screen, point, 20, (255, 0, 0), -1)
                cv2.imshow(window_name, calib_screen)

                # Gözbebeklerini ekranda çizme
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    left_iris = face_landmarks.landmark[468]
                    right_iris = face_landmarks.landmark[473]

                    left = (int(left_iris.x * screen_width), int(left_iris.y * screen_height))
                    right = (int(right_iris.x * screen_width), int(right_iris.y * screen_height))

                    cv2.circle(calib_screen, left, 10, (0, 255, 0), -1)  # Sol gözbebeği
                    cv2.circle(calib_screen, right, 10, (0, 255, 0), -1)  # Sağ gözbebeği


                start_time = time.time()
                while time.time() - start_time < 2:  # 2 saniye beklemeden veri alma
                    if cv2.waitKey(1) & 0xFF == 32:  # SPACE tuşuna basıldıysa
                        if results.multi_face_landmarks:
                            face_landmarks = results.multi_face_landmarks[0]

                            left_iris = face_landmarks.landmark[468]
                            right_iris = face_landmarks.landmark[473]

                            left = (int(left_iris.x * screen_width), int(left_iris.y * screen_height))
                            right = (int(right_iris.x * screen_width), int(right_iris.y * screen_height))

                            # Gerçek gözbebekleri arası mesafeyi (60mm) ve anlık mesafeyi hesapla
                            real_pupil_distance = 60  # Gerçek gözbebekleri arası mesafe (mm)
                            pupil_distance = np.linalg.norm(np.array(left) - np.array(right))

                            calibration_data.append([ 
                                user_name, screen_distance,
                                point[0], point[1],
                                left[0], left[1],
                                right[0], right[1],
                                real_pupil_distance, pupil_distance
                            ])

                            print(f"{point} noktasında veri kaydedildi: sol göz {left}, sağ göz {right}, gerçek gözbebekleri arası mesafe: {real_pupil_distance} mm, anlık gözbebekleri arası mesafe: {pupil_distance}")
                            collected = True
                        else:
                            print("Yüz algılanamadı, tekrar deneyin.")
                if not collected:
                    print("Zaman aşımı! Lütfen tekrar deneyin.")

    cap.release()
    cv2.destroyAllWindows()

    # CSV dosyasına yazma işlemi
    csv_path = "gaze_data.csv"
    write_header = not os.path.exists(csv_path)

    try:
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow([  # Başlık satırını yazıyoruz
                    "user_name", "screen_distance",
                    "target_x", "target_y",
                    "left_x", "left_y",
                    "right_x", "right_y",
                    "real_pupil_distance", "pupil_distance"
                ])
            writer.writerows(calibration_data)

        print(f"Kalibrasyon tamamlandı. Veriler '{csv_path}' dosyasına kaydedildi.")
    except PermissionError:
        print("CSV dosyasına yazılamadı. Dosya açık olabilir, lütfen kapatın.")

if __name__ == "__main__":
    show_calibration_button()
