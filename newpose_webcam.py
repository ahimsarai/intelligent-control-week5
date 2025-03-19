from ultralytics import YOLO
import cv2
import numpy as np

# Load model YOLOv8 Pose
model = YOLO("yolov8m-pose.pt")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Indeks titik kunci untuk tangan dan kaki (berdasarkan model YOLOv8 Pose)
KEYPOINTS_TO_KEEP = {
    0: "Hidung",
    1: "Mata Kiri",
    2: "Mata Kanan",
    3: "Telinga Kiri",
    4: "Telinga Kanan",
    5: "Bahu Kiri",
    6: "Bahu Kanan",
    7: "Siku Kiri",
    8: "Siku Kanan",
    9: "Pergelangan Tangan Kiri",
    10: "Pergelangan Tangan Kanan",
    11: "Lutut Kiri",
    12: "Lutut Kanan",
    13: "Pergelangan Kaki Kiri",
    14: "Pergelangan Kaki Kanan",
    15: "Jari Kaki Kiri",
    16: "Jari Kaki Kanan"
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Deteksi pose
    results = model(frame)
    
    for result in results:
        annotated_frame = frame.copy()
        
        for kp in result.keypoints.xy:
            for i, name in KEYPOINTS_TO_KEEP.items():
                x, y = int(kp[i][0]), int(kp[i][1])
                cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)  # Gambar titik sendi
                cv2.putText(annotated_frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
        cv2.imshow("Filtered YOLOv8 Pose Estimation", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()