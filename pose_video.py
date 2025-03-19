from ultralytics import YOLO
 # Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")
 # Deteksi pose pada video
results = model("WhatsApp Video 2025-03-13 at 09.45.46 (1).mp4", save=True, show=True)