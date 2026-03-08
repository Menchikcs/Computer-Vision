import cv2 as cv
import os
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, "video")

VIDEO_PATH = os.path.join(VIDEO_DIR, "13403360-hd_1920_1080_30fps.mp4")

cap = cv.VideoCapture(VIDEO_PATH)

model = YOLO("yolov8n.pt")

CONF_THRESHOLD = 0.4
RESIZE_WIDTH = 960

VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

prev_time = time.time()
fps = 0.0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame = cv.resize(frame, (int(w * scale), int(h * scale)))

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    counts = {
        "car": 0,
        "bus": 0,
        "truck": 0,
        "motorcycle": 0,
        "bicycle": 0
    }

    for r in results:

        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls in VEHICLE_CLASSES:

                label = VEHICLE_CLASSES[cls]
                counts[label] += 1

                cv.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv.putText(frame,f"{label} {conf:.2f}",(x1, max(20, y1-10)),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    now = time.time()
    dt = now - prev_time
    prev_time = now

    if dt > 0:
        fps = 1.0 / dt

    cv.putText(frame, f"Cars: {counts['car']}", (20,40),cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv.putText(frame, f"Buses: {counts['bus']}", (20,70),cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv.putText(frame, f"Trucks: {counts['truck']}", (20,100),cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv.putText(frame, f"Motorcycles: {counts['motorcycle']}", (20,130),cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv.putText(frame, f"Bicycles: {counts['bicycle']}", (20,160),cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv.putText(frame, f"FPS: {fps:.1f}", (20,200),cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv.imshow("YOLO Transport Detection", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()