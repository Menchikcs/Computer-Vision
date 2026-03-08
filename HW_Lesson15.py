import cv2 as cv
import os
import time
from ultralytics import YOLO
import yt_dlp

PROJECT_DIR = os.path.dirname(__file__)

YOUTUBE_URL = "https://youtu.be/gcUHp8Wm7D0?si=Nrn98S2oiK8Ltlkd"

def get_youtube_stream_url(url):
    ydl_opts = {
        "format": "best[ext=mp4]",
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]


stream_url = get_youtube_stream_url(YOUTUBE_URL)

cap = cv.VideoCapture(stream_url)

model = YOLO("yolov8n.pt")

CONF_THRESHOLD = 0.4
RESIZE_WIDTH = 960

CAT_CLASS_ID = 15
DOG_CLASS_ID = 16

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

    cat_count = 0
    dog_count = 0

    for r in results:

        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == CAT_CLASS_ID:
                cat_count += 1
                cv.rectangle(frame, (x1,y1),(x2,y2),(255,0,0),2)
                cv.putText(frame,f"Cat {conf:.2f}",(x1,y1-10), cv.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

            if cls == DOG_CLASS_ID:
                dog_count += 1
                cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv.putText(frame,f"Dog {conf:.2f}",(x1,y1-10), cv.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    total_animals = cat_count + dog_count

    now = time.time()
    dt = now - prev_time
    prev_time = now

    if dt > 0:
        fps = 1.0 / dt

    cv.putText(frame,f"Cats: {cat_count}",(20,40), cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv.putText(frame,f"Dogs: {dog_count}",(20,80), cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv.putText(frame,f"Total animals: {total_animals}",(20,120), cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv.putText(frame,f"FPS: {fps:.1f}",(20,160), cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv.imshow("YOLO Animals Detection", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()