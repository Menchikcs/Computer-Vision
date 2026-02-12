import cv2
import os
from ultralytics import YOLO
import time
import csv
import yt_dlp

def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        "format": "best",
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]
PROJECT_DIR = os.path.dirname(__file__)

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_DIR = os.path.join(OUTPUT_DIR, 'videos')

INPUT_VIDEO_PATH = os.path.join(VIDEO_DIR, 'video.mp4')
OUTPUT_VIDEO_PATH = os.path.join(VIDEO_DIR, 'output_video.mp4')

USE_WEBCAM = False
USE_YOUTUBE = True

YOUTUBE_URL = "https://www.youtube.com/live/Lxqcg1qt0XU?si=P9eHvMmjf9wfak6b"

if USE_WEBCAM:
    source = 0

elif USE_YOUTUBE:
    print("Connecting to YouTube stream...")
    stream_url = get_youtube_stream_url(YOUTUBE_URL)
    source = stream_url

else:
    source = INPUT_VIDEO_PATH


MODEL_PATH = "yolov8n.pt"

CONF_THRESH = 0.5

TRACKER = "bytetrack.yaml" #стандартний YOLO трекер

SAVE_VIDEO = True

model = YOLO(MODEL_PATH)

def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        "format": "best[height<=480]",
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


cap = cv2.VideoCapture(source)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps:
    fps = 30

writer = None

if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

seen_id_total = set() #множина унікальних айді

seen_id_class = {} #айді наших класів

LINE1_Y = 450
LINE2_Y = 650
REAL_DISTANCE_METERS = 10

vehicle_data = {}      # зберігаємо час старту і стартову лінію
vehicle_speeds = {}    # вже пораховані швидкості
previous_centers = {}  # попередні координати центру

CSV_FILE = os.path.join(OUTPUT_DIR, "speeds.csv")

with open(CSV_FILE, mode="w", newline="") as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(["Vehicle_ID", "Speed_km_h"])


while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.track(frame, conf=CONF_THRESH, tracker=TRACKER, persist=True, verbose=True)

    r = result[0]

    if r.boxes is None or len(r.boxes) == 0:
        cv2.imshow('frame', frame)

        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = r.boxes

    xyxy = boxes.xyxy.cpu().numpy()

    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    track_id = None
    if boxes.id is not None:
        track_id = boxes.id.cpu().numpy()

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].astype(int)
        class_id = int(cls[i])
        class_name = model.names[class_id]
        score = conf[i]

        VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]

        if class_id not in VEHICLE_CLASS_IDS:
            continue

        tid = -1
        if track_id is not None:
            tid = int(track_id[i])

        if tid == -1:
            continue

        # Центр авто
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Попередній центр
        prev_cy = previous_centers.get(tid, None)
        previous_centers[tid] = cy

        # ==========================
        # ЛОГІКА ПЕРЕТИНУ ЛІНІЙ
        # ==========================

        if prev_cy is not None:

            # Перетин LINE1
            if (prev_cy < LINE1_Y and cy >= LINE1_Y) or (prev_cy > LINE1_Y and cy <= LINE1_Y):

                if tid not in vehicle_data:
                    vehicle_data[tid] = {
                        "start_time": time.time(),
                        "start_line": 1
                    }

                elif vehicle_data[tid]["start_line"] == 2 and tid not in vehicle_speeds:

                    elapsed = time.time() - vehicle_data[tid]["start_time"]

                    if elapsed > 0:
                        speed = (REAL_DISTANCE_METERS / elapsed) * 3.6
                        vehicle_speeds[tid] = speed

                        with open(CSV_FILE, mode="a", newline="") as f:
                            writer_csv = csv.writer(f)
                            writer_csv.writerow([tid, round(speed, 2)])

            # Перетин LINE2
            if (prev_cy < LINE2_Y and cy >= LINE2_Y) or (prev_cy > LINE2_Y and cy <= LINE2_Y):

                if tid not in vehicle_data:
                    vehicle_data[tid] = {
                        "start_time": time.time(),
                        "start_line": 2
                    }

                elif vehicle_data[tid]["start_line"] == 1 and tid not in vehicle_speeds:

                    elapsed = time.time() - vehicle_data[tid]["start_time"]

                    if elapsed > 0:
                        speed = (REAL_DISTANCE_METERS / elapsed) * 3.6
                        vehicle_speeds[tid] = speed

                        with open(CSV_FILE, mode="a", newline="") as f:
                            writer_csv = csv.writer(f)
                            writer_csv.writerow([tid, round(speed, 2)])

        label = f'{class_name} {score:.2f} ID {tid}'
        if tid in vehicle_speeds:
            label += f' {vehicle_speeds[tid]:.1f} km/h'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.line(frame, (0, LINE1_Y), (frame_width, LINE1_Y), (255, 0, 0), 2)
        cv2.line(frame, (0, LINE2_Y), (frame_width, LINE2_Y), (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()