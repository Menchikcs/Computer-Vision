import cv2 as cv
import yt_dlp
from ultralytics import YOLO

# 🔗 ВСТАВ СЮДИ ПОСИЛАННЯ НА YOUTUBE
YOUTUBE_URL = "https://youtu.be/eO19UTm93GQ?si=EN80isFrwj6D-QF7"

CONF_THRESH = 0.5

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

CLASS_NAMES = {2: 'car', 3: 'bike', 5: 'bus', 7: 'truck'}

CLASS_COLORS = {
    'car': (0, 255, 0),
    'bike': (0, 255, 255),
    'bus': (0, 0, 255),
    'truck': (255, 0, 255)
}


def get_stream_url(url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']


print("Отримання stream URL...")
stream_url = get_stream_url(YOUTUBE_URL)

# Завантаження моделі
model = YOLO('yolov8n.pt')

# Підключення до відео
cap = cv.VideoCapture(stream_url)

# 🔥 Зберігаємо унікальні ID
unique_vehicle_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Проблема зі стрімом або відео завершилось")
        break

    # Трекінг
    results = model.track(
        frame,
        conf=CONF_THRESH,
        classes=VEHICLE_CLASSES,
        persist=True
    )

    current_counts = {name: 0 for name in ['car', 'bike', 'bus', 'truck']}

    for result in results:

        boxes = result.boxes

        if boxes.id is None:
            continue

        for box, track_id in zip(boxes, boxes.id):

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            track_id = int(track_id)

            class_name = CLASS_NAMES.get(class_id, 'vehicle')
            color = CLASS_COLORS.get(class_name, (0, 255, 0))

            current_counts[class_name] += 1
            unique_vehicle_ids.add(track_id)

            # Малюємо рамку
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name} ID:{track_id}"
            cv.putText(frame,label,(x1, y1 - 10),cv.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    y_pos = 30
    cv.putText(frame,"Transport na kadri:",(10, y_pos),cv.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2)

    for vehicle, count in current_counts.items():
        y_pos += 25
        color = CLASS_COLORS.get(vehicle, (255, 255, 255))
        cv.putText(frame,f"{vehicle}: {count}",(10, y_pos),cv.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # Унікальні авто
    y_pos += 35
    cv.putText(frame,f"Unique vehicles: {len(unique_vehicle_ids)}",(10, y_pos),cv.FONT_HERSHEY_SIMPLEX,0.7,(0, 200, 255),2)

    cv.imshow("YOLO Transport Tracking (YouTube)", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
