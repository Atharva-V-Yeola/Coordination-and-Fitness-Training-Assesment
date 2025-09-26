import cv2
import time
import threading
import pyttsx3
from ultralytics import YOLO

# ---------------------------
# Shared Stop Flag
# ---------------------------
stop_event = threading.Event()

# ---------------------------
# YOLO Model
# ---------------------------
model = YOLO("runs/detect/all_in_one_yolov82/weights/best.pt")

# ---------------------------
# TTS Engine
# ---------------------------
engine = pyttsx3.init('espeak')

# ---------------------------
# Thread 1 → YOLO + Counting
# ---------------------------
def yolo_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    line_x = 300  # region line
    seen = set()
    count = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection with tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else []

            for (x1, y1, x2, y2), cls, oid in zip(boxes, clss, ids):
                label = model.names[cls]
                if label == "ball":
                    cx = (x1 + x2) // 2
                    if cx > line_x and oid not in seen:
                        count += 1
                        seen.add(oid)
                        print(f"[YOLO] Ball crossed! Count = {count}")

        # Draw line & overlay
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 2)
        cv2.putText(frame, f"Count: {count}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        cv2.imshow("YOLO Ball Counter", frame)

        # Press 'q' to stop everything
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[YOLO] Final Count: {count}")

# ---------------------------
# Thread 2 → TTS & Timing
# ---------------------------
def tts_loop():
    for cnt in range(3):
        if stop_event.is_set():
            break

        engine.say("Ready")
        engine.runAndWait()
        time.sleep(2)

        for word in ['One','Two','Three','Four','Five']:
            if stop_event.is_set():
                break
            engine.say(word)
            engine.runAndWait()
            time.sleep(1)

        if cnt < 2:
            engine.say("30 seconds break")
            engine.runAndWait()
            for _ in range(30):
                if stop_event.is_set():
                    break
                time.sleep(1)

    if not stop_event.is_set():
        engine.say("Congratulations! You have successfully completed the task. Great job!")
        engine.runAndWait()

# ---------------------------
# Run Threads
# ---------------------------
if __name__ == "__main__":
    t1 = threading.Thread(target=yolo_loop)
    t2 = threading.Thread(target=tts_loop)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("[INFO] Program stopped gracefully.")
