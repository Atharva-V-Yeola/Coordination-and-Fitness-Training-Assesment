
# Commands Trails


import cv2
import mediapipe as mp
import pyttsx3
import time

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # live webcam

# Game parameters
SETS_PER_LEG = 3
HOLD_TIME = 20  # seconds
REST_TIME = 30  # seconds
threshold = 0.05  # leg difference threshold

current_leg = "LEFT"  # first leg (can alternate)
sets_done = 0
hold_start = None
rest_start = None
in_rest = False

speak("Balance game started. Get ready on your left leg.")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            diff = abs(left_ankle.y - right_ankle.y)

            if not in_rest:
                if diff > threshold:  # standing on one leg
                    if hold_start is None:
                        hold_start = time.time()
                        speak("Hold steady")

                    elapsed = time.time() - hold_start
                    cv2.putText(image, f"Holding... {int(HOLD_TIME - elapsed)} sec left",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if elapsed >= HOLD_TIME:
                        sets_done += 1
                        speak(f"Set {sets_done} complete. Take rest for {REST_TIME} seconds.")
                        in_rest = True
                        rest_start = time.time()
                        hold_start = None

                        if sets_done >= SETS_PER_LEG:
                            speak("All sets completed. Game over. Great job!")
                            break
                else:
                    hold_start = None
                    cv2.putText(image, "Both legs down - reset!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                elapsed_rest = time.time() - rest_start
                cv2.putText(image, f"Resting... {int(REST_TIME - elapsed_rest)} sec left",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                if elapsed_rest >= REST_TIME:
                    in_rest = False
                    speak("Start next set. Balance again.")

        cv2.imshow('Balance Statue Game', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
