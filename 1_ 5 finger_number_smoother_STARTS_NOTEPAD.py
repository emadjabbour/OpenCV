import cv2
import mediapipe as mp
import time
import subprocess

# ------------ settings ------------
ip_camera_url = "http://192.168.1.111:4747/video"

# ---- kill previous camera session if exists ----
try:
    cap.release()
except:
    pass

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.4
)

cap = cv2.VideoCapture(ip_camera_url, cv2.CAP_ANY)

if not cap.isOpened():
    print("ERROR camera")
    exit()

pTime = 0
notepad_started = False   # <---- very important flag

def count_fingers(lm):
    tip = [4,8,12,16,20]
    fingers = []

    # thumb
    fingers.append(1 if lm.landmark[tip[0]].x < lm.landmark[tip[0]-1].x else 0)

    # other
    for i in range(1,5):
        fingers.append(1 if lm.landmark[tip[i]].y < lm.landmark[tip[i]-2].y else 0)

    return sum(fingers)

while True:
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img,1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)
            
            fingers = count_fingers(lm)
            cv2.putText(img, f"Fingers: {fingers}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3)

            # -------- ACTION --------
            if fingers == 5 and not notepad_started:
                subprocess.Popen(["notepad.exe"])
                notepad_started = True
                print("Notepad started!")

            if fingers != 5:
                notepad_started = False   # reset when hand changes

    # FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Hand Control", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
