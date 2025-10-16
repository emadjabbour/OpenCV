import cv2
import os
import time

# IP camera stream
stream_url = "http://192.168.1.107:8080/video"

# Haar cascade path
cascade_path = r"D:\opencv\data\haarcascades\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Folder to save face images
dataset_path = r"D:\opencv\data\face_dataset"
os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(stream_url)

# Set frame size
frame_width = 1000
frame_height = 600
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Optional: lower brightness/exposure
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # adjust 0-255 as needed
cap.set(cv2.CAP_PROP_EXPOSURE, -6)     # may need tuning for your camera

count = 0
capture_interval = 0.5  # seconds
last_capture_time = 0
MIN_FACE_SIZE = 200      # skip partial faces

while True:
    # Grab latest frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame
    frame = cv2.resize(frame, (frame_width, frame_height))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    current_time = time.time()
    for (x, y, w, h) in faces:
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue  # skip small/partial faces

        # Draw rectangle continuously
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Capture face every 500ms
        if current_time - last_capture_time >= capture_interval:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(dataset_path, f"user_{count}.jpg"), face_img)
            last_capture_time = current_time

    cv2.imshow("Capturing Faces", frame)

    # Exit on ESC or after 100 images
    if cv2.waitKey(1) & 0xFF == 27 or count >= 100:
        break

cap.release()
cv2.destroyAllWindows()
print("Dataset captured in D:\\opencv\\data\\face_dataset")
