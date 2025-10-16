import cv2

stream_url = "http://192.168.1.107:8080/video"
cascade_path = r"D:\opencv\data\haarcascades\haarcascade_frontalface_default.xml"
trainer_path = r"D:\opencv\data\trainer.yml"

face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

cap = cv2.VideoCapture(stream_url)

# Set smaller frame width and height
frame_width = 1000
frame_height = 600

skip_frames = 5  # Process every 5th frame
frame_count = 0

MIN_FACE_SIZE = 150  # Minimum width/height in pixels

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (frame_width, frame_height))

    frame_count += 1
    if frame_count % skip_frames == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue  # Skip small false positives

            face_img = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_img)
            name = "Emad" if confidence < 70 else "Unknown"
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {int(confidence)}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Always draw rectangles for visible large faces even on skipped frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
