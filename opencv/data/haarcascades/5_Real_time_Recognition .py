import cv2

stream_url = "http://192.168.1.107:8080/video"
cascade_path = r"D:\opencv\data\haarcascades\haarcascade_frontalface_default.xml"
trainer_path = r"D:\opencv\data\trainer.yml"

face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_img)
        name = "You" if confidence < 70 else "Unknown"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {int(confidence)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    cv2.imshow("Face Recognition", frame)
    
    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for ESC
        break

cap.release()
cv2.destroyAllWindows()
