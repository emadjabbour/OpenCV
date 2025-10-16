import cv2

# Replace with your actual stream URL
stream_url = "http://192.168.1.107:8080/video"

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Cannot open stream. Check IP and port.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Mobile Stream", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
