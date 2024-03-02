import cv2
import time
import datetime
# Load the cascade classifiers for face and smile
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

smile_detected = False
last_smile_time = 0

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(face_roi, 1.3, 20)

        if len(smiles) > 0:
            if not smile_detected:
                smile_detected = True
                last_smile_time = time.time()
            if time.time() - last_smile_time >= 0.8:  # 0.8 seconds have passed since the smile was first detected
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Generate a timestamp
                filename = f"/home/bbpi/Documents/mirror_cam/smile_pics/smile_photo_{timestamp}.jpg"  # Create a filename with the timestamp
                cv2.imwrite(filename, img)  # Save the photo in the specified folder with the timestamp
                print("Smile captured!")
                smile_detected = False  # Reset smile detection
        else:
            smile_detected = False  # Reset smile detection if no smiles are currently detected

    cv2.imshow('Smile Detector', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

