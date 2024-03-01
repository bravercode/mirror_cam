import cv2
import time
import datetime
import RPi.GPIO as GPIO

# Initialize GPIO for PIR sensor
PIR_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

# Load the cascade classifiers for face and smile
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Function to handle smile detection and image saving
def detect_smile_and_save():
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    
    while (time.time() - start_time) < 60:  # Keep the script running for 1 minute
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            smiles = smile_cascade.detectMultiScale(face_roi, 1.8, 20)

            if len(smiles) > 0:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"/home/bbpi/Documents/mirror_cam/smile_pics/pir_smile_photo_{timestamp}.jpg"
                cv2.imwrite(filename, img)
                print("Smile captured!")

        cv2.imshow('Smile Detector', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

try:
    print("PIR Module Test (CTRL+C to exit)")
    time.sleep(2)  # Allow PIR to stabilize

    while True:
        if GPIO.input(PIR_PIN):
            print("Motion Detected! Starting smile detection.")
            detect_smile_and_save()
        else:
            print("Waiting for motion...")
        time.sleep(1)  # Check every second for PIR input

except KeyboardInterrupt:
    print("Quit")
    GPIO.cleanup()
