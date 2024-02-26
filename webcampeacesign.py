import cv2
import mediapipe as mp
import time  # For timestamping saved images
import os
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
save_folder = '/home/bbpi/Documents/mirror_cam/pics'

# Initialize webcam
cap = cv2.VideoCapture(2)  # Change the device index if your webcam is not the default

def is_peace_sign(hand_landmarks):
    # Extract y-coordinates of fingertips
    index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y
    
    # Extract x-coordinates to check if fingers are apart
    index_finger_tip_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x
    middle_finger_tip_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x
    
    # Relaxing conditions: Slightly lesser y-coordinate difference and checking fingers are apart
    y_diff_threshold = 0.2  # Adjust based on sensitivity needs
    x_diff_min = 0.1  # Adjust for sensitivity to how far apart fingers need to be
    
    if (index_finger_tip < (ring_finger_tip - y_diff_threshold) and 
        middle_finger_tip < (ring_finger_tip - y_diff_threshold) and 
        abs(index_finger_tip_x - middle_finger_tip_x) > x_diff_min):
        return True
    return False




try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Check if a peace sign is detected and save the image
                if is_peace_sign(hand_landmarks):
                    filename = f"peace_sign_{int(time.time())}.jpg"
                    filepath = os.path.join(save_folder, filename)  # This creates the full path to save the file
                    cv2.imwrite(filepath, image)


        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
