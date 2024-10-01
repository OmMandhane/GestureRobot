import cv2
import mediapipe as mp
import numpy as np

class GestureRecognizerWithDirection:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Define directions
        self.directions = ['forward', 'stop', 'left', 'right']
    
    def calculate_finger_angle(self, hand_landmarks):
        """
        Calculate the angle of the index finger relative to the wrist.
        """
        index_tip = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
        wrist = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                          hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y])
        
        # Calculate the vector from wrist to index finger tip
        finger_vector = index_tip - wrist
        
        # Calculate the angle of this vector relative to the horizontal axis
        angle = np.arctan2(finger_vector[1], finger_vector[0])  # Angle in radians
        return angle

    def classify_direction(self, hand_landmarks):
        # Calculate the angle based on hand landmarks
        angle = self.calculate_finger_angle(hand_landmarks)
        
        # Classify the angle into one of the four directions
        if -np.pi/4 < angle < np.pi/4:
            return 'right'
        elif np.pi/4 <= angle < 3*np.pi/4:
            return 'stop'
        elif -3*np.pi/4 <= angle < -np.pi/4:
            return 'forward'
        else:
            return 'left'

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Classify direction
                    direction = self.classify_direction(hand_landmarks)
                    
                    # Print results
                    print(f"Direction: {direction}")
                    
                    # Visualize the result on the image
                    cv2.putText(image, f"Direction: {direction}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    recognizer = GestureRecognizerWithDirection()
    recognizer.run()
