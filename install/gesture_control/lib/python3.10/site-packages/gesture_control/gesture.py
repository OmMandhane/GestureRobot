import cv2
import mediapipe as mp
import numpy as np
from scipy.stats import norm

class GestureRecognizerWithDirection:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Define directions
        self.directions = ['forward', 'backward', 'left', 'right']
        
        # Prior probabilities for directions (even distribution)
        self.priors = {direction: 1/len(self.directions) for direction in self.directions}
        
        # Likelihood models based on angles (in radians)
        self.likelihoods = {
            'forward': norm(loc=-np.pi / 2, scale=0.4),  # Upward (90 degrees or pi/2 radians)
            'backward': norm(loc=np.pi / 2, scale=0.4),  # Downward (-90 degrees or -pi/2 radians)
            'left': norm(loc=np.pi, scale=0.6),  # Leftward (180 degrees or pi radians)
            'right': norm(loc=0, scale=0.6)  # Rightward (0 degrees or 0 radians)
        }
    
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

    def bayesian_inference_for_direction(self, angle):
        """
        Perform Bayesian inference to classify the direction based on finger angle.
        """
        posteriors = {}
        for direction in self.directions:
            # Calculate the likelihood of this angle for each direction
            likelihood = self.likelihoods[direction].pdf(angle)
            # Prior probability for each direction
            prior = self.priors[direction]
            # Compute the posterior probability for each direction
            posterior = likelihood * prior
            posteriors[direction] = posterior

        # Normalize the posteriors so they sum to 1
        total = sum(posteriors.values())
        posteriors = {direction: posterior / total for direction, posterior in posteriors.items()}
        
        return posteriors

    def classify_direction(self, hand_landmarks):
        # Calculate the angle based on hand landmarks
        angle = self.calculate_finger_angle(hand_landmarks)
        
        # Perform Bayesian inference to classify the direction based on angle
        posteriors = self.bayesian_inference_for_direction(angle)
        
        # Select the direction with the highest posterior probability
        best_direction = max(posteriors, key=posteriors.get)
        
        return best_direction

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
