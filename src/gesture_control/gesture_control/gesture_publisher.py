import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import cv2
import mediapipe as mp
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class GestureRecognizerWithDirection(Node):
    def __init__(self):
        super().__init__('gesture_recognizer')

        # Create a publisher for TwistStamped messages
        self.publisher = self.create_publisher(TwistStamped, 'new_robot_controller/cmd_vel', 10)

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.08)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Define directions
        self.directions = ['forward', 'stop', 'left', 'right']
        
        # Prior probabilities for directions (even distribution)
        self.priors = {direction: 1/len(self.directions) for direction in self.directions}
        
        # Likelihood models based on angles (in radians)
        self.likelihoods = {
            'forward': norm(loc=-np.pi / 2, scale=0.6),  # Upward (90 degrees or pi/2 radians)
            'stop': norm(loc=np.pi / 2, scale=0.6),  # Downward (-90 degrees or -pi/2 radians)
            'left': [norm(loc=np.pi, scale=0.4), norm(loc=-np.pi, scale=0.4)],  # Leftward (180 degrees or pi radians)
            'right': norm(loc=0, scale=0.4)  # Rightward (0 degrees or 0 radians)
        }
        self.fixed_likelihoods = {
            'forward': 0.25,
            'left': 0.25,
            'right': 0.25,
            'stop': 0.25
        }

        # Initialize real-time plot for posterior visualization
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(0, 1)
        self.bars = self.ax.bar(self.directions, [0.25, 0.25, 0.25, 0.25])  # Initial bars
        plt.ion()  # Interactive mode for real-time updates
        plt.show()

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

        # Calculate the likelihood for the 'left' direction using the closest distribution
    def calculate_left_likelihood(self, angle):
        # Wrap the angle to the range [-pi, pi]
        angle = np.arctan2(np.sin(angle), np.cos(angle))  # Wraps the angle

        # Calculate the circular distance to both pi and -pi
        distance_to_pi = abs(angle - np.pi)
        distance_to_minus_pi = abs(angle - (-np.pi))
        
        # Use the closest distribution based on the distance
        if distance_to_pi < distance_to_minus_pi:
            likelihood = self.likelihoods['left'][0].pdf(angle)  # Use N(pi, 0.6)
        else:
            likelihood = self.likelihoods['left'][1].pdf(angle)  # Use N(-pi, 0.6)
        
        return likelihood    

    def update_priors(self, detected_direction):
        """
        Update priors dynamically based on the detected direction.
        """
        # Reset all priors to an even distribution
        self.priors = {direction: 1/len(self.directions) for direction in self.directions}
        
        # Update the prior to heavily favor the detected direction
        self.priors[detected_direction] = 0.7  # Increase this prior significantly
        
        # Re-normalize the priors to ensure they sum to 1
        total = sum(self.priors.values())
        self.priors = {direction: prior / total for direction, prior in self.priors.items()}

    def bayesian_inference_for_direction(self, angle=None):
        """
        Perform Bayesian inference to classify the direction based on finger angle.
        """
        posteriors = {}
        for direction in self.directions:
            # Use likelihood based on angle if available
            if angle is not None:
                if direction == 'left':
                    likelihood = self.calculate_left_likelihood(angle)
                else:    
                    likelihood = self.likelihoods[direction].pdf(angle)
            else:
                # Use fixed likelihoods when no angle is detected
                likelihood = self.fixed_likelihoods[direction]
            
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
        
        # Visualize the posterior probabilities as a real-time graph
        self.visualize_posteriors(posteriors)
        
        # Select the direction with the highest posterior probability
        best_direction = max(posteriors, key=posteriors.get)
        
        return best_direction

    def visualize_likelihoods(self):
        """
        Visualize the likelihood distributions for each direction.
        """
        angles = np.linspace(-np.pi, np.pi, 1000)  # Generate angles from -π to π (radians)

        plt.figure(figsize=(10, 6))
        
        # Plot each direction's likelihood distribution
        for direction, likelihood in self.likelihoods.items():
            if direction == 'left':
                # For the 'left' direction, we consider both distributions
                pdf_left_pos = likelihood[0].pdf(angles)  # N(pi, 0.6)
                pdf_left_neg = likelihood[1].pdf(angles)  # N(-pi, 0.6)
                # Combine the two distributions for visualization
                combined_pdf = pdf_left_pos + pdf_left_neg
                plt.plot(angles, combined_pdf, label=direction + " (combined)")
            else:
                pdf_values = likelihood.pdf(angles)
                plt.plot(angles, pdf_values, label=direction)
        
        plt.title("Likelihood Distributions for Directions")
        plt.xlabel("Angle (radians)")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_posteriors(self, posteriors):
        """
        Update the bar plot with the posterior probabilities.
        """
        for i, direction in enumerate(self.directions):
            self.bars[i].set_height(posteriors[direction])

        self.ax.set_title('Posterior Probabilities for Gestures')
        self.ax.set_ylabel('Probability')
        self.ax.set_xlabel('Gestures')
        plt.draw()  # Update the plot
        plt.pause(0.01)  # Add a small pause to allow for real-time updates

    def publish_velocity(self, direction):
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()

        # Set velocity based on direction
        if direction == 'forward':
            twist.twist.linear.x = 1.0
            twist.twist.angular.z = 0.0
        elif direction == 'stop':
            twist.twist.linear.x = 0.0
            twist.twist.angular.z = 0.0
        elif direction == 'left':
            twist.twist.linear.x = 0.5  # Forward component
            twist.twist.angular.z = 0.5  # Turning left
        elif direction == 'right':
            twist.twist.linear.x = 0.5  # Forward component
            twist.twist.angular.z = -0.5  # Turning right

        self.publisher.publish(twist)
        self.get_logger().info(f'Published velocity for direction: {direction}')

    def run(self):
        self.visualize_likelihoods()
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                self.get_logger().warn("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Classify direction
                    direction = self.classify_direction(hand_landmarks)
                    self.update_priors(direction)
                    # Publish velocity based on recognized direction
                    self.publish_velocity(direction)

                    # Visualize the result on the image
                    cv2.putText(image, f"Direction: {direction}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # No hand detected, use Bayesian inference with fixed likelihoods
                posteriors = self.bayesian_inference_for_direction(angle=None)
                # Select the direction with the highest posterior probability
                self.visualize_posteriors(posteriors)
                direction = max(posteriors, key=posteriors.get)
                                # Visualize the result on the image
                self.priors = posteriors               
                cv2.putText(image, f"Direction: {direction}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.publish_velocity(direction)  # Publish velocity for the best direction

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                break

        cap.release()
        cv2.destroyAllWindows()

   

def main(args=None):
    rclpy.init(args=args)
    recognizer = GestureRecognizerWithDirection()
    recognizer.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
