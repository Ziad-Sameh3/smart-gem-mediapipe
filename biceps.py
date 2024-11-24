import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # Last point
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


# Start video capture from webcam
cap = cv2.VideoCapture("http://192.168.1.3:8080/video")

# Get video frame width, height, and frames per second (FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter to save the video
"""output_video = cv2.VideoWriter(
    "kettlebell_swing_video_output.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (frame_width, frame_height),
)"""

# Initialize counters and states
count_right = 0
count_left = 0
stage_right = None  # "up" or "down"
stage_left = None  # "up" or "down"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB for MediaPipe processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Process and visualize pose landmarks
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for bicep curl (shoulder, elbow, wrist)
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
        ]
        left_elbow = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y,
        ]
        left_wrist = [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y,
        ]

        right_shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
        ]
        right_elbow = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
        ]
        right_wrist = [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y,
        ]

        # Calculate elbow angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Bicep curl logic for repetitions
        if left_elbow_angle > 160:
            stage_left = "down"
        if left_elbow_angle < 40 and stage_left == "down":
            stage_left = "up"
            count_left += 1

        if right_elbow_angle > 160:
            stage_right = "down"
        if right_elbow_angle < 40 and stage_right == "down":
            stage_right = "up"
            count_right += 1

        # Display elbow angles on screen
        cv2.putText(
            image,
            f"Right Angle: {int(left_elbow_angle)}",
            (50, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (125, 125, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Left Angle: {int(right_elbow_angle)}",
            (50, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (125, 125, 255),
            2,
            cv2.LINE_AA,
        )
        # Display counts and stages for each arm
        cv2.putText(
            image,
            f"Count Left: {count_left}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Count Right: {count_right}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Stage Left: {stage_left}",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Stage Right: {stage_right}",
            (50, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw landmarks for visual reference
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )


    # Write the frame with annotations to the video file
    #output_video.write(image)
    
    cv2.imshow("Smart Gym - Bicep Curl Checker", image)

    # Press 'q' to exit
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release resources
cap.release()
# output_video.release()  # Save the video file
cv2.destroyAllWindows()
