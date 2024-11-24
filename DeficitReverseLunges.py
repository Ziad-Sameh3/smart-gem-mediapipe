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

# Initialize counter and stage
count_lunge = 0
stage_lunge = None  # "up" or "down"

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

        # Get coordinates for the legs (hip, knee, ankle for both legs)
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
        ]
        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
        ]
        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
        ]

        right_hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
        ]
        right_knee = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
        ]
        right_ankle = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
        ]

        # Calculate knee angles for both legs (to track the lunge position)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Lunge logic to detect stage and count repetitions
        # The left leg is the one that bends for the deficit reverse lunge (assuming the person is lunging with their left leg back)
        if left_knee_angle > 160 and right_knee_angle > 160:
            stage_lunge = "up"  # Standing position
        if (
            left_knee_angle < 90 and right_knee_angle > 160 and stage_lunge == "up"
        ):  # Left knee bends lower than 90, right leg is straight
            stage_lunge = "down"
            count_lunge += 1

        # Display count and feedback on screen
        cv2.putText(
            image,
            f"Lunge Count: {count_lunge}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"Stage: {stage_lunge}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # Display angles on screen
        cv2.putText(
            image,
            f"Left Angle: {str(int(left_knee_angle))}",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (125, 125, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Right Angle: {str(int(right_knee_angle))}",
            (50, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    # Write the frame with annotations to the video file
    # output_video.write(image)

    cv2.imshow("Smart Gym - Lunge Checker", image)

    # Press 'q' to exit
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release resources
cap.release()
# output_video.release()  # Save the video file
cv2.destroyAllWindows()
