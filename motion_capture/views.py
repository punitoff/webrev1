import cv2
from django.http import StreamingHttpResponse
from django.shortcuts import render
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Constants for defining hand movements
HAND_RAISE_THRESHOLD = 0.13
HAND_MOVE_THRESHOLD = 0.13
HEAD_MOVE_THRESHOLD = 0.13


# Add constants for frame persistence
HAND_MOVEMENT_FRAMES = 60  # (120 frames = about 4 seconds)
HEAD_MOVEMENT_FRAMES = 60  # (120 frames = about 4 seconds)

def generate_frames():
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        previous_hand_landmarks = [None, None]  # Storing previous landmarks for both hands
        previous_nose_position = None

        # Counters and stored boxes for frame persistence
        hand_movement_counter = [0, 0]
        head_movement_counter = 0
        hand_boxes = [None, None]
        head_box = None

        while True:
            success, image = cap.read()
            if not success:
                break

            # Converting the image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Processing the image with Mediapipe
            results = holistic.process(image)

            # Drawing hand landmarks on the image
            for i, hand_landmarks in enumerate([results.left_hand_landmarks, results.right_hand_landmarks]):
                if hand_landmarks:
                    # If hands moved, storing a green box around them and reset the counter
                    if previous_hand_landmarks[i] is not None:
                        hand_movement = np.array(
                            [[abs(prev_lm.x - curr_lm.x), abs(prev_lm.y - curr_lm.y), abs(prev_lm.z - curr_lm.z)]
                                for prev_lm, curr_lm in zip(previous_hand_landmarks[i].landmark, hand_landmarks.landmark)]
                        )
                        if np.any(hand_movement > HAND_MOVE_THRESHOLD):
                            image_hight, image_width, _ = image.shape
                            hand_coordinates = [i for i in hand_landmarks.landmark]
                            min_x = max(0, int(min([i.x for i in hand_coordinates]) * image_width))
                            min_y = max(0, int(min([i.y for i in hand_coordinates]) * image_hight))
                            max_x = min(image_width, int(max([i.x for i in hand_coordinates]) * image_width))
                            max_y = min(image_hight, int(max([i.y for i in hand_coordinates]) * image_hight))
                            hand_boxes[i] = (min_x, min_y, max_x, max_y)
                            hand_movement_counter[i] = HAND_MOVEMENT_FRAMES

                    # Drawing the green box if the counter is not zero
                    if hand_movement_counter[i] > 0:
                        cv2.rectangle(image, (hand_boxes[i][0], hand_boxes[i][1]), (hand_boxes[i][2], hand_boxes[i][3]), (0, 255, 0), 2)
                        hand_movement_counter[i] -= 1

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS
                    )

                    previous_hand_landmarks[i] = hand_landmarks

            # Drawing face landmarks on the image
            if results.face_landmarks:
                nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]

                # If head moved, storing a green box around it and reset the counter
                if previous_nose_position is not None and (abs(nose.x - previous_nose_position.x) > HEAD_MOVE_THRESHOLD or \
                                                            abs(nose.y - previous_nose_position.y) > HEAD_MOVE_THRESHOLD or \
                                                            abs(nose.z - previous_nose_position.z) > HEAD_MOVE_THRESHOLD):
                    image_hight, image_width, _ = image.shape
                    face_coordinates = [i for i in results.face_landmarks.landmark]
                    min_x = max(0, int(min([i.x for i in face_coordinates]) * image_width))
                    min_y = max(0, int(min([i.y for i in face_coordinates]) * image_hight))
                    max_x = min(image_width, int(max([i.x for i in face_coordinates]) * image_width))
                    max_y = min(image_hight, int(max([i.y for i in face_coordinates]) * image_hight))
                    head_box = (min_x, min_y, max_x, max_y)
                    head_movement_counter = HEAD_MOVEMENT_FRAMES

                # Drawing the green box if the counter is not zero
                if head_movement_counter > 0:
                    cv2.rectangle(image, (head_box[0], head_box[1]), (head_box[2], head_box[3]), (0, 255, 0), 2)
                    head_movement_counter -= 1

                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                )

                previous_nose_position = nose

            # Converting the image back to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Encoding the image as JPEG
            ret, buffer = cv2.imencode('.jpg', image)

            # Yielding the image as bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def webcam_motion_capture_view(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def home_view(request):
    return render(request, 'webcam_motion_capture.html')
