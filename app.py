from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)

# Initialize Mediapipe solutions
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(model_complexity=1)  # Reduce complexity
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)  # Limit to 1 face
hands = mp_hands.Hands(max_num_hands=2)  # Limit to 2 hands

def draw_full_body(image, pose_landmarks, face_landmarks, hand_landmarks):
    h, w, _ = image.shape
    output_image = np.zeros_like(image)  # Create a black background

    def get_coords(landmark):
        return int(landmark.x * w), int(landmark.y * h)

    if pose_landmarks:
        left_shoulder = get_coords(pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
        right_shoulder = get_coords(pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
        left_hip = get_coords(pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP])
        right_hip = get_coords(pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP])

        torso_points = np.array([left_shoulder, right_shoulder, right_hip, left_hip], np.int32)
        cv2.fillPoly(output_image, [torso_points], (255, 255, 255))

        neck_top = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
        neck_bottom = ((left_shoulder[0] + right_shoulder[0]) // 2, neck_top[1] + 20)
        cv2.line(output_image, neck_top, neck_bottom, (255, 255, 255), 8)

        left_elbow = get_coords(pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])
        right_elbow = get_coords(pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])
        left_wrist = get_coords(pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
        right_wrist = get_coords(pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
        left_knee = get_coords(pose_landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
        right_knee = get_coords(pose_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])
        left_ankle = get_coords(pose_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
        right_ankle = get_coords(pose_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])

        cv2.line(output_image, left_shoulder, left_elbow, (255, 255, 255), 12)
        cv2.line(output_image, left_elbow, left_wrist, (255, 255, 255), 12)
        cv2.line(output_image, right_shoulder, right_elbow, (255, 255, 255), 12)
        cv2.line(output_image, right_elbow, right_wrist, (255, 255, 255), 12)

        cv2.line(output_image, left_hip, left_knee, (255, 255, 255), 12)
        cv2.line(output_image, left_knee, left_ankle, (255, 255, 255), 12)
        cv2.line(output_image, right_hip, right_knee, (255, 255, 255), 12)
        cv2.line(output_image, right_knee, right_ankle, (255, 255, 255), 12)

    if face_landmarks:
        for face in face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                output_image, face, mp_face_mesh.FACEMESH_TESSELATION,
                mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )

    if hand_landmarks:
        for hand in hand_landmarks:
            for point in hand.landmark:
                x, y = get_coords(point)
                cv2.circle(output_image, (x, y), 6, (255, 255, 255), -1)
            mp.solutions.drawing_utils.draw_landmarks(
                output_image, hand, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

    return output_image

def generate_frames():
    cap = cv2.VideoCapture(0)
    fps_limit = 10  # Process 10 frames per second
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - prev_time > 1 / fps_limit:
            prev_time = current_time

            frame = cv2.resize(frame, (640, 480))  # Reduce frame resolution
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(rgb_frame)
            face_results = face_mesh.process(rgb_frame)
            hand_results = hands.process(rgb_frame)

            output_frame = draw_full_body(
                frame,
                pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None,
                face_results.multi_face_landmarks if face_results.multi_face_landmarks else None,
                hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else None
            )

            _, buffer = cv2.imencode('.jpg', output_frame)
            output_frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
