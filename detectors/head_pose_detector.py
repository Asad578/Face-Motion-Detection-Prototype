from config.settings import HEAD_YAW_THRESHOLD, HEAD_PITCH_THRESHOLD

# MediaPipe landmark indices
NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
FOREHEAD = 10
CHIN = 152

def is_head_straight(landmarks):
    if not landmarks or len(landmarks) < 455:
        return False

    nose = landmarks[NOSE_TIP]
    left_cheek = landmarks[LEFT_CHEEK]
    right_cheek = landmarks[RIGHT_CHEEK]
    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]

    # --- Horizontal rotation (Yaw) ---
    mid_cheeks_x = (left_cheek.x + right_cheek.x) / 2
    face_width = abs(right_cheek.x - left_cheek.x)
    yaw = (nose.x - mid_cheeks_x) / face_width  # normalized relative to face width

    # --- Vertical rotation (Pitch) ---
    mid_forehead_chin_y = (forehead.y + chin.y) / 2
    face_height = abs(chin.y - forehead.y)
    pitch = (nose.y - mid_forehead_chin_y) / face_height  # normalized relative to face height

    # Check thresholds
    return (
        abs(yaw) <= HEAD_YAW_THRESHOLD and
        abs(pitch) <= HEAD_PITCH_THRESHOLD
    )
