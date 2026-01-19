from detectors.head_pose_detector import is_head_straight

def is_face_aligned(face_landmarks):
    """
    Returns True if the face is properly aligned (frontal, not tilted)
    for proctoring purposes.
    """
    if not face_landmarks:
        return False

    # Use head_pose_detector
    return is_head_straight(face_landmarks)
