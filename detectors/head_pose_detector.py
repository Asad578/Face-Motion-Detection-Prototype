from config.settings import HEAD_CENTER_DEVIATION_THRESHOLD

def is_head_straight(face, frame_shape):
    """
    Strict horizontal head movement detection.
    Works symmetrically for left and right.
    """
    x, y, w, h = face
    frame_width = frame_shape[1]

    face_left = x
    face_right = x + w
    frame_center = frame_width / 2

    left_offset = abs(face_left - frame_center) / frame_width
    right_offset = abs(face_right - frame_center) / frame_width

    deviation = min(left_offset, right_offset)

    return deviation <= HEAD_CENTER_DEVIATION_THRESHOLD
