import cv2
import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core import base_options

# Face mesh connections (from MediaPipe)
FACE_MESH_CONNECTIONS = [
    # Lips
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 10), (10, 151), (151, 406), (406, 320), (320, 307), (307, 375),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 307),
    
    # Right eyebrow
    (336, 296), (296, 334), (334, 293), (293, 300), (300, 293),
    
    # Left eyebrow
    (66, 63), (63, 105), (105, 66), (66, 107), (107, 55),
    
    # Right eye
    (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390), (390, 249), (249, 390), (390, 373),
    
    # Left eye
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), (33, 246), (246, 161),
    
    # Face outline
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
    (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]

def draw_text(frame, text, y):
    cv2.putText(frame, text, (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 255), 2)

def draw_face_mesh(frame, face_landmarks):
    """
    Draw face mesh landmarks with white color and 70% transparency.
    Uses thin lines.
    
    Args:
        frame: The input frame to draw on
        face_landmarks: MediaPipe face landmarks list
    """
    if not face_landmarks or len(face_landmarks) == 0:
        return
    
    # Create overlay for transparency
    overlay = frame.copy()
    
    # White color in BGR
    white = (255, 255, 255)
    
    # Extract landmarks from the first detected face
    # face_landmarks[0] is already the list of landmark objects
    landmarks_list = face_landmarks[0]
    h, w = frame.shape[:2]
    
    # Draw face mesh connections with thin lines
    for connection in FACE_MESH_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks_list) and end_idx < len(landmarks_list):
            start = landmarks_list[start_idx]
            end = landmarks_list[end_idx]
            
            # Convert normalized coordinates to pixel coordinates
            start_pos = (int(start.x * w), int(start.y * h))
            end_pos = (int(end.x * w), int(end.y * h))
            
            # Draw line on overlay (thin line with thickness=1)
            cv2.line(overlay, start_pos, end_pos, white, 1)
    
    # Draw individual landmarks as small circles (for better visibility)
    for landmark in landmarks_list:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(overlay, (x, y), 1, white, 1)
    
    # Blend overlay with frame (70% transparency = 30% opacity of overlay)
    cv2.addWeighted(overlay, 0.4, frame, 0.8, 0, frame)
