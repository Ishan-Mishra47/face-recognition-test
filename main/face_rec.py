import cv2
from deepface import DeepFace
import os
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_check = False
reference = cv2.imread(os.path.join(BASE_DIR, "reference.jpg"))
if reference is None:
    raise FileNotFoundError("reference.jpg not found")

def check_face(frame):
    global face_check
    try:
        if DeepFace.verify(frame, reference.copy())["verified"]:
            face_check = True
        else:
            face_check = False
    except Exception:
        face_check = False