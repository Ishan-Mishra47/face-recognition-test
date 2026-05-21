import threading
import cv2
from deepface import DeepFace
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_check = False
counter = 0
reference = cv2.imread(os.path.join(BASE_DIR, "reference.jpg"))
if reference is None:
    raise FileNotFoundError("reference.jpg not found — make sure it's in the same folder as face_rec.py")


def check_face(frame):
    global face_check
    try:
        if DeepFace.verify(frame, reference.copy())["verified"]:
            face_check = True
        else:
            face_check = False
    except Exception:
        face_check = False


while True:
    ret, frame = capture.read()
    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except Exception:
                pass
        counter += 1

        if face_check:
            cv2.putText(frame, "MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()