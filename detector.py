import dlib

detector = dlib.get_frontal_face_detector()

def detect(frame, n):
    return detector(frame, n)
