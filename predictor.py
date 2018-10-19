import dlib

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def predict(frame, rect):
    return predictor(frame, rect)
