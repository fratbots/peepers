import cv2
import time
import imutils
import dlib
from imutils import face_utils
from eyes import eye_aspect_ratio

class Game:
    """High-level game controller."""

    def __init__(self, eye_ar_thresh, eye_ar_consec_frames):
        self.EYE_AR_THRESH = eye_ar_thresh
        self.EYE_AR_CONSEC_FRAMES = eye_ar_consec_frames
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


    def start(self):
        # frame counter
        FRAMES_COUNT = 0

        # total number of blinks
        BLINKS_COUNT = 0

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 120)

        # grab the indexes of the facial landmarks for the left and right eye
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        propFps = cap.get(cv2.CAP_PROP_FPS)
        print("CAP_PROP_FPS: {0}".format(propFps))
        realFps = 0
        frameNum = 0
        timeStart = time.time()
             
        while(True):
            # capture frame-by-frame
            ret, frame = cap.read()

            frame = imutils.resize(frame, width=600)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = self.detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy array
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < self.EYE_AR_THRESH:
                    self.FRAMES_COUNT += 1

                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if self.FRAMES_COUNT >= self.EYE_AR_CONSEC_FRAMES:
                        self.BLINKS_COUNT += 1

                    # reset the eye frame counter
                    self.FRAMES_COUNT = 0

                # draw the total number of blinks on the frame along with
                # the computed eye aspect ratio for the frame
                cv2.putText(frame, "Blinks: {}".format(self.BLINKS_COUNT), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frameNum = frameNum + 1
            timeNow = time.time()
            realFps = frameNum / (timeNow - timeStart)

            if frameNum % 30 == 0:
               print("FPS: {0}".format(realFps))

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
