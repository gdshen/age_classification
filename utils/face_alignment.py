from imutils.face_utils import FACIAL_LANDMARKS_IDXS, shape_to_np, rect_to_bb
import numpy as np
import cv2
import argparse
import dlib
import imutils


# according to link https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35), desiredFaceWdith=256, desiredFaceHeight=None):
        """
        store the facial landmark predictor, desired output left eye position, and desired output face width + height
        :param predictor:
        :param desiredLeftEye:
        :param desiredFaceWdith:
        :param desiredFaceHeight:
        """
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWdith
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        """
        alignment
        :param image:
        :param gray:
        :param rect:
        :return:
        """
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)

        # extract the left and the right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS['right_eye']
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mas for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype(int)
        rightEyeCenter = rightEyePts.mean(axis=0).astype(int)

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye

        # determine the scale of the new resulting image by taking the radio of the distance between eyes
        # in the *current* image to the radio of distance between eyes in the *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates(i.e., the median point) between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2, (leftEyeCenter[1] + rightEyeCenter[1] // 2))

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
                    help="path to facial landmark predictor")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    args = vars(ap.parse_args())

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(args["image"])
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale
    # image
    cv2.imshow("Input", image)
    rects = detector(gray, 2)

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)

        # display the output images
        cv2.imshow("Original", faceOrig)
        cv2.imshow("Aligned", faceAligned)
        cv2.waitKey(0)

