import cv2
import mediapipe as mp
import numpy as mnp
import numpy as np


mPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mPose.Pose()


cap = cv2.VideoCapture('dance2.mp4')

drawspec1 = mpDraw.DrawingSpec(thickness=2,circle_radius=3,color=(0,0,255))
drawspec2 = mpDraw.DrawingSpec(thickness=3,circle_radius=8,color=(0,255,0))

while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img = cv2.resize(img,(500,400))
    result = pose.process(img)
    mpDraw.draw_landmarks(img,result.pose_landmarks, mPose.POSE_CONNECTIONS,drawspec1,drawspec2)


    h,w,c = img.shape
    imgblank = np.zeros([h,w,c])
    imgblank.fill(255)
    mpDraw.draw_landmarks(imgblank,result.pose_landmarks, mPose.POSE_CONNECTIONS,drawspec1,drawspec2)

    cv2.imshow('poseDetection',img)
    cv2.imshow('Blank Image',imgblank)
    cv2.waitKey(1)