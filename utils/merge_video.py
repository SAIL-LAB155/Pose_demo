import cv2
import numpy as np

cap1 = cv2.VideoCapture("walk/mbg2.mp4")
cap2 = cv2.VideoCapture("walk/mbg_3dpic.mp4")
cap3 = cv2.VideoCapture("walk/mbg2_processed.avi")
cap4 = cv2.VideoCapture("walk/face_seg4.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("walk/result.avi", fourcc, 28, (1280, 720))


while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    if ret1:
        f1 = cv2.resize(frame1, (640, 360))
        f2 = cv2.resize(frame2, (640, 360))
        f3 = cv2.resize(frame3, (640, 360))
        f4 = cv2.resize(frame4, (640, 360))

        r1 = np.concatenate((f1, f2), axis=0)
        r2 = np.concatenate((f3, f4), axis=0)
        r3 = np.concatenate((r1, r2), axis=1)

        cv2.imshow("img", cv2.resize(r3, (720, 540)))
        cv2.waitKey(1)
        out.write(r3)
    else:
        break

