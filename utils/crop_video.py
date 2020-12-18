import cv2
from src.utils.utils import cut_image_box

cap = cv2.VideoCapture("walk/mbg_2_3d.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("walk/mbg_3dpic.mp4", fourcc, 12, (1280, 720))
left, right, top, bottom = 580, 1180, 40, 540

while True:
    ret, frame = cap.read()
    if ret:
        img = cut_image_box(cv2.resize(frame, (1200, 600)), (left, top, right, bottom))
        cv2.imshow("img", cv2.resize(img, (720, 540)))
        cv2.waitKey(1)
        out.write(cv2.resize(img, (1280, 720)))
    else:
        out.release()
        break



