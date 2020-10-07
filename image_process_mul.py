from src.human_detection import HumanDetection
import cv2
import os
from config.config import img_folder
import numpy as np

IP = HumanDetection()


if __name__ == '__main__':
    src_folder = img_folder
    dest_folder = src_folder + "_cut"
    os.makedirs(dest_folder,exist_ok=True)
    cnt = 0
    for img_name in os.listdir(src_folder):
        cnt += 1
        print("Processing pic {}".format(cnt))
        frame = cv2.imread(os.path.join(src_folder, img_name))
        kps, boxes, _ = IP.process_img(frame, gray=False)
        img_rgb, img_black = IP.visualize()
        cv2.imshow("squat", img_rgb)
        cv2.imshow("kps", img_black)
        # cv2.imwrite(os.path.join(dest_folder, img_name), img)
        frame = img_black
        if boxes is not None:
            try:
                for idx, box in enumerate(boxes.values()):
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    x1 = 0 if x1 < 0 else x1
                    y1 = 0 if y1 < 0 else y1
                    x2 = frame.shape[1] if x2 > frame.shape[1] else x2
                    y2 = frame.shape[0] if y2 > frame.shape[0] else y2
                    img = np.asarray(frame[y1:y2, x1:x2])
                    cv2.imshow("img", img)
                    cv2.imwrite(os.path.join(dest_folder, "{}_{}_{}.jpg".format(img_name[:-4], cnt, idx)), img)
                    cv2.waitKey(1)
            except:
                pass