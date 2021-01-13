from src.human_detection import HumanDetection
import cv2
import os
from config.config import img_folder

IP = HumanDetection()


if __name__ == '__main__':
    src_folder = img_folder
    dest_folder = src_folder + "_kps"
    os.makedirs(dest_folder, exist_ok=True)
    cnt = 0
    for img_name in os.listdir(src_folder):
        cnt += 1
        print("Processing pic {}".format(cnt))
        frame = cv2.imread(os.path.join(src_folder, img_name))
        kps, _, _ = IP.process_img(frame)
        img, black_img = IP.visualize()
        IP.init()
        cv2.imshow("yoga", cv2.resize(img, (416,416)))
        cv2.waitKey(0)
        cv2.imwrite(os.path.join(dest_folder, img_name), img)
