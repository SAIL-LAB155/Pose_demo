from src.human_track_detection import ImgProcessor
import cv2
import os
from config.config import img_folder

IP = ImgProcessor()


if __name__ == '__main__':
    src_folder = img_folder
    dest_folder = src_folder + "_kps"
    os.makedirs(dest_folder,exist_ok=True)
    cnt = 0
    for img_name in os.listdir(src_folder):
        cnt += 1
        print("Processing pic {}".format(cnt))
        frame = cv2.imread(os.path.join(src_folder, img_name))
        kps, img, black_img, _, _ = IP.process_img(frame)
        cv2.imwrite(os.path.join(dest_folder, img_name), img)
