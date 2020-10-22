#-*- coding: utf-8 -*-

from src.human_detection import HumanDetection as ImgProcessor
import cv2
from config.config import video_path, write_box, write_video, frame_size, write_kps
from utils.utils import boxdict2str, kpsdict2str, kpsScoredict2str

body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
body_dict = {name: idx for idx, name in enumerate(body_parts)}

IP = ImgProcessor()
fourcc = cv2.VideoWriter_fourcc(*'XVID')


class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if write_video:
            self.out = cv2.VideoWriter(video_path[:-4] + "_processed.avi", fourcc, 12, (frame_size[0]*2, frame_size[1]))
        if write_box:
            box_file = "/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] + "_box.txt"
            self.box_txt = open(box_file, "w")
        if write_kps:
            kps_file = "/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] + "_kps.txt"
            kps_score_file = "/".join(video_path.split("/")[:-1]) + "/" + video_path.split("/")[-1][:-4] + "_kps_score.txt"
            self.kps_txt = open(kps_file, "w")
            self.kps_score_txt = open(kps_score_file, "w")

    def process_video(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            cnt += 1
            if ret:
                # frame = cv2.resize(frame, frame_size)
                kps, boxes, kps_score = IP.process_img(frame)
                img, black_img = IP.visualize()
                if boxes is not None:
                    if write_box:
                        box_str = ""
                        for k, v in boxes.items():
                            box_str += boxdict2str(k, v)
                        self.box_txt.write(box_str)
                        self.box_txt.write("\n")
                else:
                    if write_box:
                        self.box_txt.write("\n")

                if kps:
                    # cv2.putText(img, "cnt{}".format(cnt), (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 5)
                    if write_kps:
                        kps_str = ""
                        for k, v in kps.items():
                            kps_str += kpsdict2str(k, v)
                        self.kps_txt.write(kps_str)
                        self.kps_txt.write("\n")

                        kps_score_str = ""
                        for k, v in kps_score.items():
                            kps_score_str += kpsScoredict2str(k, v)
                        self.kps_score_txt.write(kps_score_str)
                        self.kps_score_txt.write("\n")

                else:
                    if write_kps:
                        self.kps_txt.write("\n")
                        self.kps_score_txt.write("\n")
                    img = frame
                    # cv2.putText(img, "cnt{}".format(cnt), (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 5)

                cv2.imshow("res", cv2.resize(img, (1080, 720)))
                cv2.waitKey(2)
                if write_video:
                    self.out.write(img)
            else:
                self.cap.release()
                if write_video:
                    self.out.release()
                break


if __name__ == '__main__':
    # src_folder = "video/push_up"
    # dest_fodler = src_folder + "kps"
    # sub_folder = [os.path.join(src_folder, folder) for folder in os.listdir(src_folder)]
    # sub_dest_folder = [os.path.join(dest_fodler, folder) for folder in os.listdir(src_folder)]

    VideoProcessor(video_path).process_video()
