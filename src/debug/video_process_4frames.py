import torch
import cv2
from config import config
import numpy as np
from src.estimator.pose_estimator import PoseEstimator
from src.estimator.visualize import KeyPointVisualizer
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.visualize import BBoxVisualizer
from src.tracker.track import ObjectTracker
from src.tracker.visualize import IDVisualizer
from src.utils.img import calibration
from src.detector.box_postprocess import crop_bbox, eliminate_nan

try:
    from config.config import yolo_weight, yolo_cfg, video_1, video_2, video_3, video_4, pose_cfg, pose_weight
except:

    from src.debug.config.cfg_4frame import yolo_weight, yolo_cfg, video_1, video_2, video_3, video_4, pose_cfg, pose_weight

tensor = torch.FloatTensor


class ImgProcessor:
    def __init__(self, show_img=True):
        self.object_detector = ObjectDetectionYolo(cfg=yolo_cfg, weight=yolo_weight)
        self.pose_estimator = PoseEstimator(pose_cfg=pose_cfg, pose_weight=pose_weight)
        self.object_tracker1 = ObjectTracker()
        self.object_tracker2 = ObjectTracker()
        self.object_tracker3 = ObjectTracker()
        self.object_tracker4 = ObjectTracker()
        self.BBV = BBoxVisualizer()
        self.KPV = KeyPointVisualizer()
        self.IDV = IDVisualizer()
        self.boxes = tensor([])
        self.boxes_scores = tensor([])
        self.frame = np.array([])
        self.id2bbox = {}
        self.kps = {}
        self.kps_score = {}
        self.show_img = show_img

    def init_sort(self):
        self.object_tracker1.init_tracker()
        self.object_tracker2.init_tracker()
        self.object_tracker3.init_tracker()
        self.object_tracker4.init_tracker()

    def clear_res(self):
        self.boxes = tensor([])
        self.boxes_scores = tensor([])
        self.frame = np.array([])
        self.id2bbox = {}
        self.kps = {}
        self.kps_score = {}

    def visualize(self):
        img_black = cv2.imread('video/black.jpg')
        if config.plot_bbox and self.boxes is not None:
            self.BBV.visualize(self.boxes, self.frame, self.boxes_scores)
            # cv2.imshow("cropped", (torch_to_im(inps[0]) * 255))
        if config.plot_kps and self.kps is not []:
            self.KPV.vis_ske(self.frame, self.kps, self.kps_score)
            self.KPV.vis_ske_black(img_black, self.kps, self.kps_score)
        if config.plot_id and self.id2bbox is not None:
            self.IDV.plot_bbox_id(self.id2bbox, self.frame)
            # frame = self.IDV.plot_skeleton_id(id2ske, copy.deepcopy(img))
        return self.frame, img_black

    def __process_single_img(self, fr, tracker):
        self.clear_res()
        self.frame = fr
        with torch.no_grad():

            box_res = self.object_detector.process(fr)
            self.boxes, self.boxes_scores = self.object_detector.cut_box_score(box_res)

            if box_res is not None:
                self.id2bbox = tracker.track(box_res)
                self.id2bbox = eliminate_nan(self.id2bbox)
                boxes = tracker.id_and_box(self.id2bbox)

                inps, pt1, pt2 = crop_bbox(fr, boxes)
                if inps is not None:
                    kps, kps_score, kps_id = self.pose_estimator.process_img(inps, boxes, pt1, pt2)
                    self.kps, self.kps_score = tracker.match_kps(kps_id, kps, kps_score)

        img, black_img = self.visualize()
        return img, black_img, self.kps, self.id2bbox, self.kps_score

    def process_img(self, fr1, fr2, fr3, fr4):
        fr1, fr2, fr3, fr4 = calibration(fr1), calibration(fr2), calibration(fr3), calibration(fr4)
        res1 = self.__process_single_img(fr1, self.object_tracker1)
        res2 = self.__process_single_img(fr2, self.object_tracker2)
        res3 = self.__process_single_img(fr3, self.object_tracker3)
        res4 = self.__process_single_img(fr4, self.object_tracker4)

        return res1, res2, res3, res4


IP = ImgProcessor()
frame_size = (720, 540)


class VideoProcessor:
    def __init__(self, vp1, vp2, vp3, vp4, show_img=True):
        self.cap1 = cv2.VideoCapture(vp1)
        self.cap2 = cv2.VideoCapture(vp2)
        self.cap3 = cv2.VideoCapture(vp3)
        self.cap4 = cv2.VideoCapture(vp4)
        self.show_img = show_img

    def process_frame(self, f1, f2, f3, f4):
        fr1, fr2, fr3, fr4 = cv2.resize(f1, frame_size), cv2.resize(f2, frame_size), cv2.resize(f3, frame_size), \
                             cv2.resize(f4, frame_size),

        res1, res2, res3, res4 = IP.process_img(fr1, fr2, fr3, fr4)
        img1, img2, img3, img4 = res1[0], res2[0], res3[0], res4[0]

        img_ver1 = np.concatenate((img1, img2), axis=0)
        img_ver2 = np.concatenate((img3, img4), axis=0)
        img = np.concatenate((img_ver1, img_ver2), axis=1)
        return img

    def process_video(self):
        cnt = 0
        while True:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            ret3, frame3 = self.cap3.read()
            ret4, frame4 = self.cap4.read()
            cnt += 1

            if ret1:
                img = self.process_frame(frame1, frame2, frame3, frame4)
                img = cv2.resize(img, frame_size)

                if self.show_img:
                    cv2.imshow("res", img)
                    cv2.waitKey(2)

            else:
                self.cap1.release()
                self.cap2.release()
                self.cap3.release()
                self.cap4.release()
                break


if __name__ == '__main__':
    VideoProcessor(video_1, video_2, video_3, video_4).process_video()
