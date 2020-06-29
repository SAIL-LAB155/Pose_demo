import torch
import cv2
import copy
import numpy as np
from config import config

try:
    from .estimator.pose_estimator import PoseEstimator
    from .estimator.visualize import KeyPointVisualizer
    from .detector.yolo_detect import ObjectDetectionYolo
    from .detector.visualize import BBoxVisualizer
    from .tracker.track_old import ObjectTracker
    from .tracker.visualize import IDVisualizer
    from .utils.utils import process_kp
    from .utils.img import torch_to_im, gray3D
    from .detector.box_postprocess import crop_bbox
    from config.config import yolo_weight, yolo_cfg, video_path, pose_weight, pose_cfg
except:
    from src.estimator.pose_estimator import PoseEstimator
    from src.estimator.visualize import KeyPointVisualizer
    from src.detector.yolo_detect import ObjectDetectionYolo
    from src.detector.visualize import BBoxVisualizer
    from src.tracker.track_old import ObjectTracker
    from src.tracker.visualize import IDVisualizer
    from src.utils.utils import process_kp
    from src.utils.img import torch_to_im, gray3D
    from src.detector.box_postprocess import crop_bbox
    from src.debug.config.cfg import yolo_weight, yolo_cfg, video_path, pose_weight, pose_cfg

tensor = torch.FloatTensor


class HumanDetection:
    def __init__(self, show_img=True):
        self.object_detector = ObjectDetectionYolo(cfg=yolo_cfg, weight=yolo_weight)
        self.object_tracker = ObjectTracker()
        self.pose_estimator = PoseEstimator(pose_cfg=pose_cfg, pose_weight=pose_weight)
        self.BBV = BBoxVisualizer()
        self.KPV = KeyPointVisualizer()
        self.IDV = IDVisualizer(with_bbox=False)
        self.boxes = tensor([])
        self.boxes_scores = tensor([])
        self.kps = tensor([])
        self.kps_score = tensor([])
        self.img_black = np.array([])
        self.frame = np.array([])
        self.id2ske = {}
        self.id2bbox = {}
        self.id2score = {}
        self.show_img = show_img

    def init_sort(self):
        self.object_tracker.init_tracker()

    def clear_res(self):
        self.boxes = tensor([])
        self.boxes_scores = tensor([])
        self.kps = tensor([])
        self.kps_score = tensor([])
        self.frame = np.array([])
        self.id2ske = {}
        self.id2bbox = {}
        self.id2score = {}

    def visualize(self):
        img_black = cv2.imread('video/black.jpg')
        if config.plot_bbox and self.boxes is not None:
            self.frame = self.BBV.visualize(self.boxes, self.frame)
            # cv2.imshow("cropped", (torch_to_im(inps[0]) * 255))
        if config.plot_kps and self.kps is not []:
            self.frame = self.KPV.vis_ske(self.frame, self.kps, self.kps_score)
            img_black = self.KPV.vis_ske_black(self.frame, self.kps, self.kps_score)
        if config.plot_id:
            self.frame = self.IDV.plot_bbox_id(self.id2bbox, self.frame)
            # frame = self.IDV.plot_skeleton_id(id2ske, copy.deepcopy(img))
        return self.frame, img_black

    def process_img(self, frame, gray=False):
        self.clear_res()
        self.frame = frame

        with torch.no_grad():
            if gray:
                gray_img = gray3D(copy.deepcopy(frame))
                self.boxes, self.boxes_scores = self.object_detector.process(gray_img)
                inps, pt1, pt2 = crop_bbox(frame, self.boxes)
            else:
                self.boxes, self.boxes_scores = self.object_detector.process(frame)
                inps, pt1, pt2 = crop_bbox(frame, self.boxes)

            if self.boxes is not None:
                self.kps, self.kps_score = self.pose_estimator.process_img(inps, self.boxes, self.boxes_scores, pt1, pt2)

                if self.kps is not []:
                    self.id2ske, self.id2bbox, self.id2score = self.object_tracker.track(self.boxes,
                                                                                         self.kps, self.kps_score)

                    if config.track_idx != "all":
                        try:
                            self.id2ske = process_kp(self.id2ske[config.track_idx], config.track_idx)
                        except KeyError:
                            self.id2ske = {}

        return self.id2ske, self.id2bbox, self.id2score


IP = HumanDetection()
frame_size = (720, 540)


class VideoProcessor:
    def __init__(self, vp):
        self.cap = cv2.VideoCapture(vp)

    def process_video(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            cnt += 1
            if ret:
                frame = cv2.resize(frame, frame_size)
                kps, boxes, kps_score = IP.process_img(frame)
                img, img_black = IP.visualize()
                cv2.imshow("res", img)
                cv2.waitKey(2)

            else:
                self.cap.release()
                break


if __name__ == '__main__':
    VideoProcessor(video_path).process_video()
