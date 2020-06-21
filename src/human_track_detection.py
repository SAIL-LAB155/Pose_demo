from src.estimator.pose_estimator import PoseEstimator
from src.estimator.visualize import KeyPointVisualizer
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.visualize import BBoxVisualizer
from src.tracker.track import ObjectTracker
from src.tracker.visualize import IDVisualizer
import torch
import cv2
import copy
from config import config
from utils.utils import gray3D
from src.detector.crop_box import crop_bbox


class ImgProcessor:
    def __init__(self, show_img=True):
        self.pose_estimator = PoseEstimator()
        self.object_detector = ObjectDetectionYolo()
        self.object_tracker = ObjectTracker()
        self.BBV = BBoxVisualizer()
        self.KPV = KeyPointVisualizer()
        self.IDV = IDVisualizer(with_bbox=False)
        self.img = []
        self.img_black = []
        self.show_img = show_img

    def init_sort(self):
        self.object_tracker.init_tracker()

    def __process_kp(self, kps, idx):
        new_kp = []
        for bdp in range(len(kps)):
            for coord in range(2):
                new_kp.append(kps[bdp][coord])
        return {idx: new_kp}

    def process_img(self, frame, gray=False):

        img_black = cv2.imread('video/black.jpg')
        with torch.no_grad():
            if gray:
                gray_img = gray3D(copy.deepcopy(frame))
                boxes, scores = self.object_detector.process(gray_img)
                inps, pt1, pt2 = crop_bbox(frame, boxes, scores)
            else:
                boxes, scores = self.object_detector.process(frame)
                inps, pt1, pt2 = crop_bbox(frame, boxes, scores)

            if boxes is not None:
                key_points, kps_scores = self.pose_estimator.process_img(inps, frame, boxes, scores, pt1, pt2)

                if config.plot_bbox:
                    frame = self.BBV.visualize(boxes, frame)
                    cv2.imshow("cropped", inps[0])

                if key_points is not []:
                    id2ske, id2bbox, id2score = self.object_tracker.track(boxes, key_points, kps_scores)

                    if config.plot_kps:
                        if key_points is not []:
                            frame = self.KPV.vis_ske(frame, key_points, kps_scores)
                            img_black = self.KPV.vis_ske_black(frame, key_points, kps_scores)

                    if config.plot_id:
                        frame = self.IDV.plot_bbox_id(id2bbox, copy.deepcopy(frame))
                        # frame = self.IDV.plot_skeleton_id(id2ske, copy.deepcopy(img))

                    if config.track_idx != "all":
                        try:
                            kps = self.__process_kp(id2ske[config.track_idx], config.track_idx)
                        except KeyError:
                            kps = {}
                    else:
                        kps = id2ske
                        kp_scores = id2score

                    return kps, frame, img_black, id2bbox, kp_scores
                else:
                    id2bbox = self.object_tracker.track_box(boxes)
                    return {}, frame, img_black, id2bbox, {}
            else:
                return {}, frame, frame, boxes, {}

