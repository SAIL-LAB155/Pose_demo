from ..estimator.pose_estimator import PoseEstimator
from ..estimator.visualize import KeyPointVisualizer
from ..detector.yolo_detect import ObjectDetectionYolo
from ..detector.visualize import BBoxVisualizer
from ..tracker.track import ObjectTracker
from ..tracker.visualize import IDVisualizer
import torch
import cv2
import copy
from config import config
from ..utils.img import gray3D
from ..utils.utils import process_kp
from ..detector.box_postprocess import crop_bbox, merge_box


from .config.cfg_multi_detections import gray_yolo_cfg, gray_yolo_weights, black_yolo_cfg, black_yolo_weights


class ImgProcessor:
    def __init__(self, show_img=True):
        self.gray_detector = ObjectDetectionYolo(cfg=gray_yolo_cfg, weight=gray_yolo_weights)
        self.black_detector = ObjectDetectionYolo(cfg=black_yolo_cfg, weight=black_yolo_weights)
        self.pose_estimator = PoseEstimator(pose_cfg=config.pose_cfg, pose_weight=config.pose_weight)
        self.object_tracker = ObjectTracker()
        self.BBV = BBoxVisualizer()
        self.KPV = KeyPointVisualizer()
        self.IDV = IDVisualizer(with_bbox=False)
        self.img = []
        self.img_black = []
        self.show_img = show_img

    def init_sort(self):
        self.object_tracker.init_tracker()

    def process_img(self, frame, black_img):

        img_black = cv2.imread('Video/black.jpg')
        img_black = cv2.resize(img_black, config.frame_size)
        with torch.no_grad():
            gray_img = gray3D(copy.deepcopy(frame))
            orig_img, gray_boxes, gray_scores = self.gray_detector.process(gray_img)
            orig_img, black_boxes, black_scores = self.black_detector.process(black_img)

            boxes, scores = merge_box(gray_boxes, black_boxes, gray_scores, black_scores)
            inps, pt1, pt2 = crop_bbox(frame, boxes)

            if boxes is not None:
                key_points, kps_scores = self.pose_estimator.process_img(inps, boxes, scores, pt1, pt2)

                if config.plot_bbox:
                    frame = self.BBV.visualize(boxes, frame)
                    if gray_boxes is not None:
                        gray_img = self.BBV.visualize(gray_boxes, gray_img)
                    cv2.imshow("gray", gray_img)

                    if black_boxes is not None:
                        black_img = self.BBV.visualize(black_boxes, black_img)
                    cv2.imshow("black", black_img)

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
                            kps = process_kp(id2ske[config.track_idx], config.track_idx)
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
                return {}, frame, img_black, boxes, {}
