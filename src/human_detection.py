from src.estimator.pose_estimator import PoseEstimator
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.visualize import BBoxVisualizer
import torch
import cv2


class ImgProcessor:
    def __init__(self, show_img=True):
        self.pose_estimator = PoseEstimator()
        self.object_detector = ObjectDetectionYolo()
        self.img = []
        self.img_black = []
        self.show_img = show_img
        self.BBV = BBoxVisualizer()

    def process_img(self, frame):
        with torch.no_grad():
            inps, orig_img, boxes, scores, pt1, pt2 = self.object_detector.process(frame)
            if boxes is not None:
                key_points, img, img_black = self.pose_estimator.process_img(inps, orig_img, boxes, scores, pt1, pt2)

                img = self.BBV.visualize(boxes, img)
                return key_points, img, img_black
            else:
                return [], frame, cv2.imread("video/black.jpg")


class ImgProcessorForbbox:
    def __init__(self):
        self.object_detector = ObjectDetectionYolo()
        self.img = []
        self.img_black = []
        self.BBV = BBoxVisualizer()

    def process_img(self, frame, enhanced=None):
        with torch.no_grad():
            inps, orig_img, boxes, scores, pt1, pt2 = self.object_detector.process(frame)
            try:
                img = self.BBV.visualize(boxes, frame)
                return img
            except:
                return frame



