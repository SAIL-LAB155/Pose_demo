import torch
try:
    import src.debug.config.cfg_with_CNN as config
except:
    import config.config as config
import cv2
import copy
import numpy as np
from src.estimator.pose_estimator import PoseEstimator
from src.estimator.visualize import KeyPointVisualizer
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.visualize import BBoxVisualizer
from src.tracker.track import ObjectTracker
from src.tracker.visualize import IDVisualizer
from src.utils.img import gray3D
from src.detector.box_postprocess import crop_bbox, eliminate_nan
from src.CNNclassifier.inference import CNNInference

tensor = torch.FloatTensor


class HumanDetection:
    def __init__(self, resize_size, show_img=True):
        self.object_detector = ObjectDetectionYolo(cfg=config.yolo_cfg, weight=config.yolo_weight)
        self.object_tracker = ObjectTracker()
        self.pose_estimator = PoseEstimator(pose_cfg=config.pose_cfg, pose_weight=config.pose_weight)
        self.BBV = BBoxVisualizer()
        self.KPV = KeyPointVisualizer()
        self.IDV = IDVisualizer()
        self.boxes = tensor([])
        self.boxes_scores = tensor([])
        self.frame = np.array([])
        self.id2bbox = {}
        self.CNN_model = CNNInference()
        self.kps = {}
        self.kps_score = {}
        self.show_img = show_img
        self.resize_size = resize_size

    def init_sort(self):
        self.object_tracker.init_tracker()

    def clear_res(self):
        self.boxes = tensor([])
        self.boxes_scores = tensor([])
        self.frame = np.array([])
        self.id2bbox = {}
        self.kps = {}
        self.kps_score = {}

    def visualize(self):
        img_black = np.full((self.resize_size[1], self.resize_size[0], 3), 0).astype(np.uint8)
        if config.plot_bbox and self.boxes is not None:
            self.BBV.visualize(self.boxes, self.frame)
        if config.plot_kps and self.kps is not []:
            self.KPV.vis_ske(self.frame, self.kps, self.kps_score)
            self.KPV.vis_ske_black(img_black, self.kps, self.kps_score)
        if config.plot_id and self.id2bbox is not None:
            self.IDV.plot_bbox_id(self.id2bbox, self.frame)
            self.IDV.plot_skeleton_id(self.kps, self.frame)
        return self.frame, img_black

    def process_img(self, frame, gray=False):
        self.clear_res()
        self.frame = frame

        with torch.no_grad():
            if gray:
                gray_img = gray3D(copy.deepcopy(frame))
                box_res = self.object_detector.process(gray_img)
            else:
                box_res = self.object_detector.process(frame)
            self.boxes, self.boxes_scores = self.object_detector.cut_box_score(box_res)

            if box_res is not None:
                self.id2bbox = self.object_tracker.track(box_res)
                self.id2bbox = eliminate_nan(self.id2bbox)
                boxes = self.object_tracker.id_and_box(self.id2bbox)

                inps, pt1, pt2 = crop_bbox(frame, boxes)
                if inps is not None:
                    kps, kps_score, kps_id = self.pose_estimator.process_img(inps, boxes, pt1, pt2)
                    self.kps, self.kps_score = self.object_tracker.match_kps(kps_id, kps, kps_score)

        return self.kps, self.id2bbox, self.kps_score

    def classify_whole(self, pred_img, show_img):
        pred = self.CNN_model.classify_whole(pred_img, show_img)
        return pred

    def classify(self, pred_img, show_img, id2bbox):
        preds = self.CNN_model.classify(pred_img, id2bbox)
        self.CNN_model.visualize(show_img, preds)
        return preds


resize_ratio = config.resize_ratio
show_size = config.show_size
classify_type = config.classify_type
gray = config.gray


class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.resize_size = (int(self.width * resize_ratio), int(self.height * resize_ratio))
        self.IP = HumanDetection(self.resize_size)

    def process_video(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            cnt += 1
            if ret:
                frame = cv2.resize(frame, self.resize_size)
                frame2 = copy.deepcopy(frame)
                kps, boxes, kps_score = self.IP.process_img(frame, gray=gray)
                img, img_black = self.IP.visualize()
                if classify_type == 1:
                    result = self.IP.classify_whole(img_black, img)
                elif classify_type == 2:
                    result = self.IP.classify_whole(frame2, img)
                elif classify_type == 3:
                    result = self.IP.classify(img_black, img, boxes)
                elif classify_type == 4:
                    result = self.IP.classify(frame2, img, boxes)
                else:
                    raise ValueError("Not a right classification type!")

                cv2.imshow("res", cv2.resize(img, show_size))
                cv2.waitKey(2)

            else:
                self.cap.release()
                break


if __name__ == '__main__':
    VideoProcessor(config.video_path).process_video()
