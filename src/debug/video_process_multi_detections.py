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
    from .tracker.track import ObjectTracker
    from .tracker.visualize import IDVisualizer
    from .utils.utils import process_kp
    from .utils.img import torch_to_im, gray3D
    from .detector.box_postprocess import crop_bbox, merge_box
    from config.config import gray_yolo_cfg, gray_yolo_weights, black_yolo_cfg, black_yolo_weights, \
        video_path, pose_cfg, pose_weight
except:
    from src.estimator.pose_estimator import PoseEstimator
    from src.estimator.visualize import KeyPointVisualizer
    from src.detector.yolo_detect import ObjectDetectionYolo
    from src.detector.visualize import BBoxVisualizer
    from src.tracker.track_match import ObjectTracker
    from src.tracker.visualize import IDVisualizer
    from src.utils.utils import process_kp
    from src.utils.img import torch_to_im, gray3D
    from src.detector.box_postprocess import crop_bbox, merge_box
    from src.debug.config.cfg_multi_detections import gray_yolo_cfg, gray_yolo_weights, black_yolo_cfg, \
        black_yolo_weights, video_path, pose_cfg, pose_weight

tensor = torch.FloatTensor


class ImgProcessor:
    def __init__(self, show_img=True):
        self.gray_detector = ObjectDetectionYolo(cfg=gray_yolo_cfg, weight=gray_yolo_weights)
        self.black_detector = ObjectDetectionYolo(cfg=black_yolo_cfg, weight=black_yolo_weights)
        self.pose_estimator = PoseEstimator(pose_cfg=pose_cfg, pose_weight=pose_weight)
        self.object_tracker = ObjectTracker()
        self.BBV = BBoxVisualizer()
        self.KPV = KeyPointVisualizer()
        self.IDV = IDVisualizer(with_bbox=False)
        self.boxes = tensor([])
        self.boxes_scores = tensor([])
        self.img_black = np.array([])
        self.frame = np.array([])
        self.id2bbox = {}
        self.kps = {}
        self.kps_score = {}
        self.show_img = show_img

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
        img_black = cv2.imread('video/black.jpg')
        if config.plot_bbox and self.boxes is not None:
            self.frame = self.BBV.visualize(self.boxes, self.frame)
            # cv2.imshow("cropped", (torch_to_im(inps[0]) * 255))
        if config.plot_kps and self.kps is not []:
            self.frame = self.KPV.vis_ske(self.frame, self.kps, self.kps_score)
            img_black = self.KPV.vis_ske_black(self.frame, self.kps, self.kps_score)
        if config.plot_id and self.id2bbox is not None:
            self.frame = self.IDV.plot_bbox_id(self.id2bbox, self.frame)
            # frame = self.IDV.plot_skeleton_id(id2ske, copy.deepcopy(img))
        return self.frame, img_black

    def process_img(self, frame, black_img):
        self.clear_res()
        self.frame = frame
        id2ske, id2kpscore = {}, {}

        with torch.no_grad():
            gray_img = gray3D(copy.deepcopy(frame))
            gray_results = self.gray_detector.process(gray_img)
            black_results = self.black_detector.process(black_img)

            gray_boxes, gray_scores = self.gray_detector.cut_box_score(gray_results)
            black_boxes, black_scores = self.black_detector.cut_box_score(black_results)

            self.boxes, self.boxes_scores = merge_box(gray_boxes, black_boxes, gray_scores, black_scores)

            if self.show_img:
                gray_img = self.BBV.visualize(gray_boxes, gray_img, gray_scores)
                cv2.imshow("gray", gray_img)
                black_img = self.BBV.visualize(black_boxes, black_img, black_scores)
                cv2.imshow("black", black_img)

            if self.boxes is not None:
                # self.id2bbox = self.boxes
                inps, pt1, pt2 = crop_bbox(frame, self.boxes)
                self.kps, self.kps_score = self.pose_estimator.process_img(inps, self.boxes, self.boxes_scores, pt1,
                                                                           pt2)

                if self.kps is not []:
                    id2ske, self.id2bbox, id2kpscore = self.object_tracker.track(self.boxes, self.kps, self.kps_score)
                else:
                    self.id2bbox = self.object_tracker.track_box(self.boxes)

        return id2ske, self.id2bbox, id2kpscore


IP = ImgProcessor()
enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
frame_size = (540, 360)


class DrownDetector:
    def __init__(self, vp):
        self.cap = cv2.VideoCapture(vp)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def process_video(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, frame_size)
            cnt += 1
            if ret:
                fgmask = self.fgbg.apply(frame)
                background = self.fgbg.getBackgroundImage()
                diff = cv2.absdiff(frame, background)
                enhanced = cv2.filter2D(diff, -1, enhance_kernel)
                kps, boxes, kps_score = IP.process_img(frame, enhanced)
                img, black_img = IP.visualize()
                cv2.imshow("res", img)
                cv2.imshow("res_black", black_img)
                cv2.waitKey(1)
            else:
                self.cap.release()
                break


if __name__ == '__main__':
    DrownDetector(video_path).process_video()

