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
from ..utils.utils import process_kp
from src.detector.crop_box import crop_bbox
from ..utils.img import torch_to_im, gray3D


from .config.cfg import yolo_weight, yolo_cfg, video_path


class HumanDetection:
    def __init__(self, show_img=True):
        self.object_detector = ObjectDetectionYolo(cfg=yolo_cfg, weight=yolo_weight)
        self.object_tracker = ObjectTracker()
        self.pose_estimator = PoseEstimator(pose_cfg=config.pose_cfg, pose_weight=config.pose_weight)
        self.BBV = BBoxVisualizer()
        self.KPV = KeyPointVisualizer()
        self.IDV = IDVisualizer(with_bbox=False)
        self.img = []
        self.img_black = []
        self.show_img = show_img

    def init_sort(self):
        self.object_tracker.init_tracker()

    def process_img(self, frame, gray=False):

        img_black = cv2.imread('video/black.jpg')
        with torch.no_grad():
            if gray:
                gray_img = gray3D(copy.deepcopy(frame))
                boxes, scores = self.object_detector.process(gray_img)
                inps, pt1, pt2 = crop_bbox(frame, boxes)
            else:
                boxes, scores = self.object_detector.process(frame)
                inps, pt1, pt2 = crop_bbox(frame, boxes)

            if boxes is not None:
                key_points, kps_scores = self.pose_estimator.process_img(inps, boxes, scores, pt1, pt2)

                if config.plot_bbox:
                    frame = self.BBV.visualize(boxes, frame)
                    cv2.imshow("cropped", (torch_to_im(inps[0]) * 255))

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
                            id2ske = process_kp(id2ske[config.track_idx], config.track_idx)
                        except KeyError:
                            id2ske = {}

                    return id2ske, frame, img_black, id2bbox, id2score
                else:
                    id2bbox = self.object_tracker.track_box(boxes)
                    return {}, frame, img_black, id2bbox, {}
            else:
                return {}, frame, frame, boxes, {}


class VideoProcessor:
    def __init__(self, vp):
        self.cap = cv2.VideoCapture(vp)
        self.HD = HumanDetection()
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def process_video(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            cnt += 1
            if ret:
                # frame = cv2.resize(frame, frame_size)
                kps, img, black_img, boxes, kps_score = self.HD.process_img(frame)
                cv2.imshow("res", img)
                cv2.waitKey(2)

            else:
                self.cap.release()
                break


if __name__ == '__main__':
    VideoProcessor(video_path).process_video()
