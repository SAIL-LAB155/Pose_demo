import torch
import cv2
import copy
import numpy as np
from src.estimator.pose_estimator import PoseEstimator
from src.estimator.visualize import KeyPointVisualizer
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.visualize import BBoxVisualizer
from src.tracker.track import ObjectTracker
from src.tracker.visualize import IDVisualizer
from src.utils.img import torch_to_im, gray3D, cut_image_with_box
from src.detector.box_postprocess import crop_bbox
from src.CNNclassifier.inference import CNNInference

try:
    from src.debug.config.cfg_with_CNN import yolo_weight, yolo_cfg, video_path, pose_weight, pose_cfg, CNN_class
except:
    from config.config import yolo_weight, yolo_cfg, video_path, pose_weight, pose_cfg
from config import config

tensor = torch.FloatTensor


class HumanDetection:
    def __init__(self, show_img=True):
        self.object_detector = ObjectDetectionYolo(cfg=yolo_cfg, weight=yolo_weight)
        self.object_tracker = ObjectTracker()
        self.pose_estimator = PoseEstimator(pose_cfg=pose_cfg, pose_weight=pose_weight)
        self.BBV = BBoxVisualizer()
        self.KPV = KeyPointVisualizer()
        self.IDV = IDVisualizer()
        self.boxes = tensor([])
        self.boxes_scores = tensor([])
        self.img_black = np.array([])
        self.frame = np.array([])
        self.id2bbox = {}
        self.kps = {}
        self.kps_score = {}
        self.show_img = show_img
        self.CNN_model = CNNInference()

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
        img_black = cv2.imread('src/black.jpg')
        if config.plot_bbox and self.boxes is not None:
            self.BBV.visualize(self.boxes, self.frame)
        if config.plot_kps and self.kps is not []:
            self.frame = self.KPV.vis_ske(self.frame, self.kps, self.kps_score)
            img_black = self.KPV.vis_ske_black(self.frame, self.kps, self.kps_score)
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
                boxes = self.object_tracker.id_and_box(self.id2bbox)

                inps, pt1, pt2 = crop_bbox(frame, boxes)
                kps, kps_score, kps_id = self.pose_estimator.process_img(inps, boxes, pt1, pt2)
                self.kps, self.kps_score = self.object_tracker.match_kps(kps_id, kps, kps_score)

        return self.kps, self.id2bbox, self.kps_score

    def classify_whole(self, img):
        out = self.CNN_model.predict(img)
        idx = out[0].tolist().index(max(out[0].tolist()))
        pred = CNN_class[idx]
        print("The prediction is {}".format(pred))

    def classify(self, src_img, id2bbox):
        for box in id2bbox.values():
            img = cut_image_with_box(src_img, left=int(box[0]), top=int(box[1]), right=int(box[2]), bottom=int(box[3]))
            out = self.CNN_model.predict(img)
            idx = out[0].tolist().index(max(out[0].tolist()))
            pred = CNN_class[idx]
            text_location = (int((box[0]+box[2])/2)), int((box[1])+50)
            cv2.putText(self.frame, pred, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
        return self.frame


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
                IP.classify_whole(img_black)
                cv2.imshow("res", img)
                img_each = IP.classify(IP.frame, boxes)
                cv2.imshow("each", img_each)
                cv2.waitKey(2)
            else:
                self.cap.release()
                break


if __name__ == '__main__':
    VideoProcessor(video_path).process_video()
