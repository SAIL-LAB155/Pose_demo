import torch
import numpy as np
import cv2
import copy
from config import config
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.image_process_detect import ImageProcessDetection
# from src.detector.yolo_asff_detector import ObjectDetectionASFF
from src.detector.visualize import BBoxVisualizer
from src.estimator.pose_estimator import PoseEstimator
from src.estimator.visualize import KeyPointVisualizer
from src.utils.img import gray3D
from src.detector.box_postprocess import crop_bbox, filter_box, BoxEnsemble
from src.tracker.track import ObjectTracker
from src.tracker.visualize import IDVisualizer
from src.analyser.area import RegionProcessor
from src.analyser.humans import HumanProcessor
from src.utils.utils import paste_box
from src.RNNclassifier.classify import RNNInference

try:
    from config.config import gray_yolo_cfg, gray_yolo_weights, black_yolo_cfg, black_yolo_weights, video_path, \
        black_box_threshold, gray_box_threshold, pose_cfg, pose_weight
except:
    from src.debug.config.cfg_multi_detections import gray_yolo_cfg, gray_yolo_weights, black_yolo_cfg,\
        black_yolo_weights, video_path, black_box_threshold, gray_box_threshold, pose_cfg, pose_weight

fourcc = cv2.VideoWriter_fourcc(*'XVID')
empty_tensor = torch.empty([0,7])
empty_tensor4 = torch.empty([0,4])


class ImgProcessor:
    def __init__(self, show_img=True):
        self.black_yolo = ObjectDetectionYolo(cfg=black_yolo_cfg, weight=black_yolo_weights)
        self.gray_yolo = ObjectDetectionYolo(cfg=gray_yolo_cfg, weight=gray_yolo_weights)
        self.object_tracker = ObjectTracker()
        self.dip_detection = ImageProcessDetection()
        self.RNN_model = RNNInference()
        self.pose_estimator = PoseEstimator(pose_cfg=pose_cfg, pose_weight=pose_weight)
        self.KPV = KeyPointVisualizer()
        self.BBV = BBoxVisualizer()
        self.IDV = IDVisualizer(with_bbox=False)
        self.img = []
        self.id2bbox = {}
        self.img_black = []
        self.show_img = show_img
        self.RP = RegionProcessor(config.frame_size[0], config.frame_size[1], 10, 10)
        self.HP = HumanProcessor(config.frame_size[0], config.frame_size[1])
        self.BE = BoxEnsemble()
        self.kps = {}
        self.kps_score = {}

    def process_img(self, frame, background):
        rgb_kps, dip_img = copy.deepcopy(frame), copy.deepcopy(frame)
        img_black = cv2.imread("src/black.jpg")
        img_black = cv2.resize(img_black, config.frame_size)
        iou_img, black_kps, img_cnt = copy.deepcopy(img_black), copy.deepcopy(img_black), copy.deepcopy(img_black)

        black_boxes, black_scores, gray_boxes, gray_scores = empty_tensor, empty_tensor, empty_tensor, empty_tensor
        diff = cv2.absdiff(frame, background)
        dip_boxes = self.dip_detection.detect_rect(diff)
        dip_results = [dip_img, dip_boxes]

        with torch.no_grad():
            # black picture
            enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
            enhanced = cv2.filter2D(diff, -1, enhance_kernel)
            black_res = self.black_yolo.process(enhanced)
            if black_res is not None:
                black_boxes, black_scores = self.black_yolo.cut_box_score(black_res)
                enhanced = self.BBV.visualize(black_boxes, enhanced, black_scores)
                black_boxes, black_scores, black_res = \
                    filter_box(black_boxes, black_scores, black_res, black_box_threshold)
            black_results = [enhanced, black_boxes, black_scores]

            # gray pics process
            gray_img = gray3D(frame)
            gray_res = self.gray_yolo.process(gray_img)
            if gray_res is not None:
                gray_boxes, gray_scores = self.gray_yolo.cut_box_score(gray_res)
                gray_img = self.BBV.visualize(gray_boxes, gray_img, gray_scores)
                gray_boxes, gray_scores, gray_res = \
                    filter_box(gray_boxes, gray_scores, gray_res, gray_box_threshold)
            gray_results = [gray_img, gray_boxes, gray_scores]

            merged_res = self.BE.ensemble_box(black_res, gray_res)

            if len(merged_res) > 0:
                # merged_boxes, merged_scores = self.gray_yolo.cut_box_score(merged_res)
                self.id2bbox = self.object_tracker.track(merged_res)
                boxes = self.object_tracker.id_and_box(self.id2bbox)
                self.IDV.plot_bbox_id(self.id2bbox, frame)
                img_black = paste_box(rgb_kps, img_black, boxes)
                self.HP.update(self.id2bbox)
            else:
                boxes = empty_tensor4

            iou_img = self.object_tracker.plot_iou_map(iou_img)
            img_black = paste_box(rgb_kps, img_black, boxes)
            self.HP.update(self.id2bbox)

            rd_map = self.RP.process_box(boxes, frame)
            warning_idx = self.RP.get_alarmed_box_id(self.id2bbox)
            danger_idx = self.HP.box_size_warning(warning_idx)
            box_map = self.HP.vis_box_size(img_black, img_cnt)

            if danger_idx:
                danger_id2box = {k:v for k,v in self.id2bbox.items() if k in danger_idx}
                danger_box = self.object_tracker.id_and_box(danger_id2box)
                inps, pt1, pt2 = crop_bbox(rgb_kps, danger_box)
                if inps is not None:
                    kps, kps_score, kps_id = self.pose_estimator.process_img(inps, danger_box, pt1, pt2)
                    if self.kps is not []:
                        self.kps, self.kps_score = self.object_tracker.match_kps(kps_id, kps, kps_score)
                        self.HP.update_kps(self.kps)
                        rgb_kps = self.KPV.vis_ske(rgb_kps, kps, kps_score)
                        rgb_kps = self.IDV.plot_skeleton_id(self.kps, rgb_kps)
                        rgb_kps = self.BBV.visualize(danger_box, rgb_kps)
                        black_kps = self.KPV.vis_ske_black(black_kps, kps, kps_score)
                        black_kps = self.IDV.plot_skeleton_id(self.kps, black_kps)

                        for n, idx in enumerate(self.kps.keys()):
                            if self.HP.if_enough_kps(idx):
                                RNN_res = self.RNN_model.predict_action(self.HP.obtain_kps(idx))
                                self.HP.update_RNN(idx, RNN_res)
                                self.RNN_model.vis_RNN_res(n, idx, self.HP.get_RNN_preds(idx), black_kps)

            iou_img = cv2.resize(iou_img, frame_size)
            detection_map = np.concatenate((enhanced, gray_img), axis=1)
            yolo_cnt_map = np.concatenate((detection_map, rd_map), axis=0)
            yolo_map = np.concatenate((yolo_cnt_map, box_map), axis=1)
            kps_img = np.concatenate((iou_img, rgb_kps, black_kps), axis=1)
            res = np.concatenate((yolo_map, kps_img), axis=0)

        return gray_results, black_results, dip_results, res


IP = ImgProcessor()
enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
frame_size = (720, 540)
store_size = (frame_size[0]*3, frame_size[1]*3)
write_video = True


class DrownDetector:
    def __init__(self, vp):
        self.cap = cv2.VideoCapture(vp)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if write_video:
            self.out_video = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'XVID'), 15, store_size)

    def process_video(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, frame_size)
                fgmask = self.fgbg.apply(frame)
                background = self.fgbg.getBackgroundImage()
                diff = cv2.absdiff(frame, background)
                gray_res, black_res, dip_res, res_map = IP.process_img(frame, background)
                if write_video:
                    self.out_video.write(res_map)
                cv2.imshow("res", cv2.resize(res_map, (1440, 840)))
                # out.write(res)
                cnt += 1
                cv2.waitKey(1)
            else:
                self.cap.release()
                break


if __name__ == '__main__':
    DrownDetector(video_path).process_video()

