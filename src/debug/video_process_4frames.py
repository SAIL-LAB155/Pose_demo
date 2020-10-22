import torch
try:
    import src.debug.config.cfg_4frame as config
except:
    import config.config as config
import cv2
import numpy as np
from src.estimator.pose_estimator import PoseEstimator
from src.estimator.visualize import KeyPointVisualizer
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.visualize import BBoxVisualizer
from src.tracker.track import ObjectTracker
from src.tracker.visualize import IDVisualizer
from src.utils.img import calibration
from src.detector.box_postprocess import crop_bbox, eliminate_nan, filter_box
import time


tensor = torch.FloatTensor


class ImgProcessor4Yolo:
    def __init__(self, resize_size, show_img=True):
        self.object_detector = ObjectDetectionYolo(cfg=config.yolo_cfg, weight=config.yolo_weight)
        self.pose_estimator = PoseEstimator(pose_cfg=config.pose_cfg, pose_weight=config.pose_weight)
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
        self.resize_size = resize_size

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
        img_black = np.full((self.resize_size[1], self.resize_size[0], 3), 0).astype(np.uint8)
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


class ImgProcessor:
    def __init__(self, resize_size, show_img=True):
        self.object_detector = ObjectDetectionYolo(cfg=config.yolo_cfg, weight=config.yolo_weight)
        self.pose_estimator = PoseEstimator(pose_cfg=config.pose_cfg, pose_weight=config.pose_weight)
        self.object_trackers = [ObjectTracker() for k in range(4)]
        self.BBV = BBoxVisualizer()
        self.KPV = KeyPointVisualizer()
        self.IDV = IDVisualizer()
        self.show_img = show_img
        self.resize_size = resize_size

    def init_sort(self):
        for trackers in self.object_trackers:
            trackers.init_tracker()

    def visualize(self, img, boxes=None, box_scores=None, kps=None, kps_scores=None, id2box=None):
        img_black = np.full((self.resize_size[1], self.resize_size[0], 3), 0).astype(np.uint8)
        if config.plot_bbox and boxes is not None:
            self.BBV.visualize(boxes, img, box_scores)
            # cv2.imshow("cropped", (torch_to_im(inps[0]) * 255))
        if config.plot_kps and kps is not []:
            self.KPV.vis_ske(img, kps, kps_scores)
            self.KPV.vis_ske_black(img_black, kps, kps_scores)
        if config.plot_id and id2box is not None:
            self.IDV.plot_bbox_id(id2box, img)
            # frame = self.IDV.plot_skeleton_id(id2ske, copy.deepcopy(img))
        return img, img_black

    def process_img(self, fr1, fr2, fr3, fr4):
        fr1, fr2, fr3, fr4 = calibration(fr1), calibration(fr2), calibration(fr3), calibration(fr4)
        results = []
        frames = np.vstack((np.expand_dims(fr1, axis=0), np.expand_dims(fr2, axis=0), np.expand_dims(fr3, axis=0),
                                 np.expand_dims(fr4, axis=0)))

        all_boxes = self.object_detector.process(frames)
        for cam_idx, tracker in enumerate(self.object_trackers):
            kps, kps_score, boxes, id2bbox = {}, {}, [], {}
            box_idx = [idx for idx, item in enumerate(all_boxes) if item[0] == cam_idx]
            box_res = all_boxes[box_idx][:,1:]

            boxes, boxes_scores = self.object_detector.cut_box_score(box_res)
            boxes, boxes_scores, box_res = filter_box(boxes, boxes_scores, box_res, config.yolo_threshold)
            if box_res is not None:
                id2bbox = eliminate_nan(tracker.track(box_res))
                boxes = tracker.id_and_box(id2bbox)
                inps, pt1, pt2 = crop_bbox(frames[cam_idx], boxes)
                if inps is not None:
                    kps, kps_score, kps_id = self.pose_estimator.process_img(inps, boxes, pt1, pt2)
                    kps, kps_score = tracker.match_kps(kps_id, kps, kps_score)

            img, black_img = self.visualize(frames[cam_idx], boxes, boxes_scores, kps, kps_score, id2bbox)
            results.append([img, black_img, kps, kps_score, id2bbox])

        return results[0], results[1], results[2], results[3]


resize_ratio = config.resize_ratio
show_size = config.show_size


class VideoProcessor:
    def __init__(self, vp1, vp2, vp3, vp4, show_img=True):
        self.cap1 = cv2.VideoCapture(vp1)
        self.cap2 = cv2.VideoCapture(vp2)
        self.cap3 = cv2.VideoCapture(vp3)
        self.cap4 = cv2.VideoCapture(vp4)

        self.height, self.width = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.resize_size = (int(self.width * resize_ratio), int(self.height * resize_ratio))
        self.show_img = show_img
        self.IP = ImgProcessor4Yolo(self.resize_size)

    def process_frame(self, f1, f2, f3, f4):
        f1, f2, f3, f4 = cv2.resize(f1, self.resize_size), cv2.resize(f2, self.resize_size), \
                             cv2.resize(f3, self.resize_size), cv2.resize(f4, self.resize_size),

        res1, res2, res3, res4 = self.IP.process_img(f1, f2, f3, f4)
        img1, img2, img3, img4 = res1[0], res2[0], res3[0], res4[0]

        img_ver1 = np.concatenate((img1, img2), axis=0)
        img_ver2 = np.concatenate((img3, img4), axis=0)
        img = np.concatenate((img_ver1, img_ver2), axis=1)
        return img

    def process_video(self):
        cnt = 0
        while True:
            begin_time = time.time()
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            ret3, frame3 = self.cap3.read()
            ret4, frame4 = self.cap4.read()
            cnt += 1

            if ret1:
                img = self.process_frame(frame1, frame2, frame3, frame4)
                # img = cv2.resize(img, frame_size)
                print("Time used : {}".format(time.time() - begin_time))
                if self.show_img:
                    cv2.imshow("res", cv2.resize(img, show_size))
                    cv2.waitKey(2)

            else:
                self.cap1.release()
                self.cap2.release()
                self.cap3.release()
                self.cap4.release()
                break


if __name__ == '__main__':
    VideoProcessor(config.video_1, config.video_2, config.video_3, config.video_4).process_video()
