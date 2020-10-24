import torch
try:
    import src.debug.config.cfg as config
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
        self.kps = {}
        self.kps_score = {}
        self.show_img = show_img
        self.resize_size = resize_size

    def init(self):
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


resize_ratio = config.resize_ratio
show_size = config.show_size


class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.resize_size = (int(self.width * resize_ratio), int(self.height * resize_ratio))
        self.IP = HumanDetection(self.resize_size)

    def process_video(self):
        cnt = 0
        self.IP.init()
        while True:
            ret, frame = self.cap.read()
            cnt += 1
            if ret:
                frame = cv2.resize(frame, self.resize_size)
                kps, boxes, kps_score = self.IP.process_img(frame)
                img, img_black = self.IP.visualize()
                cv2.imshow("res", cv2.resize(img, show_size))
                cv2.waitKey(2)

            else:
                self.cap.release()
                break

from threading import Thread
import time
from queue import Queue
#https://github.com/Kjue/python-opencv-gpu-video


class VideoProcessorThread:
    def __init__(self, path, queueSize=3000):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.resize_size = (int(self.width * resize_ratio), int(self.height * resize_ratio))
        self.IP = HumanDetection(self.resize_size)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def update(self):
        # keep looping infinitely
        self.IP.init()
        # IP.object_tracker.init_tracker()
        cnt = 0
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.cap.read()
                start = time.time()
                if grabbed:
                    frame = cv2.resize(frame, self.resize_size)
                    kps, boxes, kps_score = self.IP.process_img(frame)
                    img, img_black = self.IP.visualize()
                    cv2.imshow("res", cv2.resize(img, show_size))
                    cv2.waitKey(2)
                    # out.write(res)
                    all_time = time.time() - start
                    print("time is:", all_time)
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                # add the frame to the queue
                self.Q.put(frame)
            else:
                self.Q.queue.clear()


if __name__ == '__main__':
    #VideoProcessor(config.video_path).process_video()
    VideoProcessorThread(config.video_path).update()
