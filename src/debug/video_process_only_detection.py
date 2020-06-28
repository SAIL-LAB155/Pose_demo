from ..detector.yolo_detect import ObjectDetectionYolo
from ..detector.image_process_detect import ImageProcessDetection
from ..detector.yolo_asff_detector import ObjectDetectionASFF
from ..detector.visualize import BBoxVisualizer
from config import config
from ..utils.img import gray3D
import torch
import numpy as np
import cv2
import copy


from .config.cfg_only_detections import gray_yolo_cfg, gray_yolo_weights, black_yolo_cfg, black_yolo_weights, video_path
frame_size = (540, 360)


class RegionDetector(object):
    def __init__(self, path):
        self.black_yolo = ObjectDetectionYolo(cfg=black_yolo_cfg, weight=black_yolo_weights)
        self.gray_yolo = ObjectDetectionYolo(cfg=gray_yolo_cfg, weight=gray_yolo_weights)

        self.BBV = BBoxVisualizer()
        self.dip_detection = ImageProcessDetection()
        self.cap = cv2.VideoCapture(path)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)

    def process(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, config.frame_size)
                fgmask = self.fgbg.apply(frame)
                background = self.fgbg.getBackgroundImage()
                diff = cv2.absdiff(frame, background)
                dip_img = copy.deepcopy(frame)
                dip_boxes = self.dip_detection.detect_rect(diff)
                if len(dip_boxes) > 0:
                    dip_img = self.BBV.visualize(dip_boxes, dip_img)

                with torch.no_grad():
                    # black picture
                    enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
                    enhanced = cv2.filter2D(diff, -1, enhance_kernel)
                    black_boxes, black_scores = self.black_yolo.process(enhanced)
                    if len(black_boxes) > 0:
                        enhanced = self.BBV.visualize(black_boxes, enhanced)

                    # gray pics process
                    gray_img = gray3D(frame)
                    gray_boxes, gray_scores = self.gray_yolo.process(gray_img)
                    if len(gray_boxes) > 0:
                        gray_img = self.BBV.visualize(gray_boxes, gray_img)

                dip_img = cv2.resize(dip_img, frame_size)
                cv2.imshow("dip_result", dip_img)
                enhanced = cv2.resize(enhanced, frame_size)
                cv2.imshow("black_result", enhanced)
                gray_img = cv2.resize(gray_img, frame_size)
                cv2.imshow("gray_result", gray_img)

                cnt += 1
                cv2.waitKey(10)
            else:
                self.cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    RD = RegionDetector(video_path)
    RD.process()
