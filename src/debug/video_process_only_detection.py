import torch
import numpy as np
import cv2
import copy
from config import config

try:
    from .detector.yolo_detect import ObjectDetectionYolo
    from .detector.image_process_detect import ImageProcessDetection
    # from .detector.yolo_asff_detector import ObjectDetectionASFF
    from .detector.visualize import BBoxVisualizer
    from .utils.img import gray3D
    from .detector.box_postprocess import crop_bbox, merge_box
    from .tracker.track import ObjectTracker
    from config.config import gray_yolo_cfg, gray_yolo_weights, black_yolo_cfg, black_yolo_weights, video_path
except:
    from src.detector.yolo_detect import ObjectDetectionYolo
    from src.detector.image_process_detect import ImageProcessDetection
    # from src.detector.yolo_asff_detector import ObjectDetectionASFF
    from src.detector.visualize import BBoxVisualizer
    from src.utils.img import gray3D
    from src.detector.box_postprocess import crop_bbox, merge_box
    from src.tracker.track import ObjectTracker
    from src.tracker.visualize import IDVisualizer
    from src.debug.config.cfg_only_detections import gray_yolo_cfg, gray_yolo_weights, black_yolo_cfg, black_yolo_weights, video_path


class ImgProcessor:
    def __init__(self, show_img=True):
        self.black_yolo = ObjectDetectionYolo(cfg=black_yolo_cfg, weight=black_yolo_weights)
        self.gray_yolo = ObjectDetectionYolo(cfg=gray_yolo_cfg, weight=gray_yolo_weights)
        self.BBV = BBoxVisualizer()
        self.object_tracker = ObjectTracker()
        self.dip_detection = ImageProcessDetection()
        self.BBV = BBoxVisualizer()
        self.IDV = IDVisualizer(with_bbox=False)
        self.img = []
        self.img_black = []
        self.show_img = show_img

    def process_img(self, frame, background):
        black_boxes, black_scores, gray_boxes, gray_scores = None, None, None, None
        diff = cv2.absdiff(frame, background)

        dip_img = copy.deepcopy(frame)
        dip_boxes = self.dip_detection.detect_rect(diff)
        # if len(dip_boxes) > 0:
        #     dip_img = self.BBV.visualize(dip_boxes, dip_img)
        dip_results = [dip_img, dip_boxes]

        with torch.no_grad():
            # black picture
            enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
            enhanced = cv2.filter2D(diff, -1, enhance_kernel)
            black_res = self.black_yolo.process(enhanced)
            if black_res is not None:
                black_boxes, black_scores = self.black_yolo.cut_box_score(black_res)
                enhanced = self.BBV.visualize(black_boxes, enhanced)
            black_results = [enhanced, black_boxes, black_scores]

            # gray pics process
            gray_img = gray3D(frame)
            gray_res = self.gray_yolo.process(gray_img)
            if gray_res is not None:
                gray_boxes, gray_scores = self.gray_yolo.cut_box_score(gray_res)
                gray_img = self.BBV.visualize(gray_boxes, gray_img)
            gray_results = [gray_img, gray_boxes, gray_scores]

            # boxes, scores = merge_box(gray_boxes, black_boxes, gray_scores, black_scores)
            # if gray_res is not None:
            #     tracked_object = self.object_tracker.track_box_with_high_conf(gray_res)
            #     gray_img = self.IDV.plot_bbox_id(tracked_object, gray_img)

            # inps, pt1, pt2 = crop_bbox(frame, boxes)
        return gray_results, black_results, dip_results


frame_size = (540, 360)
IP = ImgProcessor()


class RegionDetector(object):
    def __init__(self, path):

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

                gray_res, black_res, dip_res = IP.process_img(frame, background)

                # dip_img = cv2.resize(dip_res[0], frame_size)
                # cv2.imshow("dip_result", dip_img)
                enhanced = cv2.resize(black_res[0], frame_size)
                cv2.imshow("black_result", enhanced)
                gray_img = cv2.resize(gray_res[0], frame_size)
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
