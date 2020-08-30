from .sort import Sort
import torch
import cv2
from src.utils.plot import colors, thicks, sizes

tensor = torch.FloatTensor


class ObjectTracker(object):
    def __init__(self):
        self.tracker = Sort()
        self.ids = []
        self.id2box = {}
        self.boxes = []

    def init_tracker(self):
        self.tracker.init_KF()

    def clear(self):
        self.ids = []
        self.id2box = {}
        self.boxes = []

    def track(self, box_res):
        self.clear()
        tracked_box = self.tracker.update(box_res.cpu())
        self.id2box = {int(box[4]): tensor(box[:4]) for box in tracked_box}
        # self.id2box = sorted(tracked_box.items(),key=lambda x:x[0])
        return self.id2box

    def id_and_box(self, tracked_box):
        boxes = sorted(tracked_box.items(), key=lambda x: x[0])
        self.ids = [item[0] for item in boxes]
        self.boxes = [item[1].tolist() for item in boxes]
        return tensor(self.boxes)

    def match_kps(self, kps_id, kps, kps_score):
        id2kps, id2kpScore = {}, {}
        for idx, (kp_id) in enumerate(kps_id):
            id2kps[self.ids[kp_id]] = kps[idx]
            id2kpScore[self.ids[kp_id]] = kps_score[idx]
        return id2kps, id2kpScore

    def get_pred(self):
        return self.tracker.id2pred

    def plot_iou_map(self, img, h_interval=40, w_interval=80):
        iou_matrix = self.tracker.mat
        match_pairs = [(pair[0], pair[1]) for pair in self.tracker.match_indices]
        if len(iou_matrix) > 0:
            for h_idx, h_item in enumerate(iou_matrix):
                if h_idx == 0:
                    color = colors["purple"]
                    cv2.line(img, (0, 25), (img.shape[1], 25), color, thicks["line"])

                for w_idx, item in enumerate(h_item):
                    if w_idx == 0 or h_idx == 0:
                        color = colors["purple"]
                        if h_idx == 0:
                            cv2.line(img, (80, 0), (80, img.shape[0]), color, thicks["line"])
                    elif (w_idx-1, h_idx-1) in match_pairs:
                        color = colors["red"]
                    else:
                        color = colors["yellow"]
                    cv2.putText(img, item, (-60+w_interval*w_idx, 20+h_interval*h_idx), cv2.FONT_HERSHEY_PLAIN,
                                sizes["table"], color, thicks["table"])
        return img
