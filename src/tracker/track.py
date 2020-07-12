from .sort import Sort
import torch

tensor = torch.FloatTensor


class ObjectTracker(object):
    def __init__(self):
        self.tracker = Sort()
        self.box = []
        self.tracked_box = []

    def init_tracker(self):
        self.tracker.init_KF()

    def track(self, box_res):
        self.box = box_res
        self.tracked_box = self.tracker.update(box_res.cpu())

    def match(self, kps, kps_score):
        id2box, id2kps, id2kpScore = {}, {}, {}
        for item in self.tracked_box:
            mark1, mark2 = item[0].tolist(), item[1].tolist()
            for j in range(len(self.box)):
                if self.box[j][0].tolist() == mark1 and self.box[j][1].tolist() == mark2:
                    idx = item[4]
                    id2box[idx] = item[:4]
                    id2kps[idx] = kps[j]
                    id2kpScore = kps_score[j]
        return id2kps, id2box, id2kpScore

    def track_box(self):
        return {int(box[4]): tensor(box[:4]) for box in self.tracked_box}

