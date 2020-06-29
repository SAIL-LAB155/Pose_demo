from .sort import Sort
import torch

Tensor = torch.cuda.FloatTensor


class ObjectTracker(object):
    def __init__(self):
        self.tracker = Sort()

    def init_tracker(self):
        self.tracker.init_KF()

    def track_bbox(self, box_res):
        tracked_bbox = self.tracker.update(box_res.cpu()).tolist()
        return tracked_bbox
