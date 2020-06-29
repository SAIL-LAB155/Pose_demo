from .sort import Sort
import torch

Tensor = torch.cuda.FloatTensor


class ObjectTracker(object):
    def __init__(self):
        self.tracker = Sort()
        self.bboxes = []
        self.tracked_bbox = []

    def init_tracker(self):
        self.tracker.init_KF()

    def __track_bbox(self, box_res):
        box_tensor = Tensor([box + [0.999, 0.999, 0] for box in self.bboxes])
        self.tracked_bbox = self.tracker.update(box_tensor.cpu()).tolist()
        id2bbox = {int(box[4]): [box[0], box[1], box[2], box[3]] for box in self.tracked_bbox}
        return id2bbox
