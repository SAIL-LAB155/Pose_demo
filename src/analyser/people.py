from .box import Box
from .keypoint import Keypoint


class Person:
    def __init__(self, idx, box, kps):
        self.id = idx
        self.BOX = Box()
        self.BOX.append(box)
        self.KPS = Keypoint()
        self.KPS.append(kps)
        self.pred = []
        self.disappear = 0
        self.img = []

    def box_len(self):
        return len(self.BOX.box)

    def kps_len(self):
        return len(self.KPS.kps)

    def clear(self):
        self.BOX = Box()
        self.KPS = Keypoint()
        self.disappear = 0

