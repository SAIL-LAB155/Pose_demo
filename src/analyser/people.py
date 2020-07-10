from .box import Box
from .keypoint import Keypoint


class People:
    def __init__(self, idx, box, kps):
        self.id = idx
        self.BOX = Box()
        self.BOX.append(box)
        self.KPS = Keypoint()
        self.KPS.append(kps)
        self.pred = []

    def __len__(self):
        return len(self.BOX.box), len(self.KPS.kps)

    def clear(self):
        self.BOX = Box()
        self.KPS = Keypoint()
