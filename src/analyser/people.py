from .box import Box
from .keypoint import Keypoint

max_disappear = 10
max_model_pred = 5


class Person:
    def __init__(self, idx, box):
        self.id = idx
        self.BOX = Box(box)
        self.KPS = Keypoint()
        self.disappear = 0
        self.RNN_preds = []
        self.CNN_preds = []

    def clear(self):
        self.KPS = Keypoint()
        self.disappear = 0
        self.RNN_preds = []
        self.CNN_preds = []

    def box_len(self):
        return len(self.BOX)

    def kps_len(self):
        return len(self.KPS)

    def update_disappear(self, flag):
        if flag == 1:
            self.disappear = 0
        elif flag == 0:
            if self.disappear < max_disappear:
                self.disappear += 1

    def update_RNN_pred(self, pred):
        self.RNN_preds.append(pred)
        if len(self.RNN_preds) >= max_model_pred:
            self.RNN_preds = self.RNN_preds[-max_model_pred:]

    def update_CNN_pred(self, pred):
        self.CNN_preds.append(pred)
        if len(self.CNN_preds) >= max_model_pred:
            self.CNN_preds = self.CNN_preds[-max_model_pred:]