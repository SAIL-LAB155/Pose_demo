try:
    from config.config import RNN_frame_length
except:
    from src.debug.config.cfg_multi_detections import RNN_frame_length

class Keypoint:
    def __init__(self):
        self.kps = []
        self.enough = False

    def append(self, kps):
        self.kps.append(kps)
        if len(self) < RNN_frame_length:
            self.enough = False
        else:
            self.kps = self.kps[-RNN_frame_length:]
            self.enough = True

    def __len__(self):
        return len(self.kps)

    def clear(self):
        self.kps = []
        self.enough = False
