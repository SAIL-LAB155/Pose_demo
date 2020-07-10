

class Keypoint:
    def __init__(self):
        self.kps = []

    def append(self, kps):
        self.kps.append(kps)

    def __len__(self):
        return len(self.kps)

    def clear(self):
        self.kps = []
