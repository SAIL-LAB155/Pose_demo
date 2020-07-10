

class Box:
    def __init__(self):
        self.box = []

    def append(self, box):
        self.box.append(box)

    def __len__(self):
        return len(self.box)

    def clear(self):
        self.box = []