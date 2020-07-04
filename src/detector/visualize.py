import cv2


class BBoxVisualizer(object):
    def __init__(self):
        self.color = (0, 0, 255)

    def visualize(self, bboxes, img):
        if isinstance(bboxes, dict):
            bboxes = self.__dict2ls(bboxes)

        for bbox in bboxes:
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.color, 4)
        return img

    def __dict2ls(self, id2box):
        return [v for k, v in id2box.items()]