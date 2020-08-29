from config.config import CNN_class
import cv2
from src.utils.img import cut_image_with_box
from .inference import CNNInference


class CNN_Classifier:
    def __init__(self):
        self.CNN_model = CNNInference()
        self.pred = {}

    def classify_whole(self, img):
        out = self.CNN_model.predict(img)
        idx = out[0].tolist().index(max(out[0].tolist()))
        pred = CNN_class[idx]
        print("The prediction is {}".format(pred))

    def classify(self, img_black, id2bbox, frame=None):
        self.pred = {}
        for idx, box in id2bbox.items():
            img = cut_image_with_box(img_black, left=int(box[0]), top=int(box[1]), right=int(box[2]),
                                     bottom=int(box[3]))
            out = self.CNN_model.predict(img)
            idx = out[0].tolist().index(max(out[0].tolist()))
            pred = CNN_class[idx]
            self.pred[idx] = pred
            text_location = (int((box[0] + box[2]) / 2)), int((box[1]) + 50)
            cv2.putText(frame, pred, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
        return frame, self.pred
