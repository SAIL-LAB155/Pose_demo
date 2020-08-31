# from .BiLSTM.test_BiLSTM import BiLSTMPredictor
# from .ConvGRU.test_ConvGRU import ConvGRUPredictor
from .TCN.test_TCN import TCNPredictor
# from .ConvLSTM.test_ConvLstm import ConvLSTMPredictor
# try:
#     from .LSTM.test_LSTM import LSTMPredictor
#     lstm = True
# except:
#     lstm = False

import cv2
import numpy as np
from src.utils.plot import colors, sizes, thicks

try:
    from config.config import RNN_backbone, RNN_class, RNN_weight
except:
    from src.debug.config.cfg_multi_detections import RNN_backbone, RNN_class, RNN_weight


class RNNInference:
    def __init__(self, model_path=RNN_weight):
        self.tester = self.__get_tester(model_path)

    def __get_tester(self, model):
        # if "ConvLSTM" == RNN_backbone:
        #     return ConvLSTMPredictor(model, len(RNN_class))
        # if "BiLSTM" == RNN_backbone:
        #     return BiLSTMPredictor(model, len(RNN_class))
        # if "ConvGRU" == RNN_backbone:
        #     return ConvGRUPredictor(model, len(RNN_class))
        # if 'LSTM' == RNN_backbone:
        #     if lstm:
        #         return LSTMPredictor(model)
        #     else:
        #         print("lstm is not usable")
        if "TCN" == RNN_backbone:
            return TCNPredictor(model, len(RNN_class))

    def predict_pos(self, inp):
        pred = self.tester.predict(np.array(inp).astype(np.float32))
        return pred

    def predict_class(self, inp):
        output = self.predict_pos(inp)
        pred = output.data.max(1, keepdim=True)[1]
        return pred

    def predict_action(self, inp):
        pred = self.predict_class(inp)
        return RNN_class[pred]

    def vis_RNN_res(self, n, idx, preds, img):
        cv2.putText(img, "id{}".format(idx), (20 + 140*n, 40), cv2.FONT_HERSHEY_SIMPLEX, thicks["list"],
                    colors["violet"], sizes["list"])
        for i, pred in enumerate(preds):
            cv2.putText(img, "f{}: {}".format(i, pred), (20 + 140 * n, 80 + 40*i), cv2.FONT_HERSHEY_SIMPLEX,
                        thicks["list"], self.vis_color(pred), sizes["list"])

    def vis_color(self, pred):
        if "drown" in pred:
            return colors["red"]
        return colors["silver"]
